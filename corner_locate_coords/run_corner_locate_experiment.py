#!/usr/bin/env python3
"""
run_corner_locate_experiment.py — Coordinate-regression experiment for SEM corner localization.

Companion to corner_judge_categorical/run_fewshot_experiment.py. Same direct-vLLM
discipline (no yJarvis facade), same ladder pattern, same rigor (Wilson CI, repeats,
fixed exemplar order). The TASK is different:

  - Exemplar images carry a GREEN crosshair drawn at the true corner location, plus the
    matching (x, y) answer in the assistant turn. They act as visual demonstrations.
  - Eval images do NOT have the green cross. The VLM must locate the corner from the
    SEM content alone and output (x, y).
  - A RED crosshair may appear in either set; it is the camera-center marker and must
    be ignored.

We grade with multiple tolerance thresholds (hit@1%, hit@2%, hit@5%, hit@10% of
min(W,H)) plus optional BBOX-in, plus L2 stats and a wrong-quadrant analog of the
categorical experiment's wrongDir%.

Environment
-----------
    VLLM_BASE_URL   e.g. http://your-vllm-host:8000/v1
    VLLM_API_KEY    vLLM usually accepts any value (default "EMPTY")
    VLM_MODEL       e.g. Qwen3.6-27B   (or pass --model)

Data layout (--data-dir)
------------------------
    <data-dir>/
        eval_manifest.json       # eval items, NO green cross on images
        exemplar_manifest.json   # exemplar items, GREEN cross drawn at GT corner
        <images...>

Manifest entry schema:
    {
        "image": "f.png",
        "corner_x": 487, "corner_y": 312,      # ground truth pixel coords (top-left origin)
        "bbox": [470, 295, 505, 330]           # OPTIONAL; if present, hit_bbox% is computed
    }

Usage
-----
    python run_corner_locate_experiment.py --data-dir ./sem_loc_eval --repeats 3
    python run_corner_locate_experiment.py --data-dir ./sem_loc_eval \\
        --conditions zeroshot_guided_reasoning_pixel fewshot5_guided_reasoning_pixel
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

# ---------------------------------------------------------------------------
# Prompts & schemas
# ---------------------------------------------------------------------------

_TASK_RULES = """任務規則:
- 「角」= 兩條亮邊垂直相交的 L 形頂點(corner of the SEM target structure)。
- 範例影像會在角的真實位置畫有「綠色」十字,assistant 回答給出該綠色十字中心的 (x, y);這些是視覺示範。
- 測試影像不會有綠色十字,你必須只依靠 SEM 影像內容判斷角的位置,並以同格式輸出 (x, y) 像素座標。
- 影像中可能出現「紅色」十字,代表機台 camera 中心,**與本任務無關,請完全忽略**。
- 影像座標 (0,0) 為左上角;x 向右為正,y 向下為正。
- 找不到角(移出視野/失焦/雜訊蓋過):corner_found=false,x 與 y 填 0,aligned 概念在此實驗不適用。
- 數值精度:像素整數 (mode=pixel) 或 0..1000 整數正規化座標 (mode=norm1000)。"""

_FORMAT_PIXEL = (
    "座標格式:像素整數。影像尺寸為 {W}×{H};輸出的 x ∈ [0, {Wm1}],y ∈ [0, {Hm1}]。"
)
_FORMAT_NORM = (
    "座標格式:正規化整數。**不論影像尺寸**,輸出的 x、y ∈ [0, 1000](0=左/上,1000=右/下)。"
    "我會在程式端把 1000 對應到影像寬高再換算回像素。"
)

_OUTPUT_REASONING = (
    "輸出規則:只輸出一個 JSON 物件,reasoning 欄位放最前面(先描述看到的角結構、十字線/marker、"
    "角相對影像的方位,再決定座標),不得有任何額外文字或 markdown。"
)
_OUTPUT_NO_REASONING = "輸出規則:只輸出一個 JSON 物件,不得有任何額外文字或 markdown。"


def system_prompt(coord_mode: str, reasoning: bool, W: int, H: int) -> str:
    fmt = (_FORMAT_PIXEL.format(W=W, H=H, Wm1=W - 1, Hm1=H - 1)
           if coord_mode == "pixel" else _FORMAT_NORM)
    out = _OUTPUT_REASONING if reasoning else _OUTPUT_NO_REASONING
    return (
        "你是半導體 SEM 對齊的視覺座標模型,輸出影像中目標結構「角」的 (x, y) 座標。\n"
        + _TASK_RULES + "\n" + fmt + "\n" + out
    )


USER_TURN_TEXT = "請輸出這張 SEM 影像中,目標結構角的 (x, y) 座標,依 schema 輸出。"


def _schema(coord_mode: str, reasoning: bool) -> Dict[str, Any]:
    if coord_mode == "pixel":
        x_prop = {"type": "integer", "minimum": 0}
        y_prop = {"type": "integer", "minimum": 0}
    else:  # norm1000
        x_prop = {"type": "integer", "minimum": 0, "maximum": 1000}
        y_prop = {"type": "integer", "minimum": 0, "maximum": 1000}
    core = {"corner_found": {"type": "boolean"}, "x": x_prop, "y": y_prop}
    if reasoning:
        props = {"reasoning": {"type": "string"}, **core}
        req = ["reasoning", *core.keys()]
    else:
        props = dict(core)
        req = list(core.keys())
    return {"type": "object", "properties": props, "required": req, "additionalProperties": False}


# ---------------------------------------------------------------------------
# Condition ladder
# ---------------------------------------------------------------------------

CONDITIONS: List[Dict[str, Any]] = [
    # baseline reads
    {"name": "zeroshot_freetext_pixel",          "k": 0, "guided": False, "reasoning": True,  "coord": "pixel"},
    {"name": "zeroshot_guided_pixel",            "k": 0, "guided": True,  "reasoning": False, "coord": "pixel"},
    {"name": "zeroshot_guided_reasoning_pixel",  "k": 0, "guided": True,  "reasoning": True,  "coord": "pixel"},
    # the real lever: visual-demo few-shot
    {"name": "fewshot3_guided_reasoning_pixel",  "k": 3, "guided": True,  "reasoning": True,  "coord": "pixel"},
    {"name": "fewshot5_guided_reasoning_pixel",  "k": 5, "guided": True,  "reasoning": True,  "coord": "pixel"},
    # coord-format ablation, matched to the best few-shot setup
    {"name": "fewshot5_guided_reasoning_norm",   "k": 5, "guided": True,  "reasoning": True,  "coord": "norm1000"},
    # reasoning x few-shot interaction
    {"name": "fewshot5_guided_noreasoning_pixel","k": 5, "guided": True,  "reasoning": False, "coord": "pixel"},
]
_CONDITION_NAMES = {c["name"] for c in CONDITIONS}

# Tolerance fractions for hit@τ; τ is fraction of min(W,H).
DEFAULT_TOLERANCES = (0.01, 0.02, 0.05, 0.10)

# Retryable transport errors (same set as the categorical experiment)
_RETRYABLE_ERROR_NAMES = {
    "APITimeoutError", "APIConnectionError", "InternalServerError",
    "RateLimitError", "ConnectionError", "Timeout", "ReadTimeout",
}

# ---------------------------------------------------------------------------
# vLLM transport
# ---------------------------------------------------------------------------

_CLIENT = None

def _client():
    global _CLIENT
    if _CLIENT is None:
        if OpenAI is None:
            raise RuntimeError("openai package not installed: pip install openai")
        _CLIENT = OpenAI(
            base_url=os.environ["VLLM_BASE_URL"],
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        )
    return _CLIENT


def _structured_kwargs(schema: Optional[Dict[str, Any]], mechanism: str) -> Dict[str, Any]:
    if schema is None:
        return {}
    if mechanism == "response_format":
        return {"response_format": {"type": "json_schema",
                                    "json_schema": {"name": "result", "schema": schema, "strict": True}}}
    return {"extra_body": {"guided_json": schema}}


def call_vllm(messages: List[Dict[str, Any]], schema: Optional[Dict[str, Any]],
              model: str, temperature: float, max_tokens: int, timeout: int,
              mechanism: str, retries: int = 2) -> Tuple[str, float, Optional[str]]:
    t0 = time.time()
    last_err: Optional[str] = None
    backoffs = [1.0, 4.0, 16.0]
    for attempt in range(retries + 1):
        try:
            completion = _client().chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=max_tokens, timeout=timeout,
                **_structured_kwargs(schema, mechanism),
            )
            content = completion.choices[0].message.content or ""
            return content, (time.time() - t0) * 1000.0, None
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if attempt >= retries or type(e).__name__ not in _RETRYABLE_ERROR_NAMES:
                break
            time.sleep(backoffs[min(attempt, len(backoffs) - 1)])
    return "", (time.time() - t0) * 1000.0, last_err


# ---------------------------------------------------------------------------
# Data / message building
# ---------------------------------------------------------------------------

_URI_CACHE: Dict[str, str] = {}
_DIMS_CACHE: Dict[str, Tuple[int, int]] = {}


def _data_uri(img_path: str) -> str:
    with Image.open(img_path) as raw:
        img = raw.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _cached_data_uri(img_path: str) -> str:
    uri = _URI_CACHE.get(img_path)
    if uri is None:
        uri = _data_uri(img_path)
        _URI_CACHE[img_path] = uri
    return uri


def _image_dims(img_path: str) -> Tuple[int, int]:
    dims = _DIMS_CACHE.get(img_path)
    if dims is None:
        with Image.open(img_path) as im:
            dims = im.size  # (W, H)
        _DIMS_CACHE[img_path] = dims
    return dims


def _image_content(img_path: str, cache: bool = False) -> List[Dict[str, Any]]:
    uri = _cached_data_uri(img_path) if cache else _data_uri(img_path)
    return [{"type": "text", "text": USER_TURN_TEXT},
            {"type": "image_url", "image_url": {"url": uri}}]


def load_manifest(data_dir: str, name: str) -> List[Dict[str, Any]]:
    path = os.path.join(data_dir, name)
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    for e in entries:
        e["_img_path"] = os.path.join(data_dir, e["image"])
        if not os.path.isfile(e["_img_path"]):
            raise FileNotFoundError(f"image not found: {e['_img_path']}")
        # GT coords are required for both eval and exemplar (exemplar uses them as the answer).
        if "corner_x" not in e or "corner_y" not in e:
            raise KeyError(f"manifest entry missing corner_x/corner_y: {e['image']}")
    return entries


def _exemplar_answer(ex: Dict[str, Any], coord_mode: str, reasoning: bool) -> Dict[str, Any]:
    """Build the assistant-turn JSON answer for an exemplar.

    Exemplars must declare GT coords; we render them into the coord_mode the model is
    being asked to use (pixel = raw, norm1000 = scaled by image dims). If the manifest
    has a 'reasoning' string we pass it through; otherwise we synthesize a minimal one
    so few-shot still shows the reasoning-first convention.
    """
    W, H = _image_dims(ex["_img_path"])
    if coord_mode == "pixel":
        x_val, y_val = int(ex["corner_x"]), int(ex["corner_y"])
    else:
        x_val = max(0, min(1000, round(1000 * ex["corner_x"] / max(1, W - 1))))
        y_val = max(0, min(1000, round(1000 * ex["corner_y"] / max(1, H - 1))))
    ans: Dict[str, Any] = {"corner_found": True, "x": x_val, "y": y_val}
    if reasoning:
        r = ex.get("reasoning") or "影像中綠色十字標示角的位置;依其中心輸出座標。"
        ans = {"reasoning": r, **ans}
    return ans


def build_messages(cond: Dict[str, Any], item: Dict[str, Any],
                   exemplars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    W, H = _image_dims(item["_img_path"])
    system = system_prompt(cond["coord"], cond["reasoning"], W, H)
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    for ex in exemplars[: cond["k"]]:
        ans = _exemplar_answer(ex, cond["coord"], cond["reasoning"])
        msgs.append({"role": "user", "content": _image_content(ex["_img_path"], cache=True)})
        msgs.append({"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)})
    msgs.append({"role": "user", "content": _image_content(item["_img_path"])})
    return msgs


# ---------------------------------------------------------------------------
# Parsing & scoring
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        start = raw.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def parse_prediction(content: str, coord_mode: str) -> Optional[Dict[str, Any]]:
    obj = _extract_json(content)
    if not isinstance(obj, dict):
        return None
    try:
        cf, x, y = obj["corner_found"], obj["x"], obj["y"]
    except KeyError:
        return None
    if not isinstance(cf, bool):
        return None
    if not (isinstance(x, int) and isinstance(y, int)):
        # Some models emit floats; coerce if integral, else reject.
        try:
            xf = float(x); yf = float(y)
            if not (xf.is_integer() and yf.is_integer()):
                return None
            x, y = int(xf), int(yf)
        except (TypeError, ValueError):
            return None
    if coord_mode == "norm1000" and not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    if coord_mode == "pixel" and (x < 0 or y < 0):
        return None
    return {"corner_found": cf, "x": x, "y": y}


def _to_pixels(pred: Dict[str, Any], coord_mode: str, W: int, H: int) -> Tuple[int, int]:
    if coord_mode == "pixel":
        return pred["x"], pred["y"]
    # norm1000 -> pixels
    return (round(pred["x"] * (W - 1) / 1000), round(pred["y"] * (H - 1) / 1000))


def _quadrant(x: int, y: int, W: int, H: int) -> Tuple[int, int]:
    """Sign of position relative to image center, with a dead-band of 5% to avoid noise."""
    cx, cy = W / 2.0, H / 2.0
    band_x = 0.05 * W
    band_y = 0.05 * H
    sx = 0 if abs(x - cx) <= band_x else (1 if x > cx else -1)
    sy = 0 if abs(y - cy) <= band_y else (1 if y > cy else -1)
    return sx, sy


def score_one(gt: Dict[str, Any], pred: Optional[Dict[str, Any]], coord_mode: str,
              W: int, H: int, tolerances: Tuple[float, ...]) -> Dict[str, Any]:
    """Per-item flags. Hits/L2 are computed in pixel space using the smaller image dim
    as the normalization base, so percentages are transferable across resolutions.
    """
    found_subset = bool(gt["corner_found"])
    out: Dict[str, Any] = {"format_ok": 0 if pred is None else 1,
                           "corner_found_ok": None, "L2_px": None, "L2_norm": None,
                           "hit_bbox": None, "wrong_quadrant": None,
                           "found_subset": found_subset}
    for tau in tolerances:
        out[f"hit@{int(tau * 100)}pct"] = None
    if pred is None:
        # Malformed reply: a missing localization in production is a controller stall.
        # We score it as worst-case on the corner-present subset: no hit at any tau,
        # but L2 is left None (no point estimate to compare).
        if found_subset:
            for tau in tolerances:
                out[f"hit@{int(tau * 100)}pct"] = 0
            out["wrong_quadrant"] = 1  # treat malformed as risk, mirroring categorical experiment
            if gt.get("bbox"):
                out["hit_bbox"] = 0
        out["corner_found_ok"] = 0
        return out
    out["corner_found_ok"] = int(pred["corner_found"] == gt["corner_found"])
    if not found_subset:
        return out  # nothing more to score; predicting on a not-present case is its own metric
    px_x, px_y = _to_pixels(pred, coord_mode, W, H)
    gx, gy = int(gt["corner_x"]), int(gt["corner_y"])
    l2 = math.hypot(px_x - gx, px_y - gy)
    norm = l2 / float(min(W, H))
    out["L2_px"] = l2
    out["L2_norm"] = norm
    for tau in tolerances:
        out[f"hit@{int(tau * 100)}pct"] = int(norm <= tau)
    if gt.get("bbox"):
        x0, y0, x1, y1 = gt["bbox"]
        out["hit_bbox"] = int(x0 <= px_x <= x1 and y0 <= px_y <= y1)
    gq = _quadrant(gx, gy, W, H)
    pq = _quadrant(px_x, px_y, W, H)
    # Both axes flipped (relative to image center) = controller would diverge on both axes.
    out["wrong_quadrant"] = int(gq[0] != 0 and pq[0] != 0 and gq[0] == -pq[0]
                                and gq[1] != 0 and pq[1] != 0 and gq[1] == -pq[1])
    return out


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    if args.conditions:
        unknown = [n for n in args.conditions if n not in _CONDITION_NAMES]
        if unknown:
            print(f"[warn] unknown --conditions ignored: {unknown}. "
                  f"Known: {sorted(_CONDITION_NAMES)}", file=sys.stderr)
    eval_set = load_manifest(args.data_dir, "eval_manifest.json")
    exemplars = load_manifest(args.data_dir, "exemplar_manifest.json")
    if args.limit:
        eval_set = eval_set[: args.limit]
    conds = [c for c in CONDITIONS if (not args.conditions or c["name"] in args.conditions)]
    if not conds:
        print("[abort] no matching conditions to run", file=sys.stderr)
        return
    tolerances = tuple(args.tolerances) if args.tolerances else DEFAULT_TOLERANCES
    model = args.model or os.environ.get("VLM_MODEL", "Qwen3.6-27B")
    os.makedirs(args.out_dir, exist_ok=True)
    raw_rows: List[Dict[str, Any]] = []
    for cond in conds:
        if cond["k"] > len(exemplars):
            print(f"[skip] {cond['name']}: needs {cond['k']} exemplars, only {len(exemplars)} available")
            continue
        print(f"\n=== condition: {cond['name']} (k={cond['k']}, guided={cond['guided']}, "
              f"reasoning={cond['reasoning']}, coord={cond['coord']}) ===")
        for rep in range(args.repeats):
            for item in eval_set:
                W, H = _image_dims(item["_img_path"])
                schema = _schema(cond["coord"], cond["reasoning"]) if cond["guided"] else None
                msgs = build_messages(cond, item, exemplars)
                content, latency, err = call_vllm(
                    msgs, schema, model, args.temperature, args.max_tokens, args.timeout, args.mechanism)
                pred = None if err else parse_prediction(content, cond["coord"])
                gt = {"corner_found": True, "corner_x": item["corner_x"], "corner_y": item["corner_y"]}
                if "bbox" in item:
                    gt["bbox"] = item["bbox"]
                if "corner_found" in item:
                    gt["corner_found"] = bool(item["corner_found"])
                sc = score_one(gt, pred, cond["coord"], W, H, tolerances)
                raw_rows.append({"condition": cond["name"], "repeat": rep, "image": item["image"],
                                 "error": err or "", "latency_ms": round(latency, 1),
                                 "raw": content[:500], "W": W, "H": H, **sc})
            print(f"  repeat {rep + 1}/{args.repeats} done")
    _write_raw_csv(raw_rows, os.path.join(args.out_dir, "raw_results.csv"), tolerances)
    summary = _aggregate(raw_rows, args.repeats, tolerances)
    _print_summary(summary, tolerances)
    _write_summary_md(summary, model, args, tolerances, os.path.join(args.out_dir, "summary.md"))
    print(f"\nWrote: {args.out_dir}/raw_results.csv  and  {args.out_dir}/summary.md")


def _rate(rows: List[Dict[str, Any]], key: str) -> Tuple[int, int]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals), len(vals)


def _aggregate(rows: List[Dict[str, Any]], repeats: int,
               tolerances: Tuple[float, ...]) -> List[Dict[str, Any]]:
    out = []
    conds: List[str] = []
    for r in rows:
        if r["condition"] not in conds:
            conds.append(r["condition"])
    for cname in conds:
        cr = [r for r in rows if r["condition"] == cname]
        fk, fn = _rate(cr, "format_ok")
        wdk, wdn = _rate(cr, "wrong_quadrant")
        bbk, bbn = _rate(cr, "hit_bbox")
        l2s = [r["L2_norm"] for r in cr if r.get("L2_norm") is not None]
        lat = [r["latency_ms"] for r in cr if r.get("latency_ms") is not None]
        hit_rates: Dict[str, Tuple[int, int, Tuple[float, float]]] = {}
        per_rep_hit5: List[float] = []
        for tau in tolerances:
            col = f"hit@{int(tau * 100)}pct"
            k, n = _rate(cr, col)
            lo, hi = wilson(k, n)
            hit_rates[col] = (k, n, (100 * lo, 100 * hi))
        # variance: track headline hit@5pct per repeat
        for rep in range(repeats):
            rr = [r for r in cr if r["repeat"] == rep]
            k, n = _rate(rr, "hit@5pct")
            if n:
                per_rep_hit5.append(100 * k / n)
        mean_rep = sum(per_rep_hit5) / len(per_rep_hit5) if per_rep_hit5 else 0.0
        std_rep = (sum((x - mean_rep) ** 2 for x in per_rep_hit5) / len(per_rep_hit5)) ** 0.5 \
                  if len(per_rep_hit5) > 1 else 0.0
        out.append({
            "condition": cname, "n_calls": len(cr),
            "format_pct": 100 * fk / fn if fn else 0.0,
            "wrong_quadrant_pct": 100 * wdk / wdn if wdn else 0.0,
            "hit_bbox_pct": 100 * bbk / bbn if bbn else float("nan"),
            "mean_L2_pct": 100 * (sum(l2s) / len(l2s)) if l2s else float("nan"),
            "median_L2_pct": 100 * sorted(l2s)[len(l2s) // 2] if l2s else float("nan"),
            "hit_rates": hit_rates,
            "hit5_std": std_rep,
            "latency_ms": sum(lat) / len(lat) if lat else 0.0,
        })
    return out


def _print_summary(summary: List[Dict[str, Any]], tolerances: Tuple[float, ...]) -> None:
    print("\n" + "=" * 130)
    hit_cols = [f"hit@{int(t * 100)}%" for t in tolerances]
    hdr = ["condition", "fmt%", *hit_cols, "hit5±sd", "bbox%", "L2µ%", "L2~%", "wrongQ%", "lat_ms"]
    fmt = "{:<32} {:>5} " + " ".join("{:>7}" for _ in tolerances) + " {:>10} {:>6} {:>5} {:>5} {:>8} {:>7}"
    print(fmt.format(*hdr))
    print("-" * 130)
    for s in summary:
        hits = [f"{100 * s['hit_rates'][c][0] / s['hit_rates'][c][1]:.0f}"
                if s['hit_rates'][c][1] else "n/a"
                for c in (f"hit@{int(t * 100)}pct" for t in tolerances)]
        bbox = f"{s['hit_bbox_pct']:.0f}" if s["hit_bbox_pct"] == s["hit_bbox_pct"] else "n/a"
        l2u = f"{s['mean_L2_pct']:.1f}" if s["mean_L2_pct"] == s["mean_L2_pct"] else "n/a"
        l2m = f"{s['median_L2_pct']:.1f}" if s["median_L2_pct"] == s["median_L2_pct"] else "n/a"
        hit5_col = f"hit@{int(0.05 * 100)}pct"
        h5 = s["hit_rates"].get(hit5_col)
        h5_str = (f"{100 * h5[0] / h5[1]:.0f}±{s['hit5_std']:.1f}"
                  if (h5 and h5[1]) else "n/a")
        print(fmt.format(s["condition"], f"{s['format_pct']:.0f}", *hits,
                         h5_str, bbox, l2u, l2m,
                         f"{s['wrong_quadrant_pct']:.1f}", f"{s['latency_ms']:.0f}"))
    print("=" * 130)
    print("關鍵看 wrongQ%(對角象限預測,閉環會發散)、hit@5%/hit@2%(實際命中率)、L2~%(中位數誤差)。")


def _write_raw_csv(rows: List[Dict[str, Any]], path: str, tolerances: Tuple[float, ...]) -> None:
    if not rows:
        return
    cols = ["condition", "repeat", "image", "W", "H", "format_ok", "corner_found_ok",
            "L2_px", "L2_norm", "hit_bbox", "wrong_quadrant",
            *(f"hit@{int(t * 100)}pct" for t in tolerances),
            "latency_ms", "error", "raw"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_summary_md(summary: List[Dict[str, Any]], model: str, args: argparse.Namespace,
                      tolerances: Tuple[float, ...], path: str) -> None:
    hit_headers = [f"hit@{int(t * 100)}%" for t in tolerances]
    lines = [f"# SEM corner localization — summary",
             f"- model: `{model}` | temperature: {args.temperature} | repeats: {args.repeats} "
             f"| guided mechanism: {args.mechanism}",
             f"- eval images: see `{args.data_dir}/eval_manifest.json` | "
             f"exemplars: `{args.data_dir}/exemplar_manifest.json`",
             f"- tolerances are fractions of min(W, H); hit@τ counts items with L2/min(W,H) ≤ τ.",
             "",
             "| condition | fmt% | " + " | ".join(hit_headers)
             + " | hit5±sd | 95%CI(hit5) | bbox% | meanL2% | medL2% | wrongQ% | lat_ms |",
             "|---|---|" + "---|" * len(tolerances) + "---|---|---|---|---|---|---|"]
    for s in summary:
        hits = []
        for tau in tolerances:
            col = f"hit@{int(tau * 100)}pct"
            k, n, _ = s["hit_rates"][col]
            hits.append(f"{100 * k / n:.0f}" if n else "n/a")
        bbox = f"{s['hit_bbox_pct']:.0f}" if s["hit_bbox_pct"] == s["hit_bbox_pct"] else "n/a"
        l2u = f"{s['mean_L2_pct']:.1f}" if s["mean_L2_pct"] == s["mean_L2_pct"] else "n/a"
        l2m = f"{s['median_L2_pct']:.1f}" if s["median_L2_pct"] == s["median_L2_pct"] else "n/a"
        h5_col = "hit@5pct"
        k5, n5, (lo5, hi5) = s["hit_rates"].get(h5_col, (0, 0, (0.0, 0.0)))
        h5 = f"{100 * k5 / n5:.0f}±{s['hit5_std']:.1f}" if n5 else "n/a"
        ci5 = f"[{lo5:.0f},{hi5:.0f}]" if n5 else "n/a"
        lines.append(f"| {s['condition']} | {s['format_pct']:.0f} | "
                     + " | ".join(hits)
                     + f" | {h5} | {ci5} | {bbox} | {l2u} | {l2m} | "
                     f"{s['wrong_quadrant_pct']:.1f} | {s['latency_ms']:.0f} |")
    lines += ["",
              "**判讀**:`wrongQ%`(對角象限預測→閉環發散)與 `hit@τ%`(實際命中率)為安全/效能關鍵;"
              "`medL2%` 看典型誤差,`meanL2%` 易被離群值拉高。小樣本看 95%CI 寬度。"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Coordinate-regression validation for SEM corner localization (direct vLLM)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="./experiment_out")
    ap.add_argument("--conditions", nargs="*", help="subset of condition names; default all")
    ap.add_argument("--repeats", type=int, default=1, help="repeats per item (>=3 to characterize variance)")
    ap.add_argument("--limit", type=int, default=0, help="cap eval items (0=all)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--mechanism", choices=["guided_json", "response_format"], default="guided_json")
    ap.add_argument("--tolerances", type=float, nargs="*", default=None,
                    help="hit@τ thresholds as fractions of min(W,H). Default: 0.01 0.02 0.05 0.10")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
