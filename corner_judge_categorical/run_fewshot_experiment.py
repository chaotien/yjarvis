#!/usr/bin/env python3
"""
run_fewshot_experiment.py — Few-shot validation experiment for SEM corner judgment.

Hits vLLM **directly** via the `openai` package (no yJarvis facade) so the experiment
has full control over messages / guided_json / generation params. Runs a ladder of
ablation conditions over a labeled eval set and reports metrics tuned for closed-loop
alignment (where wrong-direction and false-aligned are the dangerous errors).

Environment
-----------
    VLLM_BASE_URL   e.g. http://your-vllm-host:8000/v1
    VLLM_API_KEY    vLLM usually accepts any value (default "EMPTY")
    VLM_MODEL       e.g. Qwen3.6-27B   (or pass --model)

Data layout (--data-dir)
------------------------
    <data-dir>/
        eval_manifest.json       # [{ "image": "f.png", "answer": {...} }, ...]  (HELD OUT)
        exemplar_manifest.json   # [{ "image": "g.png", "answer": {...} }, ...]  (DISJOINT from eval)
        <images...>

The `answer` schema matches sem_corner_judge:
    { reasoning, corner_found(bool), offset_x(left|center|right|unknown),
      offset_y(above|center|below|unknown), magnitude(none|slight|moderate|large|unknown),
      aligned(bool) }

Usage
-----
    python run_fewshot_experiment.py --data-dir ./sem_eval --repeats 3
    python run_fewshot_experiment.py --data-dir ./sem_eval --conditions zeroshot_guided_reasoning fewshot5_guided_reasoning
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
except ImportError:  # allow import without openai (e.g. offline scoring tests)
    OpenAI = None  # type: ignore

# ---------------------------------------------------------------------------
# Prompts & schemas (kept consistent with sem_corner_judge so we test the real thing)
# ---------------------------------------------------------------------------

_OFFSET_RULES = """偏移語意(角的頂點相對十字線中心):
- offset_x: left(在左)/ right(在右)/ center(水平已對齊)
- offset_y: above(在上)/ below(在下)/ center(垂直已對齊)
- magnitude: none(約1/20以內,視為對齊)/ slight(約1/10內)/ moderate(1/10~1/4)/ large(超過1/4)
- aligned: 當且僅當 offset_x 與 offset_y 皆為 center 時為 true
- 找不到角(移出視野/失焦/雜訊蓋過): corner_found=false,offset_x/offset_y/magnitude 填 "unknown",aligned=false"""

# Domain knowledge: tells the model HOW to identify the corner, not just what it looks like.
# Without this the model has to guess "which L-vertex"; with it, the corner is pinned to a
# single well-defined point (bottom-left of the patterned area).
_DOMAIN_KNOWLEDGE = """領域知識(area 與角的辨識特徵):
- Area 形狀與位置:area 是一個長方形區域,從 cropped ROI 的「右上角」開始,連續向「左下方」延伸。
  ROI 右上角必定在 area 內,左下角必定在 area 外;area 的上邊界、右邊界通常與 ROI 上、右邊重合。
- Area 內部特徵:area 內可觀察到清楚的「重複出現的 pattern」(例如週期性的亮暗紋理 / 陣列結構);
  area 外則沒有此 pattern,或重複 pattern 中斷消失。
- Area 邊界判斷:由右上向左下追蹤,重複 pattern「不再繼續」的位置即為 area 邊界。
    左邊界:一條垂直線(pattern 在水平方向最左端,X_left)。
    下邊界:一條水平線(pattern 在垂直方向最下端,Y_bottom)。
- 角(corner)的精確定義:area 左邊界與下邊界的「交點」= (X_left, Y_bottom);
  也就是 area 形狀上「最左下方」的 L 形頂點。視覺上呈「└」形 —— 從交點往「上」是 area 的左邊界、
  往「右」是 area 的下邊界,L 形開口朝右上(area 本體所在方向)。

判斷流程建議:
(a) 在 ROI 右上半部確認 area 存在(可見重複 pattern)。
(b) 由右上往左下追蹤,找出 pattern 中斷的最左欄 → X_left。
(c) 同樣追蹤,找出 pattern 中斷的最下列 → Y_bottom。
(d) (X_left, Y_bottom) 即為 area corner。"""

SYSTEM_REASONING = (
    "你是半導體 SEM 對齊的視覺判斷器,判斷影像中目標結構的「角」相對於畫面中央十字線中心的對位狀態。\n"
    + _DOMAIN_KNOWLEDGE + "\n" + _OFFSET_RULES +
    "\n輸出規則:只輸出一個 JSON 物件,reasoning 欄位放最前面"
    "(先依領域知識描述看到的 area 範圍與重複 pattern 邊界,接著定位 corner (X_left, Y_bottom),"
    "再描述十字線中心位置與 corner 相對它的方位,最後給結論),不得有任何額外文字或 markdown。"
)
SYSTEM_NO_REASONING = (
    "你是半導體 SEM 對齊的視覺判斷器,判斷影像中目標結構的「角」相對於畫面中央十字線中心的對位狀態。\n"
    + _DOMAIN_KNOWLEDGE + "\n" + _OFFSET_RULES +
    "\n輸出規則:只輸出一個 JSON 物件,不得有任何額外文字或 markdown。"
)

USER_TURN_TEXT = "判斷這張 SEM 影像中,目標結構的角相對於 target marker(十字線中心)的對位狀態,依 schema 輸出。"

_PROPS_CORE = {
    "corner_found": {"type": "boolean"},
    "offset_x": {"type": "string", "enum": ["left", "center", "right", "unknown"]},
    "offset_y": {"type": "string", "enum": ["above", "center", "below", "unknown"]},
    "magnitude": {"type": "string", "enum": ["none", "slight", "moderate", "large", "unknown"]},
    "aligned": {"type": "boolean"},
}
SCHEMA_REASONING = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, **_PROPS_CORE},
    "required": ["reasoning", *_PROPS_CORE.keys()],
    "additionalProperties": False,
}
SCHEMA_NO_REASONING = {
    "type": "object",
    "properties": dict(_PROPS_CORE),
    "required": list(_PROPS_CORE.keys()),
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Condition ladder. Each rung isolates one decision (see the .md for the mapping).
# guided=False -> free text, parse JSON from prose (current yellow_score_box style).
# ---------------------------------------------------------------------------

CONDITIONS: List[Dict[str, Any]] = [
    {"name": "zeroshot_freetext",          "k": 0, "guided": False, "reasoning": True},
    {"name": "zeroshot_guided",            "k": 0, "guided": True,  "reasoning": False},
    {"name": "zeroshot_guided_reasoning",  "k": 0, "guided": True,  "reasoning": True},
    {"name": "fewshot3_guided_reasoning",  "k": 3, "guided": True,  "reasoning": True},
    {"name": "fewshot5_guided_reasoning",  "k": 5, "guided": True,  "reasoning": True},
    {"name": "fewshot5_guided_noreasoning","k": 5, "guided": True,  "reasoning": False},
]
_CONDITION_NAMES = {c["name"] for c in CONDITIONS}

ALLOWED = {
    "offset_x": {"left", "center", "right", "unknown"},
    "offset_y": {"above", "center", "below", "unknown"},
    "magnitude": {"none", "slight", "moderate", "large", "unknown"},
}
MAG_ORD = {"none": 0, "slight": 1, "moderate": 2, "large": 3}

# Transient transport errors worth retrying. Anything else (auth, bad request,
# schema rejection) is signal and must surface immediately.
_RETRYABLE_ERROR_NAMES = {
    "APITimeoutError", "APIConnectionError", "InternalServerError",
    "RateLimitError", "ConnectionError", "Timeout", "ReadTimeout",
}

# ---------------------------------------------------------------------------
# vLLM transport (direct openai)
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
    if mechanism == "response_format":  # vLLM mechanism B (newer)
        return {"response_format": {"type": "json_schema",
                                    "json_schema": {"name": "result", "schema": schema, "strict": True}}}
    return {"extra_body": {"guided_json": schema}}  # mechanism A (default, broadest)

def call_vllm(messages: List[Dict[str, Any]], schema: Optional[Dict[str, Any]],
              model: str, temperature: float, max_tokens: int, timeout: int,
              mechanism: str, retries: int = 2) -> Tuple[str, float, Optional[str]]:
    """Single call with bounded retry on transient transport errors.
    Returns (content, latency_ms, error). Total latency includes retry waits.
    """
    t0 = time.time()
    last_err: Optional[str] = None
    # Attempts = 1 initial + `retries` retries. Backoff 1s, 4s (matches doc tone).
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

def _data_uri(img_path: str) -> str:
    with Image.open(img_path) as raw:
        img = raw.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# Cache: data URIs for exemplar images are constant across all eval items / repeats /
# conditions, so encode each exemplar image at most once per run.
_URI_CACHE: Dict[str, str] = {}

def _cached_data_uri(img_path: str) -> str:
    uri = _URI_CACHE.get(img_path)
    if uri is None:
        uri = _data_uri(img_path)
        _URI_CACHE[img_path] = uri
    return uri

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
    return entries

def build_messages(cond: Dict[str, Any], item: Dict[str, Any],
                   exemplars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    system = SYSTEM_REASONING if cond["reasoning"] else SYSTEM_NO_REASONING
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    for ex in exemplars[: cond["k"]]:  # fixed order; see order-sensitivity note in .md
        ans = dict(ex["answer"])
        if not cond["reasoning"]:
            ans.pop("reasoning", None)
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

def parse_prediction(content: str) -> Optional[Dict[str, Any]]:
    obj = _extract_json(content)
    if not isinstance(obj, dict):
        return None
    try:
        cf, ox, oy = obj["corner_found"], obj["offset_x"], obj["offset_y"]
        mg, al = obj["magnitude"], obj["aligned"]
    except KeyError:
        return None
    if not isinstance(cf, bool) or not isinstance(al, bool):
        return None
    if ox not in ALLOWED["offset_x"] or oy not in ALLOWED["offset_y"] or mg not in ALLOWED["magnitude"]:
        return None
    return {"corner_found": cf, "offset_x": ox, "offset_y": oy, "magnitude": mg, "aligned": al}

def _opp(axis_vals: Tuple[str, str], gt: str, pred: str) -> bool:
    a, b = axis_vals
    return {gt, pred} == {a, b}

def score_one(gt: Dict[str, Any], pred: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-item flags. Directional metrics only count on the 'corner present' subset.

    Malformed prediction (pred is None): counts as wrong_dir=1 on the found subset
    (a missing reply IS a controller risk — the loop has to fall back / retry),
    but as false_aligned=0 on the not-aligned subset (a missing reply never makes
    the false claim `aligned=true`, so it can't trigger the premature-stop failure).
    Asymmetric, but each side matches what the closed-loop controller would do.
    """
    found_subset = bool(gt["corner_found"])
    not_aligned_subset = (gt["corner_found"] and not gt["aligned"])
    if pred is None:
        return {"format_ok": 0, "corner_found_ok": 0, "aligned_ok": 0,
                "ox_ok": None, "oy_ok": None, "mag_ae": None,
                "wrong_dir": (1 if found_subset else None),
                "false_aligned": (0 if not_aligned_subset else None),
                "found_subset": found_subset, "not_aligned_subset": not_aligned_subset}
    ox_ok = oy_ok = mag_ae = wrong_dir = None
    if found_subset:
        ox_ok = int(pred["offset_x"] == gt["offset_x"])
        oy_ok = int(pred["offset_y"] == gt["offset_y"])
        if gt["magnitude"] in MAG_ORD and pred["magnitude"] in MAG_ORD:
            mag_ae = abs(MAG_ORD[gt["magnitude"]] - MAG_ORD[pred["magnitude"]])
        wrong_dir = int(_opp(("left", "right"), gt["offset_x"], pred["offset_x"])
                        or _opp(("above", "below"), gt["offset_y"], pred["offset_y"]))
    false_aligned = int(pred["aligned"]) if not_aligned_subset else None
    return {"format_ok": 1,
            "corner_found_ok": int(pred["corner_found"] == gt["corner_found"]),
            "aligned_ok": int(pred["aligned"] == gt["aligned"]),
            "ox_ok": ox_ok, "oy_ok": oy_ok, "mag_ae": mag_ae,
            "wrong_dir": wrong_dir, "false_aligned": false_aligned,
            "found_subset": found_subset, "not_aligned_subset": not_aligned_subset}

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
    model = args.model or os.environ.get("VLM_MODEL", "Qwen3.6-27B")
    os.makedirs(args.out_dir, exist_ok=True)
    raw_rows: List[Dict[str, Any]] = []
    for cond in conds:
        if cond["k"] > len(exemplars):
            print(f"[skip] {cond['name']}: needs {cond['k']} exemplars, only {len(exemplars)} available")
            continue
        print(f"\n=== condition: {cond['name']} (k={cond['k']}, guided={cond['guided']}, "
              f"reasoning={cond['reasoning']}) ===")
        for rep in range(args.repeats):
            for item in eval_set:
                schema = (SCHEMA_REASONING if cond["reasoning"] else SCHEMA_NO_REASONING) if cond["guided"] else None
                msgs = build_messages(cond, item, exemplars)
                content, latency, err = call_vllm(
                    msgs, schema, model, args.temperature, args.max_tokens, args.timeout, args.mechanism)
                pred = None if err else parse_prediction(content)
                sc = score_one(item["answer"], pred)
                raw_rows.append({"condition": cond["name"], "repeat": rep, "image": item["image"],
                                 "error": err or "", "latency_ms": round(latency, 1),
                                 "raw": content[:500], **sc})
            print(f"  repeat {rep + 1}/{args.repeats} done")
    _write_raw_csv(raw_rows, os.path.join(args.out_dir, "raw_results.csv"))
    summary = _aggregate(raw_rows, args.repeats)
    _print_summary(summary)
    _write_summary_md(summary, model, args, os.path.join(args.out_dir, "summary.md"))
    print(f"\nWrote: {args.out_dir}/raw_results.csv  and  {args.out_dir}/summary.md")

def _rate(rows: List[Dict[str, Any]], key: str) -> Tuple[int, int]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals), len(vals)

def _aggregate(rows: List[Dict[str, Any]], repeats: int) -> List[Dict[str, Any]]:
    out = []
    conds = []
    for r in rows:
        if r["condition"] not in conds:
            conds.append(r["condition"])
    for cname in conds:
        cr = [r for r in rows if r["condition"] == cname]
        fk, fn = _rate(cr, "format_ok")
        ak, an = _rate(cr, "aligned_ok")
        # per-repeat aligned acc -> variance across repeats
        per_rep = []
        for rep in range(repeats):
            rr = [r for r in cr if r["repeat"] == rep]
            k, n = _rate(rr, "aligned_ok")
            if n:
                per_rep.append(k / n)
        mean_rep = sum(per_rep) / len(per_rep) if per_rep else 0.0
        std_rep = (sum((x - mean_rep) ** 2 for x in per_rep) / len(per_rep)) ** 0.5 if len(per_rep) > 1 else 0.0
        oxk, oxn = _rate(cr, "ox_ok")
        oyk, oyn = _rate(cr, "oy_ok")
        wdk, wdn = _rate(cr, "wrong_dir")
        fak, fan = _rate(cr, "false_aligned")
        maes = [r["mag_ae"] for r in cr if r.get("mag_ae") is not None]
        lat = [r["latency_ms"] for r in cr if r.get("latency_ms") is not None]
        lo, hi = wilson(ak, an)
        out.append({
            "condition": cname, "n_calls": len(cr),
            "format_pct": 100 * fk / fn if fn else 0.0,
            "aligned_acc": 100 * ak / an if an else 0.0,
            "aligned_std": 100 * std_rep,
            "aligned_ci": (100 * lo, 100 * hi),
            "ox_acc": 100 * oxk / oxn if oxn else 0.0,
            "oy_acc": 100 * oyk / oyn if oyn else 0.0,
            "mag_mae": sum(maes) / len(maes) if maes else float("nan"),
            "wrong_dir_pct": 100 * wdk / wdn if wdn else 0.0,
            "false_aligned_pct": 100 * fak / fan if fan else 0.0,
            "latency_ms": sum(lat) / len(lat) if lat else 0.0,
        })
    return out

def _print_summary(summary: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 110)
    hdr = ("condition", "fmt%", "align%(±sd)", "95%CI", "ox%", "oy%", "magMAE", "wrongDir%", "falseAln%", "lat_ms")
    print("{:<30} {:>6} {:>12} {:>13} {:>5} {:>5} {:>7} {:>10} {:>10} {:>8}".format(*hdr))
    print("-" * 110)
    for s in summary:
        ci = f"[{s['aligned_ci'][0]:.0f},{s['aligned_ci'][1]:.0f}]"
        print("{:<30} {:>6.1f} {:>12} {:>13} {:>5.0f} {:>5.0f} {:>7} {:>10.1f} {:>10.1f} {:>8.0f}".format(
            s["condition"], s["format_pct"], f"{s['aligned_acc']:.1f}±{s['aligned_std']:.1f}", ci,
            s["ox_acc"], s["oy_acc"],
            f"{s['mag_mae']:.2f}" if s["mag_mae"] == s["mag_mae"] else "n/a",
            s["wrong_dir_pct"], s["false_aligned_pct"], s["latency_ms"]))
    print("=" * 110)
    print("關鍵看 wrongDir%(方向相反,閉環會發散)與 falseAln%(誤判已對齊,過早停),overall align% 最不具參考性。")

def _write_raw_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    cols = ["condition", "repeat", "image", "format_ok", "corner_found_ok", "aligned_ok",
            "ox_ok", "oy_ok", "mag_ae", "wrong_dir", "false_aligned", "latency_ms", "error", "raw"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def _write_summary_md(summary: List[Dict[str, Any]], model: str, args: argparse.Namespace, path: str) -> None:
    lines = [f"# SEM corner few-shot experiment — summary",
             f"- model: `{model}` | temperature: {args.temperature} | repeats: {args.repeats} "
             f"| guided mechanism: {args.mechanism}",
             f"- eval images: see `{args.data_dir}/eval_manifest.json` | "
             f"exemplars: `{args.data_dir}/exemplar_manifest.json`", "",
             "| condition | fmt% | align%±sd | 95%CI | ox% | oy% | magMAE | wrongDir% | falseAln% | lat_ms |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        ci = f"[{s['aligned_ci'][0]:.0f},{s['aligned_ci'][1]:.0f}]"
        mae = f"{s['mag_mae']:.2f}" if s["mag_mae"] == s["mag_mae"] else "n/a"
        lines.append(f"| {s['condition']} | {s['format_pct']:.1f} | "
                     f"{s['aligned_acc']:.1f}±{s['aligned_std']:.1f} | {ci} | {s['ox_acc']:.0f} | "
                     f"{s['oy_acc']:.0f} | {mae} | {s['wrong_dir_pct']:.1f} | "
                     f"{s['false_aligned_pct']:.1f} | {s['latency_ms']:.0f} |")
    lines += ["", "**判讀**:wrongDir%(方向相反→閉環發散)與 falseAln%(誤判對齊→過早停)為安全關鍵指標;"
              "overall align% 在類別不平衡時最易誤導。小樣本請看 95%CI 寬度,不要過度解讀點估計。"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def main() -> None:
    ap = argparse.ArgumentParser(description="Few-shot validation for SEM corner judgment (direct vLLM)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="./experiment_out")
    ap.add_argument("--conditions", nargs="*", help="subset of condition names; default all")
    ap.add_argument("--repeats", type=int, default=1, help="repeats per item (>=3 to characterize variance)")
    ap.add_argument("--limit", type=int, default=0, help="cap eval items (0=all)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--mechanism", choices=["guided_json", "response_format"], default="guided_json",
                    help="vLLM structured-output mechanism (A=guided_json default, B=response_format)")
    run(ap.parse_args())

if __name__ == "__main__":
    main()
