#!/usr/bin/env python3
"""
run_fewshot_experiment.py — Few-shot validation for def/ref patch single-vs-cluster judgment.

Sister experiment to ../corner_judge_categorical/. Two things make this one different and
drive every design choice:

1. **Each example is a PAIR of small patches** — `def_patch` (under test) and `ref_patch`
   (reference). The patches are co-registered (same location/size) and mostly overlap in gray
   level (GLV); only a partial diff matters. The task is a **binary classification**:

       single  — def_patch has ONE relatively independent small bright round dot, brighter
                 than the same spot in ref_patch.
       cluster — def_patch's extra-bright region is multiple nearby dots flickering together,
                 OR one large blob (一大坨), OR a broad area lighting up (整片). Anything that
                 is NOT a single isolated small dot is `cluster`.

2. **The patches are tiny (e.g. 64x64) and the target can be ~5px.** Sent raw, the VLM's
   internal resizing can wipe out a 5px dot, so **client-side upscaling is a first-class
   experiment axis** (see PREP_CONDITIONS / --ladder preprocess and the --upscale/--interp/
   --contrast overrides). Interpolation matters for THIS task specifically: smoothing
   (bilinear/lanczos) can merge nearby dots into a blob -> false `cluster`; nearest preserves
   discreteness. That hypothesis is exactly what the preprocess ladder measures.

Hits vLLM **directly** via the `openai` package (no yJarvis facade) so the experiment has full
control over messages / guided_json / generation params.

Environment
-----------
    VLLM_BASE_URL   e.g. http://your-vllm-host:8000/v1
    VLLM_API_KEY    vLLM usually accepts any value (default "EMPTY")
    VLM_MODEL       e.g. Qwen3.6-27B   (or pass --model)

Data layout (--data-dir)
------------------------
    <data-dir>/
        eval_manifest.json       # [{ "def_patch": "...", "ref_patch": "...", "answer": {...} }, ...]  (HELD OUT)
        exemplar_manifest.json   # same shape, DISJOINT from eval (the few-shot examples)
        <patch images...>

The `answer` schema:
    { reasoning, brighter_region_found(bool),
      morphology(single_dot|multi_dots|large_blob|broad_area|unknown),
      label(single|cluster) }

Usage
-----
    # Stage 1 — find the best preprocessing (fixed k=6 prompt, vary upscale/interp/contrast):
    python run_fewshot_experiment.py --data-dir ./patch_eval --ladder preprocess --repeats 3

    # Stage 2 — run the prompt ladder at the chosen preprocessing:
    python run_fewshot_experiment.py --data-dir ./patch_eval --ladder prompt \
        --upscale 8 --interp nearest --repeats 3
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
# Prompts & schemas
# ---------------------------------------------------------------------------

_LABEL_RULES = """類別語意(def_patch 相對 ref_patch 新增的亮區形態):
- single: 新增亮區是「一個相對獨立、邊界清楚的小圓點」—— 面積小、孤立、周圍沒有其他同時變亮的點。
- cluster: 新增亮區是「多個鄰近同時變亮的點」、或「一大坨(blob)」、或「整片區域一起變亮」。
  判斷準則:只要「不是單一孤立小圓點」,就歸 cluster。
- morphology 細分(供推理用):
    single_dot  -> 單一孤立小圓點          (=> label single)
    multi_dots  -> 多個鄰近同時變亮的點      (=> label cluster)
    large_blob  -> 一大坨、邊界糊成一團      (=> label cluster)
    broad_area  -> 整片區域一起變亮          (=> label cluster)
    unknown     -> 看不出來/沒有明顯新增亮區 (=> 仍須在 single/cluster 二擇一給最接近者)
- brighter_region_found: def_patch 是否存在「明顯比 ref_patch 亮」的局部區域。若兩張幾乎一致則為 false。"""

# Domain knowledge: tells the model HOW to compare the two patches, not just what the labels
# mean. Without it the model tends to grade absolute brightness of def_patch alone; with it,
# the judgment is pinned to the *difference* def-minus-ref and the morphology of that diff.
_DOMAIN_KNOWLEDGE = """領域知識(def/ref patch 比對方法):
- 輸入是「兩張同位置、同尺寸、已對齊的小圖」:第一張 def_patch(待測)、第二張 ref_patch(參考)。
  兩張的灰階(GLV)大部分重疊或近似,通常只有局部差異。
- 影像可能已被放大顯示(原始 patch 很小、目標可能只有數個像素);請以「形態」而非絕對像素數判斷。
- 核心比較:想像把 def_patch 與 ref_patch 對齊後「逐像素相減」,只看「def 明顯比 ref 亮」的區域。
  忽略 (1) 兩張都亮或都暗的共同區域、(2) 整體一致的亮度位移(可能是曝光/背景漂移,不算缺陷)。
- 不要被「兩張都亮」的共同亮區誤導 —— 那不是 def 相對 ref 的新增亮點。
  關鍵永遠是「def 相對 ref 多出來的亮」這塊的形態。

判斷流程建議:
(a) 對齊兩張 patch,定位 def 明顯比 ref 亮的區域;若沒有 → brighter_region_found=false。
(b) 描述該新增亮區的形態:單一小圓點?多個鄰近點?一大坨?整片?
(c) 單一孤立小圓點 -> morphology=single_dot -> label=single;其餘形態 -> label=cluster。"""

SYSTEM_REASONING = (
    "你是半導體缺陷檢測的視覺判斷器,比對同位置的 def_patch(待測)與 ref_patch(參考)兩張小圖,"
    "判斷 def_patch 相對 ref_patch 新增的亮區是「單一孤立小圓點(single)」還是"
    "「多點/成團/整片(cluster)」。\n"
    + _DOMAIN_KNOWLEDGE + "\n" + _LABEL_RULES +
    "\n輸出規則:只輸出一個 JSON 物件,reasoning 欄位放最前面"
    "(先依領域知識描述 def 相對 ref 多出來的亮區位置與形態,再判斷 morphology,最後給 label 結論),"
    "不得有任何額外文字或 markdown。"
)
SYSTEM_NO_REASONING = (
    "你是半導體缺陷檢測的視覺判斷器,比對同位置的 def_patch(待測)與 ref_patch(參考)兩張小圖,"
    "判斷 def_patch 相對 ref_patch 新增的亮區是「單一孤立小圓點(single)」還是"
    "「多點/成團/整片(cluster)」。\n"
    + _DOMAIN_KNOWLEDGE + "\n" + _LABEL_RULES +
    "\n輸出規則:只輸出一個 JSON 物件,不得有任何額外文字或 markdown。"
)

# Each turn carries TWO images. The text pins their order so the model never confuses which is
# def and which is ref (image order in a multimodal turn is the only signal it has).
USER_TURN_TEXT = (
    "以下是同一位置、已對齊的兩張 patch(可能已被放大顯示)。第一張影像 = def_patch(待測),"
    "第二張影像 = ref_patch(參考)。請判斷 def_patch 相對 ref_patch 新增的亮區是 single"
    "(單一孤立小圓點)還是 cluster(多點/成團/整片),依 schema 輸出。"
)

_MORPHOLOGY = ["single_dot", "multi_dots", "large_blob", "broad_area", "unknown"]
_PROPS_CORE = {
    "brighter_region_found": {"type": "boolean"},
    "morphology": {"type": "string", "enum": list(_MORPHOLOGY)},
    "label": {"type": "string", "enum": ["single", "cluster"]},
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

# morphology -> label, so we can recompute the boolean answer from the descriptive field and
# flag self-contradictory predictions (mirrors the `aligned` recompute in the sibling experiment).
_MORPH_TO_LABEL = {"single_dot": "single", "multi_dots": "cluster",
                   "large_blob": "cluster", "broad_area": "cluster"}

# ---------------------------------------------------------------------------
# Image preprocessing — the lever the preprocess ladder ablates.
# A `prep` dict is {scale:int, interp:str, contrast:bool}. RAW_PREP = send patches as-is.
# ---------------------------------------------------------------------------

_INTERP = {
    "nearest": Image.NEAREST,    # preserves hard edges / discreteness of nearby dots
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,    # smooth; may merge nearby dots into a blob -> false cluster
}
RAW_PREP: Dict[str, Any] = {"scale": 1, "interp": "nearest", "contrast": False}

def _prep_key(prep: Dict[str, Any]) -> str:
    return f"s{prep['scale']}-{prep['interp']}-c{int(bool(prep['contrast']))}"

# ---------------------------------------------------------------------------
# Condition ladders. Each rung adds ONE thing, so an adjacent-rung difference isolates that
# decision. The user expects ~6 exemplars, so the few-shot rungs are k=3 and k=6.
#
#   prompt ladder      — vary k / guided / reasoning at a fixed preprocessing.
#   preprocess ladder  — vary upscale / interpolation / contrast at the fixed best prompt
#                        (k=6, guided, reasoning). This is the lever for tiny ~5px targets.
# ---------------------------------------------------------------------------

CONDITIONS: List[Dict[str, Any]] = [
    {"name": "zeroshot_freetext",           "k": 0, "guided": False, "reasoning": True},
    {"name": "zeroshot_guided",             "k": 0, "guided": True,  "reasoning": False},
    {"name": "zeroshot_guided_reasoning",   "k": 0, "guided": True,  "reasoning": True},
    {"name": "fewshot3_guided_reasoning",   "k": 3, "guided": True,  "reasoning": True},
    {"name": "fewshot6_guided_reasoning",   "k": 6, "guided": True,  "reasoning": True},
    {"name": "fewshot6_guided_noreasoning", "k": 6, "guided": True,  "reasoning": False},
]

def _prep_cond(name: str, scale: int, interp: str, contrast: bool) -> Dict[str, Any]:
    return {"name": name, "k": 6, "guided": True, "reasoning": True,
            "prep": {"scale": scale, "interp": interp, "contrast": contrast}}

PREP_CONDITIONS: List[Dict[str, Any]] = [
    _prep_cond("prep_raw_x1",             1,  "nearest", False),  # baseline: tiny patch as-is
    _prep_cond("prep_x4_nearest",         4,  "nearest", False),  # 64 -> 256
    _prep_cond("prep_x8_nearest",         8,  "nearest", False),  # 64 -> 512
    _prep_cond("prep_x8_lanczos",         8,  "lanczos", False),  # smoothing: does it blur dots together?
    _prep_cond("prep_x8_nearest_contrast",8,  "nearest", True),   # + joint contrast stretch for subtle GLV diff
    _prep_cond("prep_x12_nearest",        12, "nearest", False),  # 64 -> 768 (token-cost upper end)
]

LADDERS = {"prompt": CONDITIONS, "preprocess": PREP_CONDITIONS,
           "all": CONDITIONS + PREP_CONDITIONS}
_CONDITION_NAMES = {c["name"] for c in (CONDITIONS + PREP_CONDITIONS)}

ALLOWED_LABELS = {"single", "cluster"}

# Transient transport errors worth retrying. Anything else (auth, bad request, schema
# rejection) is signal and must surface immediately.
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
# Data / message building (preprocessing applied uniformly to exemplars AND eval)
# ---------------------------------------------------------------------------

def _joint_contrast(def_img: Image.Image, ref_img: Image.Image,
                    lo_pct: float = 2.0, hi_pct: float = 98.0) -> Tuple[Image.Image, Image.Image]:
    """Percentile stretch using a SINGLE LUT computed over BOTH patches, so the def-vs-ref
    brightness relationship is preserved. An independent per-image stretch would destroy the
    very diff we are trying to read (def's extra-bright dot could be normalized away)."""
    combined = sorted(list(def_img.getdata()) + list(ref_img.getdata()))
    n = len(combined)
    lo = combined[max(0, int(n * lo_pct / 100))]
    hi = combined[min(n - 1, int(n * hi_pct / 100))]
    if hi <= lo:
        return def_img, ref_img
    sc = 255.0 / (hi - lo)
    lut = [min(255, max(0, int((v - lo) * sc))) for v in range(256)]
    return def_img.point(lut), ref_img.point(lut)

def _resize(img: Image.Image, prep: Dict[str, Any]) -> Image.Image:
    if prep["scale"] == 1:
        return img
    w, h = img.size
    return img.resize((w * prep["scale"], h * prep["scale"]), _INTERP[prep["interp"]])

def _to_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")  # PNG lossless; JPEG artifacts corrupt the diff
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# Cache: exemplar pair URIs are constant across all eval items / repeats given a fixed prep, so
# encode each (def, ref, prep) at most once. Keyed by prep because contrast couples the pair.
_PAIR_URI_CACHE: Dict[Tuple[str, str, str], Tuple[str, str]] = {}

def _pair_uris(def_path: str, ref_path: str, prep: Dict[str, Any],
               cache: bool = False) -> Tuple[str, str]:
    key = (def_path, ref_path, _prep_key(prep))
    if cache and key in _PAIR_URI_CACHE:
        return _PAIR_URI_CACHE[key]
    with Image.open(def_path) as d0, Image.open(ref_path) as r0:
        d, r = d0.convert("L"), r0.convert("L")
    if prep["contrast"]:
        d, r = _joint_contrast(d, r)
    d, r = _resize(d, prep), _resize(r, prep)
    uris = (_to_uri(d), _to_uri(r))
    if cache:
        _PAIR_URI_CACHE[key] = uris
    return uris

def _pair_content(def_path: str, ref_path: str, prep: Dict[str, Any],
                  cache: bool = False) -> List[Dict[str, Any]]:
    """One user turn = prompt text + def_patch image + ref_patch image (in that fixed order)."""
    du, ru = _pair_uris(def_path, ref_path, prep, cache=cache)
    return [
        {"type": "text", "text": USER_TURN_TEXT},
        {"type": "image_url", "image_url": {"url": du}},
        {"type": "image_url", "image_url": {"url": ru}},
    ]

def load_manifest(data_dir: str, name: str) -> List[Dict[str, Any]]:
    path = os.path.join(data_dir, name)
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    for e in entries:
        e["_def_path"] = os.path.join(data_dir, e["def_patch"])
        e["_ref_path"] = os.path.join(data_dir, e["ref_patch"])
        for p in (e["_def_path"], e["_ref_path"]):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"image not found: {p}")
    return entries

def build_messages(cond: Dict[str, Any], item: Dict[str, Any],
                   exemplars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prep = cond.get("prep", RAW_PREP)
    system = SYSTEM_REASONING if cond["reasoning"] else SYSTEM_NO_REASONING
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    for ex in exemplars[: cond["k"]]:  # fixed order; see order-sensitivity note in .md
        ans = dict(ex["answer"])
        if not cond["reasoning"]:
            ans.pop("reasoning", None)
        msgs.append({"role": "user",
                     "content": _pair_content(ex["_def_path"], ex["_ref_path"], prep, cache=True)})
        msgs.append({"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)})
    msgs.append({"role": "user", "content": _pair_content(item["_def_path"], item["_ref_path"], prep)})
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
        brf, mor, lab = obj["brighter_region_found"], obj["morphology"], obj["label"]
    except KeyError:
        return None
    if not isinstance(brf, bool):
        return None
    if mor not in _MORPHOLOGY or lab not in ALLOWED_LABELS:
        return None
    return {"brighter_region_found": brf, "morphology": mor, "label": lab}

def score_one(gt: Dict[str, Any], pred: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-item flags for a binary single/cluster judgment.

    Malformed prediction (pred is None): label_ok=0 and pred_label="malformed" so it lands in
    the confusion matrix as "not correctly classified" for whatever the GT class was — a
    missing reply is unusable downstream, identical to a wrong class for recall purposes.
    """
    gt_label = gt["label"]
    if pred is None:
        return {"format_ok": 0, "label_ok": 0, "consistency_ok": None,
                "gt_label": gt_label, "pred_label": "malformed"}
    pred_label = pred["label"]
    expected = _MORPH_TO_LABEL.get(pred["morphology"])  # None for "unknown"
    return {
        "format_ok": 1,
        "label_ok": int(pred_label == gt_label),
        "consistency_ok": (int(pred_label == expected) if expected is not None else None),
        "gt_label": gt_label,
        "pred_label": pred_label,
    }

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

def _resolve_conditions(args: argparse.Namespace) -> List[Dict[str, Any]]:
    base = LADDERS[args.ladder]
    if args.conditions:
        unknown = [n for n in args.conditions if n not in _CONDITION_NAMES]
        if unknown:
            print(f"[warn] unknown --conditions ignored: {unknown}. "
                  f"Known: {sorted(_CONDITION_NAMES)}", file=sys.stderr)
        base = [c for c in base if c["name"] in args.conditions]
    # Global preprocessing override: applies to ALL selected conditions. Lets you run the
    # prompt ladder at one chosen preprocessing (Stage 2 above).
    if args.upscale is not None or args.interp is not None or args.contrast:
        ov = {"scale": args.upscale if args.upscale is not None else 1,
              "interp": args.interp or "nearest", "contrast": bool(args.contrast)}
        base = [{**c, "prep": ov} for c in base]
        print(f"[prep override] all conditions -> {_prep_key(ov)}")
    return base

def run(args: argparse.Namespace) -> None:
    eval_set = load_manifest(args.data_dir, "eval_manifest.json")
    exemplars = load_manifest(args.data_dir, "exemplar_manifest.json")
    if args.limit:
        eval_set = eval_set[: args.limit]
    conds = _resolve_conditions(args)
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
        prep = cond.get("prep", RAW_PREP)
        print(f"\n=== condition: {cond['name']} (k={cond['k']}, guided={cond['guided']}, "
              f"reasoning={cond['reasoning']}, prep={_prep_key(prep)}) ===")
        for rep in range(args.repeats):
            for item in eval_set:
                schema = (SCHEMA_REASONING if cond["reasoning"] else SCHEMA_NO_REASONING) if cond["guided"] else None
                msgs = build_messages(cond, item, exemplars)
                content, latency, err = call_vllm(
                    msgs, schema, model, args.temperature, args.max_tokens, args.timeout, args.mechanism)
                pred = None if err else parse_prediction(content)
                sc = score_one(item["answer"], pred)
                raw_rows.append({"condition": cond["name"], "repeat": rep, "prep": _prep_key(prep),
                                 "def_patch": item["def_patch"], "ref_patch": item["ref_patch"],
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

def _recall(rows: List[Dict[str, Any]], cls: str) -> Tuple[int, int]:
    """Among items whose GT is `cls`, how many were predicted `cls`."""
    sub = [r for r in rows if r["gt_label"] == cls]
    return sum(1 for r in sub if r["pred_label"] == cls), len(sub)

def _precision(rows: List[Dict[str, Any]], cls: str) -> Tuple[int, int]:
    """Among items PREDICTED `cls`, how many were actually `cls`."""
    sub = [r for r in rows if r["pred_label"] == cls]
    return sum(1 for r in sub if r["gt_label"] == cls), len(sub)

def _aggregate(rows: List[Dict[str, Any]], repeats: int) -> List[Dict[str, Any]]:
    out = []
    conds: List[str] = []
    for r in rows:
        if r["condition"] not in conds:
            conds.append(r["condition"])
    for cname in conds:
        cr = [r for r in rows if r["condition"] == cname]
        fk, fn = _rate(cr, "format_ok")
        ak, an = _rate(cr, "label_ok")
        sr_k, sr_n = _recall(cr, "single")
        cl_k, cl_n = _recall(cr, "cluster")
        sp_k, sp_n = _precision(cr, "single")
        cp_k, cp_n = _precision(cr, "cluster")
        single_rec = sr_k / sr_n if sr_n else 0.0
        cluster_rec = cl_k / cl_n if cl_n else 0.0
        bal_acc = (single_rec + cluster_rec) / 2
        # per-repeat balanced acc -> variance across repeats
        per_rep = []
        for rep in range(repeats):
            rr = [r for r in cr if r["repeat"] == rep]
            srk, srn = _recall(rr, "single")
            clk, cln = _recall(rr, "cluster")
            if srn and cln:
                per_rep.append(((srk / srn) + (clk / cln)) / 2)
        mean_rep = sum(per_rep) / len(per_rep) if per_rep else 0.0
        std_rep = (sum((x - mean_rep) ** 2 for x in per_rep) / len(per_rep)) ** 0.5 if len(per_rep) > 1 else 0.0
        ck, cn = _rate(cr, "consistency_ok")
        lat = [r["latency_ms"] for r in cr if r.get("latency_ms") is not None]
        lo, hi = wilson(ak, an)
        out.append({
            "condition": cname, "prep": cr[0].get("prep", ""), "n_calls": len(cr),
            "format_pct": 100 * fk / fn if fn else 0.0,
            "acc": 100 * ak / an if an else 0.0,
            "acc_ci": (100 * lo, 100 * hi),
            "single_recall": 100 * single_rec,
            "cluster_recall": 100 * cluster_rec,
            "single_prec": 100 * sp_k / sp_n if sp_n else 0.0,
            "cluster_prec": 100 * cp_k / cp_n if cp_n else 0.0,
            "bal_acc": 100 * bal_acc,
            "bal_std": 100 * std_rep,
            # the two directional errors: miss_single = GT single called cluster (PRIORITY:
            # single-miss is the worse error here); miss_cluster = GT cluster called single.
            "miss_single_pct": 100 * (1 - single_rec),
            "miss_cluster_pct": 100 * (1 - cluster_rec),
            "consistency_pct": 100 * ck / cn if cn else float("nan"),
            "latency_ms": sum(lat) / len(lat) if lat else 0.0,
        })
    return out

def _print_summary(summary: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 128)
    hdr = ("condition", "prep", "fmt%", "acc%", "balAcc%(±sd)", "95%CI", "sRec%", "cRec%",
           "missSgl%", "missClu%", "consist%", "lat_ms")
    print("{:<26} {:>12} {:>5} {:>5} {:>13} {:>11} {:>6} {:>6} {:>8} {:>8} {:>9} {:>8}".format(*hdr))
    print("-" * 128)
    for s in summary:
        ci = f"[{s['acc_ci'][0]:.0f},{s['acc_ci'][1]:.0f}]"
        cons = f"{s['consistency_pct']:.0f}" if s["consistency_pct"] == s["consistency_pct"] else "n/a"
        print("{:<26} {:>12} {:>5.1f} {:>5.1f} {:>13} {:>11} {:>6.0f} {:>6.0f} {:>8.1f} {:>8.1f} {:>9} {:>8.0f}".format(
            s["condition"], s["prep"], s["format_pct"], s["acc"],
            f"{s['bal_acc']:.1f}±{s['bal_std']:.1f}", ci,
            s["single_recall"], s["cluster_recall"],
            s["miss_single_pct"], s["miss_cluster_pct"], cons, s["latency_ms"]))
    print("=" * 128)
    print("優先指標(此用途 single 漏判較嚴重):missSgl% = 真 single 被判成 cluster,先壓這個。")
    print("  次要:missClu% = 真 cluster 被判成 single;balAcc% = 兩類 recall 平均(類別不平衡時比 acc% 可靠)。")
    print("  consist% = label 與 morphology 是否自洽;preprocess ladder 另看 lat_ms(放大=更多 token=更慢)。")

def _write_raw_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    cols = ["condition", "prep", "repeat", "def_patch", "ref_patch", "format_ok", "label_ok",
            "gt_label", "pred_label", "consistency_ok", "latency_ms", "error", "raw"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def _write_summary_md(summary: List[Dict[str, Any]], model: str, args: argparse.Namespace, path: str) -> None:
    lines = [f"# def/ref patch single-vs-cluster few-shot experiment — summary",
             f"- model: `{model}` | temperature: {args.temperature} | repeats: {args.repeats} "
             f"| ladder: {args.ladder} | guided mechanism: {args.mechanism}",
             f"- eval pairs: see `{args.data_dir}/eval_manifest.json` | "
             f"exemplars: `{args.data_dir}/exemplar_manifest.json`", "",
             "| condition | prep | fmt% | acc% | balAcc%±sd | 95%CI | sRec% | cRec% | missSgl% | missClu% | consist% | lat_ms |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        ci = f"[{s['acc_ci'][0]:.0f},{s['acc_ci'][1]:.0f}]"
        cons = f"{s['consistency_pct']:.0f}" if s["consistency_pct"] == s["consistency_pct"] else "n/a"
        lines.append(f"| {s['condition']} | {s['prep']} | {s['format_pct']:.1f} | {s['acc']:.1f} | "
                     f"{s['bal_acc']:.1f}±{s['bal_std']:.1f} | {ci} | {s['single_recall']:.0f} | "
                     f"{s['cluster_recall']:.0f} | {s['miss_single_pct']:.1f} | "
                     f"{s['miss_cluster_pct']:.1f} | {cons} | {s['latency_ms']:.0f} |")
    lines += ["", "**判讀**:此用途 **single 漏判較嚴重**,優先壓 `missSgl%`(真 single 被判成 cluster);"
              "`missClu%`(真 cluster 被判成 single)次要。類別不平衡時 `balAcc%` 比 `acc%` 可靠。"
              "preprocess ladder 額外比較放大倍率/內插:smoothing(lanczos)可能把鄰近點糊成一坨而誤判,"
              "nearest 較能保留離散性;`lat_ms` 反映放大帶來的 token 成本。小樣本看 95%CI 寬度,別過度解讀點估計。"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Few-shot validation for def/ref patch single-vs-cluster judgment (direct vLLM)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="./experiment_out")
    ap.add_argument("--ladder", choices=list(LADDERS), default="prompt",
                    help="which condition ladder: prompt (vary k/guided/reasoning), "
                         "preprocess (vary upscale/interp/contrast at fixed k=6), or all")
    ap.add_argument("--conditions", nargs="*", help="subset of condition names; default = whole ladder")
    ap.add_argument("--upscale", type=int, default=None,
                    help="override: integer client-side upscale factor applied to ALL conditions")
    ap.add_argument("--interp", choices=list(_INTERP), default=None,
                    help="override: interpolation for --upscale (default nearest)")
    ap.add_argument("--contrast", action="store_true",
                    help="override: joint percentile contrast stretch across the def/ref pair")
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
