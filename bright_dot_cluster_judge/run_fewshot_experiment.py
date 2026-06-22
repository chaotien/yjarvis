#!/usr/bin/env python3
"""
run_fewshot_experiment.py — Few-shot validation for def/ref patch single-vs-cluster judgment.

Sister experiment to ../corner_judge_categorical/. The difference that drives every design
choice here: **each example is a PAIR of small patches** — `def_patch` (the patch under test)
and `ref_patch` (the reference) — and the task is a **binary classification**:

    single  — def_patch has ONE relatively independent small bright round dot, brighter than
              the same spot in ref_patch.
    cluster — def_patch's extra-bright region is multiple nearby dots flickering together,
              OR one large blob (一大坨), OR a broad area lighting up (整片). Anything that is
              NOT a single isolated small dot is `cluster`.

The two patches mostly overlap in gray level (GLV); only a partial diff matters. The model is
asked to reason about "what's brighter in def vs ref" and classify the morphology of that diff.

Hits vLLM **directly** via the `openai` package (no yJarvis facade) so the experiment has full
control over messages / guided_json / generation params, exactly like the sibling experiments.

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
    python run_fewshot_experiment.py --data-dir ./patch_eval --repeats 3
    python run_fewshot_experiment.py --data-dir ./patch_eval \
        --conditions zeroshot_guided_reasoning fewshot6_guided_reasoning
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
- 輸入是「兩張同位置、同尺寸的小圖」:第一張 def_patch(待測)、第二張 ref_patch(參考)。
  兩張的灰階(GLV)大部分重疊或近似,通常只有局部差異。
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
    "以下是同一位置的兩張 patch。第一張影像 = def_patch(待測),第二張影像 = ref_patch(參考)。"
    "請判斷 def_patch 相對 ref_patch 新增的亮區是 single(單一孤立小圓點)還是 cluster"
    "(多點/成團/整片),依 schema 輸出。"
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
# Condition ladder. Each rung isolates one decision (see the .md for the mapping).
# The user expects ~6 exemplars, so the few-shot rungs are k=3 and k=6.
# ---------------------------------------------------------------------------

CONDITIONS: List[Dict[str, Any]] = [
    {"name": "zeroshot_freetext",           "k": 0, "guided": False, "reasoning": True},
    {"name": "zeroshot_guided",             "k": 0, "guided": True,  "reasoning": False},
    {"name": "zeroshot_guided_reasoning",   "k": 0, "guided": True,  "reasoning": True},
    {"name": "fewshot3_guided_reasoning",   "k": 3, "guided": True,  "reasoning": True},
    {"name": "fewshot6_guided_reasoning",   "k": 6, "guided": True,  "reasoning": True},
    {"name": "fewshot6_guided_noreasoning", "k": 6, "guided": True,  "reasoning": False},
]
_CONDITION_NAMES = {c["name"] for c in CONDITIONS}

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
# Data / message building
# ---------------------------------------------------------------------------

def _data_uri(img_path: str) -> str:
    with Image.open(img_path) as raw:
        img = raw.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")  # PNG is lossless; JPEG artifacts would corrupt the diff signal
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# Cache: exemplar patch URIs are constant across all eval items / repeats / conditions, so
# encode each exemplar image at most once per run.
_URI_CACHE: Dict[str, str] = {}

def _cached_data_uri(img_path: str) -> str:
    uri = _URI_CACHE.get(img_path)
    if uri is None:
        uri = _data_uri(img_path)
        _URI_CACHE[img_path] = uri
    return uri

def _pair_content(def_path: str, ref_path: str, cache: bool = False) -> List[Dict[str, Any]]:
    """One user turn = prompt text + def_patch image + ref_patch image (in that fixed order)."""
    uri = _cached_data_uri if cache else _data_uri
    return [
        {"type": "text", "text": USER_TURN_TEXT},
        {"type": "image_url", "image_url": {"url": uri(def_path)}},
        {"type": "image_url", "image_url": {"url": uri(ref_path)}},
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
    system = SYSTEM_REASONING if cond["reasoning"] else SYSTEM_NO_REASONING
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    for ex in exemplars[: cond["k"]]:  # fixed order; see order-sensitivity note in .md
        ans = dict(ex["answer"])
        if not cond["reasoning"]:
            ans.pop("reasoning", None)
        msgs.append({"role": "user",
                     "content": _pair_content(ex["_def_path"], ex["_ref_path"], cache=True)})
        msgs.append({"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)})
    msgs.append({"role": "user", "content": _pair_content(item["_def_path"], item["_ref_path"])})
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
                raw_rows.append({"condition": cond["name"], "repeat": rep,
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
            "condition": cname, "n_calls": len(cr),
            "format_pct": 100 * fk / fn if fn else 0.0,
            "acc": 100 * ak / an if an else 0.0,
            "acc_ci": (100 * lo, 100 * hi),
            "single_recall": 100 * single_rec,
            "cluster_recall": 100 * cluster_rec,
            "single_prec": 100 * sp_k / sp_n if sp_n else 0.0,
            "cluster_prec": 100 * cp_k / cp_n if cp_n else 0.0,
            "bal_acc": 100 * bal_acc,
            "bal_std": 100 * std_rep,
            # the two directional errors: miss_cluster = GT cluster called single (under-call);
            # miss_single = GT single called cluster (over-call).
            "miss_cluster_pct": 100 * (1 - cluster_rec),
            "miss_single_pct": 100 * (1 - single_rec),
            "consistency_pct": 100 * ck / cn if cn else float("nan"),
            "latency_ms": sum(lat) / len(lat) if lat else 0.0,
        })
    return out

def _print_summary(summary: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 120)
    hdr = ("condition", "fmt%", "acc%", "balAcc%(±sd)", "95%CI", "sRec%", "cRec%",
           "missClu%", "missSgl%", "consist%", "lat_ms")
    print("{:<30} {:>5} {:>5} {:>13} {:>11} {:>6} {:>6} {:>8} {:>8} {:>9} {:>8}".format(*hdr))
    print("-" * 120)
    for s in summary:
        ci = f"[{s['acc_ci'][0]:.0f},{s['acc_ci'][1]:.0f}]"
        cons = f"{s['consistency_pct']:.0f}" if s["consistency_pct"] == s["consistency_pct"] else "n/a"
        print("{:<30} {:>5.1f} {:>5.1f} {:>13} {:>11} {:>6.0f} {:>6.0f} {:>8.1f} {:>8.1f} {:>9} {:>8.0f}".format(
            s["condition"], s["format_pct"], s["acc"],
            f"{s['bal_acc']:.1f}±{s['bal_std']:.1f}", ci,
            s["single_recall"], s["cluster_recall"],
            s["miss_cluster_pct"], s["miss_single_pct"], cons, s["latency_ms"]))
    print("=" * 120)
    print("關鍵看 balAcc%(類別不平衡時比 acc% 可靠)與兩個方向的漏判:")
    print("  missClu% = 真 cluster 被判成 single(漏報成團/缺陷低估);missSgl% = 真 single 被判成 cluster(過度報警)。")
    print("  哪一個更危險取決於下游用途;先壓你最在意的那一個。consist% = label 與 morphology 是否自洽。")

def _write_raw_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    cols = ["condition", "repeat", "def_patch", "ref_patch", "format_ok", "label_ok",
            "gt_label", "pred_label", "consistency_ok", "latency_ms", "error", "raw"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def _write_summary_md(summary: List[Dict[str, Any]], model: str, args: argparse.Namespace, path: str) -> None:
    lines = [f"# def/ref patch single-vs-cluster few-shot experiment — summary",
             f"- model: `{model}` | temperature: {args.temperature} | repeats: {args.repeats} "
             f"| guided mechanism: {args.mechanism}",
             f"- eval pairs: see `{args.data_dir}/eval_manifest.json` | "
             f"exemplars: `{args.data_dir}/exemplar_manifest.json`", "",
             "| condition | fmt% | acc% | balAcc%±sd | 95%CI | sRec% | cRec% | missClu% | missSgl% | consist% | lat_ms |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        ci = f"[{s['acc_ci'][0]:.0f},{s['acc_ci'][1]:.0f}]"
        cons = f"{s['consistency_pct']:.0f}" if s["consistency_pct"] == s["consistency_pct"] else "n/a"
        lines.append(f"| {s['condition']} | {s['format_pct']:.1f} | {s['acc']:.1f} | "
                     f"{s['bal_acc']:.1f}±{s['bal_std']:.1f} | {ci} | {s['single_recall']:.0f} | "
                     f"{s['cluster_recall']:.0f} | {s['miss_cluster_pct']:.1f} | "
                     f"{s['miss_single_pct']:.1f} | {cons} | {s['latency_ms']:.0f} |")
    lines += ["", "**判讀**:類別不平衡時 `balAcc%`(single/cluster 兩類 recall 的平均)比 `acc%` 可靠;"
              "`missClu%`(真 cluster 漏判成 single)與 `missSgl%`(真 single 過判成 cluster)是兩個方向的錯誤,"
              "依下游用途決定先壓哪個。小樣本請看 95%CI 寬度,不要過度解讀點估計。"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Few-shot validation for def/ref patch single-vs-cluster judgment (direct vLLM)")
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
