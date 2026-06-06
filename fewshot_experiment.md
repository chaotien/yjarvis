# Few-shot Validation Experiment — SEM Corner Judgment

> Goal: **measure whether few-shot actually improves SEM corner judgment before paying to
> build the yJarvis `B` endpoint for it.** This experiment hits vLLM **directly** via the
> `openai` package so we control every variable; yJarvis is deliberately out of the loop.
> Companion runnable script: `run_fewshot_experiment.py` (embedded in full in §8).

---

## 1. Purpose & hypotheses

We previously reasoned that, for a *perception-bound* judgment like SEM corner alignment
(grayscale, noise, low contrast):

- A `messages` role split mostly helps **format compliance**, not perception.
- `guided_json` mainly buys **reliability** (eliminates parse failures), and *may* cost
  some reasoning quality if the schema is too rigid.
- A **reasoning-first** field can recover that reasoning under constraint.
- **Few-shot** is the real perception lever — *if* the model can resolve the corner at all.

These are hypotheses, not facts. The experiment turns each into a measured comparison.

**H1 (reliability):** `guided_json` drives format compliance to ~100% vs free text.
**H2 (reasoning tradeoff):** a reasoning-first field is neutral-to-positive on accuracy
under `guided_json` (i.e. constraint alone doesn't tank perception).
**H3 (few-shot value):** few-shot raises directional accuracy / lowers wrong-direction rate
beyond zero-shot, by a margin larger than the confidence intervals.
**H4 (saturation):** there is a small `k` past which more shots stop helping.

**Null we must be willing to accept:** few-shot adds nothing beyond `guided_json +
reasoning`, *or* the model is perception-bound and no prompting strategy clears the bar — in
which case the answer is CV, not VLM (see §9).

---

## 2. Why hit vLLM directly (not via yJarvis)

This is offline R&D, not production. Going straight to vLLM:

- gives full control over `messages`, `guided_json`, `temperature`, `model` — yJarvis's
  current `{prompt, image_base64}` facade would hide exactly the variables we're testing;
- removes a moving part (no yJarvis deploy needed to iterate);
- means the experiment's result is what *justifies* (or doesn't) building the yJarvis `B`
  endpoint. Sequencing matters: **validate the client-side approach first, build the
  backend second.**

The only vLLM-specific knob is the structured-output mechanism (`--mechanism`): `guided_json`
via `extra_body` (default, broadest) or `response_format` json_schema (newer vLLM). Confirm
which your deployment honors; a passing run with `guided_json` *is* the confirmation.

---

## 3. Experiment design — the condition ladder

Each rung adds **one** thing, so a difference between adjacent rungs isolates that one
decision. (Defined in code as `CONDITIONS`.)

| condition | k (shots) | guided_json | reasoning field | what the comparison isolates |
|---|---|---|---|---|
| `zeroshot_freetext` | 0 | off | yes (asked in prose) | **baseline** — today's `yellow_score_box` style |
| `zeroshot_guided` | 0 | on | no | guided_json reliability + perception cost of a minimal schema |
| `zeroshot_guided_reasoning` | 0 | on | yes | effect of the **reasoning-first** field under constraint |
| `fewshot3_guided_reasoning` | 3 | on | yes | effect of **3 shots** |
| `fewshot5_guided_reasoning` | 5 | on | yes | effect of **more shots** (saturation) |
| `fewshot5_guided_noreasoning` | 5 | on | no | reasoning × few-shot **interaction** |

**Key reads:**

- `zeroshot_freetext` → `zeroshot_guided_reasoning`: does the proposed `B` config beat
  today's style? (`fmt%` for H1, `ox%/oy%` for perception.)
- `zeroshot_guided` vs `zeroshot_guided_reasoning`: H2 (does the reasoning field help?).
- `zeroshot_guided_reasoning` → `fewshot3` → `fewshot5`: H3 & H4 — the **headline few-shot
  question** and where it saturates.
- `fewshot5_guided_reasoning` vs `fewshot5_guided_noreasoning`: is reasoning still earning
  its tokens once you have shots?

---

## 4. Dataset & labeling discipline

Two manifests, **same format as `sem_corner_judge`** (`[{image, answer}, ...]`):

- `eval_manifest.json` — the **held-out** test set, hand-labeled ground truth.
- `exemplar_manifest.json` — the few-shot examples, **disjoint** from eval.

Non-negotiables (these are where experiments quietly lie to you):

1. **No leakage.** Exemplar images must never appear in the eval set. Different
   wafers/sites ideally, not just different crops of the same frame.
2. **Same distribution as production.** Real SEM crops at the **same ROI / zoom /
   resolution** the live `analyze` will receive. Clean/synthetic exemplars can *hurt*.
   Store exemplar images **already ROI-cropped** (the module crops live frames; match it).
3. **Class balance.** Cover the decision boundary: aligned, single-axis offset, both-axis
   small, both-axis large, not-found — for **both** sets. A set that's 80% "aligned" makes
   overall accuracy meaningless.
4. **Size & humility.** Aim ≥ 30, ideally 50–100 eval images. With small n the confidence
   intervals are wide; report them and don't over-read point estimates.

---

## 5. Metrics — designed for closed-loop, not for a leaderboard

The downstream consumer is an alignment controller, so the **dangerous** errors aren't
"overall accuracy" — they're errors that make the loop misbehave:

- **`wrongDir%` (critical):** predicted a definite direction *opposite* to ground truth on
  either axis. In closed loop this drives the stage the wrong way → **divergence**. Counted
  only on the corner-present subset.
- **`falseAln%` (critical):** predicted `aligned=true` when it isn't → controller **stops
  correcting prematurely**. Counted on the not-aligned subset.
- `ox% / oy%`: per-axis direction accuracy — the usable perception signal.
- `magMAE`: ordinal mean-abs-error on magnitude (none→large = 0..3) — governs convergence
  speed.
- `fmt%`: schema/format compliance — H1; should be ~100% under `guided_json`.
- `corner_found` accuracy (in raw CSV): false "not found" stalls the loop.
- `align%`: headline boolean accuracy — **least informative under class imbalance**; shown
  with ±std across repeats and a Wilson 95% CI, not as a single number.
- `lat_ms`: mean call latency — the cost axis; few-shot adds image tokens × k.

> Rule of thumb when reading the table: **minimize `wrongDir%` first, then `falseAln%`,
> then `magMAE`; treat `align%` as a sanity check, not the objective.**

---

## 6. Methodology rigor

- **Determinism vs reality.** Default `temperature=0.0` for judgment. But vLLM at temp 0 is
  *not* guaranteed bit-identical (batching/kernels), so run `--repeats 3` and look at the
  `±sd` column. Near-zero sd = stable; non-trivial sd means single-shot judgments are risky
  and you may need majority-vote in production.
- **Few-shot order sensitivity.** Few-shot results depend on exemplar order; the runner uses
  a **fixed** order (`exemplars[:k]`). If a config is borderline, re-run with a shuffled
  exemplar manifest as a robustness check before trusting it.
- **Confidence intervals.** Wilson 95% CI on `align%`. If two conditions' CIs overlap
  heavily, you **cannot** claim one is better — collect more data or call it a tie.
- **One variable at a time.** That's the whole point of the ladder; don't compare
  non-adjacent rungs and attribute the gap to a single cause.

---

## 7. Data layout & how to run

```
<data-dir>/
    eval_manifest.json
    exemplar_manifest.json
    <images referenced by both manifests>
```

`answer` per image:

```json
{ "reasoning": "", "corner_found": true,
  "offset_x": "left|center|right|unknown",
  "offset_y": "above|center|below|unknown",
  "magnitude": "none|slight|moderate|large|unknown",
  "aligned": false }
```

Run:

```bash
pip install openai pillow
export VLLM_BASE_URL="http://your-vllm-host:8000/v1"
export VLLM_API_KEY="EMPTY"          # vLLM usually ignores the value
export VLM_MODEL="Qwen3.6-27B"

# full ladder, 3 repeats for variance
python run_fewshot_experiment.py --data-dir ./sem_eval --repeats 3

# just the headline few-shot question
python run_fewshot_experiment.py --data-dir ./sem_eval \
    --conditions zeroshot_guided_reasoning fewshot3_guided_reasoning fewshot5_guided_reasoning

# if your vLLM prefers the OpenAI-native json_schema path:
python run_fewshot_experiment.py --data-dir ./sem_eval --mechanism response_format
```

Outputs (in `--out-dir`, default `./experiment_out`):

- `raw_results.csv` — every call: per-item flags, latency, truncated raw response (audit).
- `summary.md` — the per-condition table, paste-ready into an ADR.

---

## 8. The runner — `run_fewshot_experiment.py`

Full script (drop-in; directly `openai` → vLLM, no yJarvis, no yMinion module deps):

```python
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

SYSTEM_REASONING = (
    "你是半導體 SEM 對齊的視覺判斷器,判斷影像中目標結構的「角」(兩條亮邊垂直相交的 L 形頂點)"
    "相對於畫面中央十字線中心的對位狀態。\n" + _OFFSET_RULES +
    "\n輸出規則:只輸出一個 JSON 物件,reasoning 欄位放最前面(先描述看到的角結構、十字線位置、"
    "角相對中心的方位,再給結論),不得有任何額外文字或 markdown。"
)
SYSTEM_NO_REASONING = (
    "你是半導體 SEM 對齊的視覺判斷器,判斷影像中目標結構的「角」(兩條亮邊垂直相交的 L 形頂點)"
    "相對於畫面中央十字線中心的對位狀態。\n" + _OFFSET_RULES +
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

ALLOWED = {
    "offset_x": {"left", "center", "right", "unknown"},
    "offset_y": {"above", "center", "below", "unknown"},
    "magnitude": {"none", "slight", "moderate", "large", "unknown"},
}
MAG_ORD = {"none": 0, "slight": 1, "moderate": 2, "large": 3}

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
              mechanism: str) -> Tuple[str, float, Optional[str]]:
    """Single call. Returns (content, latency_ms, error)."""
    t0 = time.time()
    try:
        completion = _client().chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, timeout=timeout,
            **_structured_kwargs(schema, mechanism),
        )
        content = completion.choices[0].message.content or ""
        return content, (time.time() - t0) * 1000.0, None
    except Exception as e:  # noqa: BLE001
        return "", (time.time() - t0) * 1000.0, f"{type(e).__name__}: {e}"

# ---------------------------------------------------------------------------
# Data / message building
# ---------------------------------------------------------------------------

def _data_uri(img_path: str) -> str:
    img = Image.open(img_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def _image_content(img_path: str) -> List[Dict[str, Any]]:
    return [{"type": "text", "text": USER_TURN_TEXT},
            {"type": "image_url", "image_url": {"url": _data_uri(img_path)}}]

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
        msgs.append({"role": "user", "content": _image_content(ex["_img_path"])})
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
    """Per-item flags. Directional metrics only count on the 'corner present' subset."""
    found_subset = bool(gt["corner_found"])
    not_aligned_subset = (gt["corner_found"] and not gt["aligned"])
    if pred is None:
        return {"format_ok": 0, "corner_found_ok": 0, "aligned_ok": 0,
                "ox_ok": None, "oy_ok": None, "mag_ae": None,
                "wrong_dir": (1 if found_subset else None),       # malformed = unusable = treat as bad on found subset
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
    eval_set = load_manifest(args.data_dir, "eval_manifest.json")
    exemplars = load_manifest(args.data_dir, "exemplar_manifest.json")
    if args.limit:
        eval_set = eval_set[: args.limit]
    conds = [c for c in CONDITIONS if (not args.conditions or c["name"] in args.conditions)]
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
```

---

## 9. Interpreting results — decision tree

Read the table, then follow the branch. Each outcome maps to a concrete next move (and some
say *don't* build the few-shot machinery).

**(A) Few-shot clearly wins** — `fewshot*` improve `ox%/oy%` and/or cut `wrongDir%` vs
`zeroshot_guided_reasoning`, by more than the CIs overlap.
→ Few-shot is worth its cost. **Build the `B` endpoint** (the CLAUDE_CODE_PROMPT), assemble
few-shot client-side. Ship the **smallest `k`** that captures the gain (compare `fewshot3`
vs `fewshot5`; if equal, use 3).

**(B) `guided_json + reasoning` is already strong; few-shot adds little** — `fewshot*` CIs
overlap `zeroshot_guided_reasoning`.
→ Don't pay for exemplars. **Still build `B`** (for `guided_json`'s reliability), but
**skip the few-shot exemplar machinery** entirely — simpler, cheaper, less maintenance.
This is a perfectly good result; be willing to accept it.

**(C) `guided_json` *hurts* accuracy vs `zeroshot_freetext`** — `fmt%` jumps to ~100% but
`ox%/oy%` drop, or `wrongDir%` rises.
→ The schema is over-constraining the model's reasoning. **Loosen it**: ensure the
reasoning field is present and generated *first*, consider a freer reasoning string, or
relax `additionalProperties`. Re-run. Don't ship a rigid schema that trades perception for
tidiness.

**(D) Perception-bound across *all* conditions** — `ox%/oy%` hover near chance (~33%) and/or
`wrongDir%` stays high everywhere, few-shot included.
→ The model can't resolve the corner at this image quality. **Few-shot won't save it.** The
honest answer is **CV** — your existing `sem_array_corner_detector` (NCC + autocorrelation)
— possibly with better preprocessing/resolution, with the VLM relegated to a secondary
confirmation rung later. This is the "**bet on direction, not timing**" position made
concrete by data: the CV foundation carries production now; VLM judgment waits for a model
that clears the bar.

**(E) High variance at temp 0** — non-trivial `±sd` on `align%` across repeats.
→ Server-side nondeterminism (batching/seeds). Investigate vLLM config; in production treat
a single judgment with caution — consider **majority-vote over N calls** for the alignment
decision, or accept it only above a stability threshold.

**Cross-cutting cost check:** if a few-shot config wins but `lat_ms` (or token cost) is too
high for the loop's cycle time, mitigate before shipping — reduce `k`, keep ROI crops tight,
and see the prefix-cache lever in §10.

---

## 10. Architecture recommendation — yMinion × yJarvis

The experiment validates the **client-side** strategy *before* any backend work. Assuming
branch (A) or (B), the target shape:

```
yMinion (RcpAgent repo)                                  yJarvis (backend)        vLLM
─────────────────────────                                ─────────────────        ────
NexusEngine SOP (workflow.yaml)
   └─ verification: sem_corner_judge        ── HTTP ─▶   /api/call-chat/   ──▶   Qwen3.6
        • owns prompt + schema (git)                     (stateless relay:
        • builds few-shot messages (k from exp)           messages + guided_json
        • Python recompute of `aligned`                   forwarded verbatim)
        └─ _yjarvis_chat_caller
             (retry/backoff/timing; the §7
              contract in the CLAUDE_CODE_PROMPT)
```

**The boundary that makes this scale (from the backend hand-off):** yJarvis stays
**few-shot-agnostic** — it forwards `messages` + `guided_json` and nothing more. All
few-shot logic, exemplars, prompts, and schemas live **per-module, client-side, in git**.
That is exactly what lets one `/api/call-chat/` endpoint serve `sem_corner_judge`,
`yellow_score_box_judge`, and every future judge without backend changes. Resist any pull to
put exemplar/prompt logic in yJarvis.

**Recommended sequencing:**

1. **This experiment** → decide (A)/(B)/(C)/(D) with data; record as an ADR.
2. If (A)/(B): **build `B` endpoint** via the CLAUDE_CODE_PROMPT (discovery-first, coexists
   with `/api/call-agent/`).
3. Write **`_yjarvis_chat_caller`** (client transport; mirror `_yjarvis_caller`'s
   retry/backoff/timing, return `{content, latency_ms, error, attempts}`).
4. Wire **`sem_corner_judge`** to it; ship the `k` the experiment justified.
5. Keep this experiment as a **regression eval** (§11).

**Cost & governance levers (decide once few-shot is in):**

- **ROI crop** exemplars and live frames (already in the module) — the cheapest token win.
- **Smallest viable `k`** — taken from the saturation comparison, not a guess.
- **Prefix caching on vLLM.** The few-shot prefix (system + exemplar turns) is *constant*
  across calls; vLLM prefix/KV caching can reuse it, cutting per-call cost substantially.
  Order the messages so the constant prefix is stable and verify cache hits — this is the
  single biggest scaling lever for production few-shot.
- **Audit.** `raw_results.csv` is the offline audit trail; in production, log the same
  per-call fields (offsets, magnitude, recompute flag) for the alignment dashboard.

**Where this sits in the autonomy ladder:** the closed-loop corner judgment is the **L3**
rung (visual closed-loop) above today's **L2** (unattended execution on the CV detector).
This experiment **de-risks** that rung. The CV detector remains the safe baseline; the VLM
judge is a **modular upgrade**, not a replacement — so a negative result (branch D) costs you
nothing structurally, it just keeps you on CV until the model is ready.

---

## 11. Make it a permanent regression eval

Don't throw this away after one run:

- **Re-run on every vLLM/model upgrade** (Qwen3.6 → next, or vLLM version bumps). Structured
  output and perception both shift across versions; `summary.md` over time is your evidence.
- **Version the eval set** alongside the module (same git discipline as prompts/exemplars).
- **Gate changes on it.** Any change to the prompt, schema, or `k` re-runs the ladder; a
  regression in `wrongDir%`/`falseAln%` blocks the change.
- **Feed ADRs.** Each decision (build B? ship few-shot? what k? stay on CV?) cites a
  `summary.md` — turning "we think few-shot helps" into "few-shot cut wrongDir from X% to
  Y% (95% CI …) at k=3, N=…".
