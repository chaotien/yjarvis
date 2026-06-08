# Coordinate-Regression Experiment — SEM Corner Localization

> Sister experiment to `../corner_judge_categorical/`. The categorical one tests whether the
> VLM can judge **direction** (left/right/above/below) of a corner relative to a marker. This
> one tests whether it can output **coordinates** of the corner directly. Different output
> type → different hypotheses, different schema, different metrics, but the same direct-vLLM
> discipline (no yJarvis), same ladder pattern, same statistical rigor.

---

## 1. What changed and why this is a separate experiment

| | Categorical (sibling folder) | This experiment |
|---|---|---|
| Output | enum offsets + `aligned` boolean | `(x, y)` integer coordinates |
| Failure that diverges the loop | `wrongDir%` (axis flip) | `wrongQuadrant%` (both axes flipped relative to image center) |
| Production grading | aligned-or-not boolean | distance to GT, possibly inside a bbox |
| Few-shot demonstration | exemplar's `answer` JSON | **exemplar image has a green crosshair drawn at the GT corner**; the matching `(x, y)` is the assistant answer |
| Eval image | the live SEM crop | **the live SEM crop, with no green crosshair** — the VLM must locate the corner from SEM content alone |

The visual-demo few-shot is the key new lever. Exemplars literally *show* the model "this
is what a correctly located corner looks like, and these are the coordinates of where the
green cross sits." Whether that demonstration generalizes to unmarked eval images is the
central question of this experiment.

**Null we must accept:** if zeroshot is bad and few-shot doesn't recover, the VLM cannot do
coord regression on this image quality; the answer is **CV (template matching / NCC)** for
localization too, not VLM.

---

## 2. Hypotheses

- **H1 (format compliance).** `guided_json` drives `fmt%` to ~100%, mirroring the
  categorical experiment.
- **H2 (zeroshot is the floor).** Without exemplars, the model has no anchor for "what
  coords correspond to what's in the image." Expect low `hit@τ%` and high `wrongQ%`.
- **H3 (visual-demo few-shot is the real lever).** Exemplars that *show* where the corner
  is (via green cross) should sharply raise `hit@5%` and cut `wrongQ%`.
- **H4 (saturation).** Some small `k` past which extra shots stop helping.
- **H5 (coord format).** `pixel` vs `norm1000` (Qwen-VL convention) may give materially
  different accuracy. Tested only at the best k to keep the matrix small.
- **H6 (reasoning under constraint).** Reasoning-first should not hurt, and may help under
  schema constraint — but with few-shot demonstrations doing the heavy lifting, it may stop
  earning its tokens. The interaction is what `fewshot5_guided_noreasoning_pixel` measures.

---

## 3. Condition ladder

Defined in `CONDITIONS` in `run_corner_locate_experiment.py`. Each rung isolates one
decision.

| condition | k | guided | reasoning | coord | what it isolates |
|---|---|---|---|---|---|
| `zeroshot_freetext_pixel` | 0 | off | yes | pixel | baseline; parse coords from prose |
| `zeroshot_guided_pixel` | 0 | on | no | pixel | guided_json reliability cost |
| `zeroshot_guided_reasoning_pixel` | 0 | on | yes | pixel | H6 zeroshot side |
| `fewshot3_guided_reasoning_pixel` | 3 | on | yes | pixel | H3: visual demo helps |
| `fewshot5_guided_reasoning_pixel` | 5 | on | yes | pixel | H4: saturation upper bound |
| `fewshot5_guided_reasoning_norm` | 5 | on | yes | norm1000 | H5: format effect at best k |
| `fewshot5_guided_noreasoning_pixel` | 5 | on | no | pixel | H6: reasoning × few-shot |

**Key reads:**

- `zeroshot_guided_reasoning_pixel` → `fewshot3_*` → `fewshot5_*`: the headline lever.
- `fewshot5_*_pixel` vs `fewshot5_*_norm`: coord format effect (H5).
- `fewshot5_guided_reasoning_pixel` vs `fewshot5_guided_noreasoning_pixel`: does reasoning
  still earn its tokens once the demonstrations are present? (H6)

---

## 4. Dataset & labeling discipline

Two manifests, sibling layout to the categorical experiment:

- `eval_manifest.json` — held-out test, **no green cross on images**.
- `exemplar_manifest.json` — few-shot demonstrations, **green cross drawn at the true
  corner location** on each exemplar image.

Entry schema:

```json
{
  "image": "f.png",
  "corner_x": 487, "corner_y": 312,
  "bbox": [470, 295, 505, 330],
  "reasoning": "影像中目標結構的角在右側偏上,綠色十字標示在 (487, 312)。"
}
```

- `corner_x`, `corner_y` — required for both eval and exemplar; pixel coords, top-left
  origin, y increases downward.
- `bbox` — optional; if present, `hit_bbox%` is computed in addition to `hit@τ%`.
- `reasoning` — optional; only consumed in reasoning-on conditions for exemplar answers.
  If absent, the script synthesizes a minimal Chinese sentence so the few-shot pattern still
  shows the reasoning-first convention.

### 4.1 Labeling invariants (the prompt assumes these — make data match)

Three non-negotiables, parallel in spirit to §4.1 of the categorical doc:

1. **Exemplar images have a *green* crosshair drawn at the true corner.** Not approximate,
   not nearby — the cross center must equal `(corner_x, corner_y)` in pixels. If the cross
   you drew and the coords you typed disagree, the model sees a confused demonstration and
   few-shot loses its point.

2. **Eval images do NOT have any green crosshair.** If a green cross leaks into eval, the
   model is solving a different (much easier) task and the numbers don't transfer to
   production.

3. **A *red* crosshair may appear in any image and is to be ignored.** It represents the
   camera-center marker. The prompt explicitly tells the model to ignore red. Worth an
   informal ablation: split eval by red-present vs red-absent to check the model isn't
   secretly anchoring on it.

Plus the data-prep basics from the categorical experiment still apply: same distribution as
production, class balance (cover corners across the frame and `corner_found=false` cases),
≥ 30 ideally 50–100 eval items, exemplars disjoint from eval.

---

## 5. Metrics — closed-loop safety first, then precision

The downstream consumer is still an alignment controller. The failure modes change form
but not in spirit:

- **`wrongQuadrant%` (critical).** Predicted point lies in the diagonally opposite quadrant
  of the image relative to the GT (both x and y on the wrong side of image center). The
  controller drives the stage in the wrong direction on both axes → divergence. Computed
  with a 5%-of-image-dimension dead-band around center to avoid noise.
- **`hit@τ%` (headline precision).** Fraction of items where Euclidean distance from
  prediction to GT is ≤ `τ × min(W, H)`. Default τ ∈ {1%, 2%, 5%, 10%}. `hit@5%` is the
  default "is it actually useful" number; `hit@2%` and `hit@1%` are precision stretches.
- **`hit_bbox%`** (when GT bbox supplied) — task-defined acceptance.
- **`meanL2%`, `medianL2%`** — error magnitude in % of min(W,H). Median is robust to a few
  catastrophic predictions; mean is sensitive to them, which is exactly what you want when
  scanning for outlier failure modes.
- **`fmt%`** — schema/format compliance; should be ~100% under `guided_json`.
- **`lat_ms`** — cost axis.

> Reading the table: **minimize `wrongQ%` first**, then **maximize `hit@τ%`** (start with
> τ=5%, tighten if production needs it), then look at `medianL2%`. Don't read `meanL2%` in
> isolation — it's a smell detector.

### Wilson CI

95% Wilson confidence intervals on each `hit@τ%`. Same logic as the categorical experiment:
small n + extreme proportions need Wilson, not normal-approximation, to avoid lying.

---

## 6. Methodology — what carries over from the categorical experiment

Don't repeat what's already there. The following are settled and used identically:

- `temperature=0.0`, `--repeats 3` to characterize batch-induced variance.
- Fixed exemplar order (`exemplars[:k]`); shuffle and re-run as a robustness check if a
  config is borderline.
- One-variable-per-rung ladder; don't compare non-adjacent rungs and attribute to a single
  cause.
- `guided_json` vs `response_format` — supported via `--mechanism` exactly as before.
- Same retryable-error allow-list (timeouts, 5xx, connection drops); schema/auth errors
  surface immediately.
- Same data URI cache for exemplar images.

---

## 7. Data layout & how to run

```
corner_locate_coords/<data-dir>/
    eval_manifest.json
    exemplar_manifest.json
    <images...>
```

```bash
pip install openai pillow
export VLLM_BASE_URL="http://your-vllm-host:8000/v1"
export VLLM_API_KEY="EMPTY"
export VLM_MODEL="Qwen3.6-27B"

# smoke test
python run_corner_locate_experiment.py --data-dir ./sem_loc_eval --limit 5 --repeats 1 \
    --conditions zeroshot_guided_reasoning_pixel

# full ladder, 3 repeats
python run_corner_locate_experiment.py --data-dir ./sem_loc_eval --repeats 3

# tighten tolerances to production target
python run_corner_locate_experiment.py --data-dir ./sem_loc_eval --tolerances 0.005 0.01 0.02
```

Outputs into `--out-dir` (default `./experiment_out`): `raw_results.csv` and `summary.md`.

---

## 8. Interpreting results — decision tree

**(A) Visual-demo few-shot clearly wins.** `fewshot{3,5}_*_pixel` lift `hit@5%` and drop
`wrongQ%` vs `zeroshot_guided_reasoning_pixel` by more than CIs overlap.
→ The approach works. Pick the smaller `k` that captures the gain. Decide pixel vs
norm1000 from `fewshot5_*_pixel` vs `fewshot5_*_norm`. Build the yJarvis chat endpoint.

**(B) Few-shot doesn't recover and zeroshot is also bad.** `hit@5%` everywhere low, `wrongQ%`
high.
→ The VLM cannot do coord regression at this image quality. Don't ship coords from VLM.
**Fall back to CV** (template matching / NCC against a corner template). The VLM may still
be useful as a downstream verifier on CV proposals.

**(C) `guided_json` hurts vs `freetext`.** `fmt%` jumps to ~100% but `hit@τ%` drops or
`wrongQ%` rises.
→ Schema is over-constraining. Try the reasoning-first variant; if that doesn't fix it,
the rigid integer schema is the problem — try freer parsing or a different coord format.

**(D) Format mismatch (pixel beats norm1000 by a lot, or vice versa).** Wide gap between
`fewshot5_*_pixel` and `fewshot5_*_norm`.
→ The model has an opinion about coord conventions. Use the winning format in production,
note it in the ADR.

**(E) High variance at temp 0.** Non-trivial `hit5±sd`.
→ Same as categorical experiment: vLLM batching nondeterminism. Either majority-vote or
gate on stability in production.

---

## 9. Architecture & sequencing

Same shape as the categorical experiment (see `../corner_judge_categorical/fewshot_experiment.md` §10).
Specifically:

- yJarvis stays few-shot-agnostic — forwards `messages` + `guided_json`.
- Prompts, schemas, exemplar selection, `k` all live in the yMinion module
  (`sem_corner_locate` or similar) in git.
- This experiment validates the client-side strategy *before* any backend change.
- If branch (A) holds: build/extend the `/api/call-chat/` endpoint, write a
  `_yjarvis_chat_caller`, wire the module.
- Keep this experiment as a permanent regression eval (re-run on every vLLM/model upgrade).

The localization module is **complementary** to the categorical judge, not a replacement
for it. Localization tells the controller *where to drive*; categorical judge tells it
*whether to keep driving*. They can be staged: localize → drive → categorical re-check at
the new position → loop until aligned.

---

## 10. Open questions to revisit after the first run

These are not blockers for v1 of the experiment — record findings on the first run, then
decide whether they justify another condition:

- **Red-cross effect.** Split eval into red-present vs red-absent subsets; does `wrongQ%`
  differ? If yes, the prompt's "ignore red" instruction isn't holding.
- **Image resolution headroom.** If `hit@5%` is decent but `hit@1%` is near zero, the
  ceiling may be model coord-resolution, not perception. Try a higher-resolution version
  of the same crops.
- **Corner-not-present rows.** If you include `corner_found=false` items in eval, track
  `false-not-found%` separately — that's a controller stall, not divergence.
- **Order sensitivity.** Re-run a borderline winning config with shuffled exemplar order
  to confirm the gain isn't an artifact of which 5 exemplars are first.
