# Preprocessing Experiment — Upscaling tiny patches so the VLM can see a ~5px target

> Companion to `fewshot_experiment.md`. That experiment varies the **prompt** (k / guided /
> reasoning) at a fixed image. This one holds the prompt fixed (k=6, guided, reasoning) and
> varies the **image preprocessing** — because the patches are tiny (e.g. 64×64) and a
> `single` target can be only ~5px. Sent raw, the VLM's internal resizing can wipe out a 5px
> dot; how you upscale client-side becomes the dominant lever. Same runner
> (`run_fewshot_experiment.py`), selected with `--ladder preprocess`.

---

## 1. Why preprocessing is a first-class axis here (not a detail)

A 64×64 patch is ~4096 pixels. Vision LLMs (e.g. Qwen-VL family) tokenize images into patch
tokens with a *minimum* and *maximum* pixel budget and will **resize** the input to fit. A
64×64 image is at/under the floor, so the server may upsample it with its own (unknown)
interpolation, or represent it with very few visual tokens — either way a 5px dot can blur into
the background or vanish. By upscaling on the **client** to a size comfortably inside the
model's token budget (e.g. 512×512), you control exactly what the model sees and give the
target enough tokens to register.

But upscaling is not free of judgment, and the interpolation choice **interacts with this
specific task**:

- **Nearest** preserves hard edges and the *discreteness* of nearby dots — three separate dots
  stay three separate blocks. Good for keeping `multi_dots` distinguishable from `single_dot`,
  and a `single` dot from a `large_blob`.
- **Bilinear / bicubic / lanczos** smooth. Smoothing can **merge two nearby dots into one
  blob** (turning a true `cluster`/`multi_dots` into something that reads as `single` — a
  `missSgl`-adjacent failure if you mislabel, or a `missClu` if it reads as one big dot), and
  can also soften a genuine `single` dot's edge so it reads as a fuzzy blob → `cluster`. Either
  way, smoothing can **change the morphology the model perceives**. That is the central
  hypothesis this experiment tests.

So the question is not just "upscale yes/no" but "**which upscale factor and interpolation
preserve the single-vs-cluster morphology best, at acceptable token cost?**"

---

## 2. Hypotheses

- **P1 (raw is the floor).** `prep_raw_x1` (64×64 as-is) has the worst `balAcc%` — the target
  is too small for the model to resolve reliably.
- **P2 (upscaling helps, with diminishing returns).** `x4 → x8` improves `balAcc%`; somewhere a
  factor stops adding signal and only adds tokens/latency (`x12`).
- **P3 (interpolation matters for morphology).** `nearest` ≥ `lanczos` on `balAcc%`, and the
  gap shows up specifically in the miss directions: lanczos should raise `missClu%` (clusters
  smoothed into a single blob) and/or `missSgl%` (a single dot smeared into a blob).
- **P4 (contrast helps subtle diffs).** If the def-vs-ref GLV gap is small, a **joint**
  percentile stretch (`prep_x8_nearest_contrast`) raises recall of the brighter region — but
  only if applied jointly (see §4); an independent per-image stretch would *destroy* the diff.
- **Null to accept:** if even the best preprocessing leaves `balAcc%` near chance, the target
  is below what the VLM can resolve at this source resolution → the answer is **CV** on the
  difference image (def−ref → threshold → connected components → classify by blob count/area).

---

## 3. The preprocess ladder

Defined in `PREP_CONDITIONS` (all at the fixed best prompt: k=6, guided, reasoning):

| condition | scale | size (64→) | interp | contrast | what it isolates |
|---|---|---|---|---|---|
| `prep_raw_x1` | 1 | 64 | nearest | no | baseline floor (P1) |
| `prep_x4_nearest` | 4 | 256 | nearest | no | moderate upscale (P2) |
| `prep_x8_nearest` | 8 | 512 | nearest | no | larger upscale (P2) |
| `prep_x8_lanczos` | 8 | 512 | lanczos | no | interpolation effect (P3) — vs `prep_x8_nearest` |
| `prep_x8_nearest_contrast` | 8 | 512 | nearest | yes | joint contrast effect (P4) |
| `prep_x12_nearest` | 12 | 768 | nearest | no | saturation / cost upper end (P2) |

**Key reads (one variable at a time):**
- `prep_raw_x1 → x4 → x8 → x12`: the upscale curve. Pick the knee.
- `prep_x8_nearest` vs `prep_x8_lanczos`: interpolation, holding scale fixed (P3).
- `prep_x8_nearest` vs `prep_x8_nearest_contrast`: contrast, holding scale+interp fixed (P4).

`lat_ms` is part of the result here, not a footnote: 64→512 is 64× the pixels, so larger
factors cost more visual tokens and latency. The winner is the **smallest** prep that captures
the `balAcc%`/`missSgl%` gain.

---

## 4. Preprocessing invariants (get these wrong and you mismeasure)

1. **Same preprocessing for exemplars and eval.** The runner applies the condition's `prep` to
   *both* the few-shot exemplar pairs and the eval pair. If exemplars were raw and eval were
   upscaled, the demonstration wouldn't match the question. (Handled automatically; don't
   bypass it by pre-resizing only some images on disk.)
2. **Joint contrast, never per-image.** `--contrast` / `prep…_contrast` computes ONE percentile
   LUT over **both** patches combined and applies it to both (`_joint_contrast`). This is
   essential: an independent stretch per image would re-normalize def and ref separately and
   **erase the very brightness difference** the task depends on. If you add your own contrast
   step, keep it joint.
3. **Upscale, don't re-crop.** Upscaling must not change the field of view or re-center —
   def and ref must stay co-registered after resize (a pure integer-factor resize preserves
   this; cropping/padding would not).
4. **PNG only.** Output is PNG (lossless). JPEG compression artifacts around a 5px bright dot
   would themselves look like structure and corrupt the morphology signal.

---

## 5. How to run (two stages)

```bash
pip install openai pillow
export VLLM_BASE_URL="http://your-vllm-host:8000/v1"
export VLLM_API_KEY="EMPTY"
export VLM_MODEL="Qwen3.6-27B"

# Stage 1 — find the best preprocessing (prompt fixed at k=6):
python run_fewshot_experiment.py --data-dir ./patch_eval --ladder preprocess --repeats 3

# Stage 2 — run the PROMPT ladder at the winning preprocessing (e.g. x8 nearest):
python run_fewshot_experiment.py --data-dir ./patch_eval --ladder prompt \
    --upscale 8 --interp nearest --repeats 3

# ad-hoc: force any preprocessing onto any ladder via the global override flags
python run_fewshot_experiment.py --data-dir ./patch_eval --ladder prompt \
    --upscale 8 --interp nearest --contrast
```

The `--upscale` / `--interp` / `--contrast` flags **override** the per-condition `prep` for
every selected condition — that is how Stage 2 pins the whole prompt ladder to one chosen
preprocessing. Without them, each `prep_*` condition uses its own built-in setting.

Outputs carry a `prep` column (e.g. `s8-nearest-c0`) in both `summary.md` and
`raw_results.csv`, so preprocessing is always visible next to the metrics.

> Validate the plumbing first with synthetic data — `make_synthetic_demo.py` draws 64×64 pairs
> with ~5px single dots and multi-dot/blob/area clusters, the same regime as the real task:
> `python make_synthetic_demo.py --out-dir ./synthetic_demo && \
> python run_fewshot_experiment.py --data-dir ./synthetic_demo --ladder preprocess --repeats 1`

---

## 6. Reading the result

1. **Find the upscale knee** on `balAcc%` across `raw → x4 → x8 → x12`. Beyond the knee you pay
   `lat_ms` for nothing.
2. **Check interpolation** at the knee scale (`nearest` vs `lanczos`). If lanczos raises
   `missClu%` or `missSgl%`, smoothing is destroying morphology → use nearest.
3. **Check contrast** (`…_contrast` vs not). Helps only if the raw diff is faint; if it adds
   nothing, drop it (one less moving part).
4. **Priority error is `missSgl%`** (single-miss is worse for this use case): among
   preprocessings with comparable `balAcc%`, prefer the one with the lowest `missSgl%` — but
   reject any that wins `missSgl%` by collapsing to "always single" (watch `missClu%`).
5. **Lock it in.** Take the smallest/cheapest prep that holds the `balAcc%` and `missSgl%` gain,
   then run the prompt ladder (Stage 2) at that prep to choose `k`. Record both in the ADR:
   *"prep = x8 nearest; k = 3; missSgl X%→Y% (95% CI …)."*

If the curve never clears chance, stop here — this is the **CV** branch (difference image +
connected-component analysis), not a VLM task at this source resolution.
