# Few-shot Validation — def/ref Patch single-vs-cluster Judgment

> Sister experiment to `../corner_judge_categorical/`. That one judges a **single** SEM image
> (corner direction). This one judges a **pair** of small patches — `def_patch` (under test)
> and `ref_patch` (reference) — and answers a **binary** question: is the region that's
> brighter in def-vs-ref **one isolated small dot (`single`)** or **multiple nearby dots / a
> blob / a broad area (`cluster`)**? Same direct-vLLM discipline (no yJarvis), same condition
> ladder, same statistical rigor — but **two images per turn** and a binary metric set.
> Runnable: `run_fewshot_experiment.py`. Pipeline smoke-test data: `make_synthetic_demo.py`.

---

## 1. What changed and why this is a separate experiment

| | Corner-judge (sibling) | This experiment |
|---|---|---|
| Input per item | one SEM image | **two patches**: `def_patch` + `ref_patch` |
| Output | offset enums + `aligned` | `label ∈ {single, cluster}` (+ descriptive `morphology`) |
| What the model must do | locate a corner vs a marker | **compare** def to ref, classify the *diff*'s morphology |
| Headline metric | `wrongDir%` / `falseAln%` | `balAcc%` + the two miss directions (`missClu%`, `missSgl%`) |
| Few-shot demonstration | exemplar image + answer JSON | **exemplar PAIR (def+ref) + answer JSON** |

The new lever is the **paired-image turn**. Each user turn carries *two* images in a fixed
order (def first, ref second) and the prompt text pins which is which. Few-shot exemplars
demonstrate the *comparison*, not just an appearance — "given these two patches, this diff
counts as single / cluster."

**Null we must accept:** if zero-shot is poor and few-shot doesn't recover, the VLM can't
resolve the def-vs-ref diff morphology at this patch size/quality → the answer is **CV**
(difference image + connected-component / blob analysis), not VLM.

---

## 2. The task definition (this *is* the spec — labeling must match)

The judgment is about the **difference** def-minus-ref, not the absolute brightness of def:

- **single** — the extra-bright region in def (relative to the same spot in ref) is **one
  relatively independent, sharply-bounded small round dot**: small area, isolated, no other
  co-brightening points around it.
- **cluster** — the extra-bright region is **multiple nearby dots flickering together**, OR
  **one large blob (一大坨)**, OR **a broad area lighting up (整片)**. The rule is simple:
  **anything that is not a single isolated small dot is `cluster`.**

`morphology` (a descriptive field, used to build the reasoning path and to recompute `label`):

| morphology | meaning | → label |
|---|---|---|
| `single_dot` | one isolated small round dot | `single` |
| `multi_dots` | several nearby co-brightening dots | `cluster` |
| `large_blob` | one big blob, fused boundary | `cluster` |
| `broad_area` | a whole region brightens together | `cluster` |
| `unknown` | can't tell / no clear added bright region | (still pick nearest single/cluster) |

`brighter_region_found` (bool): is there a region in def clearly brighter than ref at all? If
the two patches are essentially identical, `false`.

### Labeling invariants (the prompt assumes these — break them and the experiment mismeasures)

1. **Judge the diff, not absolute brightness.** A region that is bright in *both* def and ref
   is **not** a defect signal — ignore it. Label only the morphology of what def adds *over*
   ref. The prompt says this explicitly (`_DOMAIN_KNOWLEDGE`); your labels must too.
2. **def is first, ref is second.** The two images go into each turn in this fixed order and
   the prompt text states it. If your data prep swaps them, the model compares the wrong way.
   Keep def/ref consistent across exemplars *and* eval.
3. **Co-registered pair (confirmed for this dataset).** The "imagine subtracting ref from def"
   instruction assumes the two patches are co-registered (same location, same size) — which is
   the case here. Keep new data co-registered too; a mis-aligned pair makes a meaningful diff
   impossible even for a perfect model.
4. **`single` means *isolated* and *small*.** A single dot that is large enough to read as a
   blob is `cluster` (`large_blob`). Two dots, however small, are `cluster` (`multi_dots`).
   Decide the size/separation threshold once, write it into your labeling guide, and apply it
   uniformly — borderline cases are where binary tasks quietly rot.

> **Tiny patches need upscaling.** The patches are small (e.g. 64×64) and a `single` target can
> be ~5px — sent raw, the VLM's internal resizing may erase it. Client-side upscaling is a
> separate experiment axis; the interpolation choice interacts with the single-vs-cluster
> decision (smoothing can merge nearby dots into a blob → false `cluster`). See
> **`preprocess_experiment.md`** (and `--ladder preprocess`) for that experiment.

---

## 3. Condition ladder

Defined in `CONDITIONS`. Each rung adds **one** thing, so an adjacent-rung difference isolates
that decision. The user expects ~6 exemplars, so the few-shot rungs are **k=3** and **k=6**.

| condition | k | guided | reasoning | what it isolates |
|---|---|---|---|---|
| `zeroshot_freetext` | 0 | off | yes (prose) | baseline; parse JSON from prose |
| `zeroshot_guided` | 0 | on | no | guided_json reliability + perception cost of a bare schema |
| `zeroshot_guided_reasoning` | 0 | on | yes | the reasoning-first field under constraint |
| `fewshot3_guided_reasoning` | 3 | on | yes | effect of 3 paired exemplars |
| `fewshot6_guided_reasoning` | 6 | on | yes | effect of 6 paired exemplars (saturation) |
| `fewshot6_guided_noreasoning` | 6 | on | no | reasoning × few-shot interaction |

**Key reads:** `zeroshot_guided_reasoning → fewshot3 → fewshot6` is the headline few-shot
question and where it saturates; `fewshot6_guided_reasoning` vs `..._noreasoning` asks whether
the reasoning field still earns its tokens once you have 6 paired demos.

---

## 4. Metrics — designed for an imbalanced binary decision

`label` accuracy alone lies under class imbalance (if 80% of pairs are `cluster`, always
answering `cluster` scores 80%). So the headline is **balanced accuracy**, with both error
directions surfaced:

- **`balAcc%` (headline):** mean of `single` recall and `cluster` recall — robust to imbalance.
- **`missSgl%` (PRIORITY — single-miss is the worse error for this use case):** fraction of true
  `single` pairs called `cluster`. Drive this down first.
- **`missClu%` (secondary):** fraction of true `cluster` pairs called `single`.
- `sRec% / cRec%`: per-class recall (the two numbers `balAcc%` averages).
- `acc%`: raw accuracy, with a **Wilson 95% CI** — a sanity check, not the objective.
- `consist%`: does `label` agree with `morphology` (via the `single_dot→single`,
  else→`cluster` map)? Low values mean the model contradicts itself — a prompt/schema smell.
- `fmt%`: schema compliance (H1; ~100% under guided_json). `lat_ms`: cost (two images × k).

> **Priority for this dataset: `missSgl%`.** Single-miss (a true `single` reported as
> `cluster`) is the worse error here, so the runner prints `missSgl%` before `missClu%` and
> you should optimize it first — but watch that pushing it down doesn't blow up `missClu%`
> (the trivial "always answer single" degenerate). Both are reported; `balAcc%` guards the
> balance.

The runner recomputes `balAcc%`'s per-repeat value to give a `±sd` across `--repeats`, and a
Wilson CI on `acc%`. Small-n: read the CI width, don't over-read point estimates.

---

## 5. How the paired-image few-shot is assembled (the one real novelty)

`build_messages` (and `_pair_content`) produce, for `k` exemplars with reasoning on:

```
system                                   # task + domain knowledge + label rules (constant)
user      [text, def_img_1, ref_img_1]   # exemplar 1: the PAIR
assistant {"reasoning":…, "morphology":"single_dot", "label":"single"}
user      [text, def_img_2, ref_img_2]   # exemplar 2
assistant {…}
…                                        # up to k exemplars
user      [text, def_img_q, ref_img_q]   # the pair to classify
```

Non-obvious points:

1. **Two `image_url` parts in one user turn, def then ref**, with the order stated in
   `USER_TURN_TEXT`. Image order is the *only* signal of which patch is which — keep it fixed.
2. **The same `USER_TURN_TEXT` on every turn** (exemplars and query) so the model treats the
   pair-comparison request as a stable, repeated pattern.
3. **`SYSTEM_REASONING` / `SYSTEM_NO_REASONING` are constants** (not assembled per-call) so the
   long prefix (system + exemplar turns) is byte-stable → vLLM prefix/KV cache can reuse it.
   With six paired exemplars (12 images), prefix caching is the biggest per-call cost lever.
4. **No-reasoning conditions strip `reasoning` from exemplar answers** (`ans.pop`), so the
   demonstrated answer shape matches the schema the model is constrained to.

See the sibling `../corner_judge_categorical/fewshot_experiment_zh.md` §1 for the deeper "why
user/assistant alternation" treatment — it applies verbatim here, just with paired images.

---

## 6. Data layout & how to run

```
<data-dir>/
    eval_manifest.json        # [{ "def_patch": "...", "ref_patch": "...", "answer": {...} }, ...]  (HELD OUT)
    exemplar_manifest.json    # same shape, DISJOINT from eval (the ~6 few-shot pairs)
    <patch PNGs>
```

`answer` per pair:

```json
{ "reasoning": "", "brighter_region_found": true,
  "morphology": "single_dot|multi_dots|large_blob|broad_area|unknown",
  "label": "single|cluster" }
```

Run:

```bash
pip install openai pillow
export VLLM_BASE_URL="http://your-vllm-host:8000/v1"
export VLLM_API_KEY="EMPTY"
export VLM_MODEL="Qwen3.6-27B"

# (optional) generate synthetic data to validate the pipeline before you have real labels:
python make_synthetic_demo.py --out-dir ./synthetic_demo
python run_fewshot_experiment.py --data-dir ./synthetic_demo --repeats 1

# real run — full ladder, 3 repeats for variance:
python run_fewshot_experiment.py --data-dir ./patch_eval --repeats 3

# just the headline few-shot question:
python run_fewshot_experiment.py --data-dir ./patch_eval \
    --conditions zeroshot_guided_reasoning fewshot3_guided_reasoning fewshot6_guided_reasoning

# if your vLLM prefers the OpenAI-native json_schema path:
python run_fewshot_experiment.py --data-dir ./patch_eval --mechanism response_format
```

Outputs (in `--out-dir`, default `./experiment_out`): `raw_results.csv` (per-call audit:
gt/pred label, flags, latency, truncated raw) and `summary.md` (the per-condition table).

---

## 7. Interpreting results — decision tree

- **(A) Few-shot clearly wins** — `fewshot*` raise `balAcc%` and/or cut your priority miss
  direction vs `zeroshot_guided_reasoning`, beyond CI overlap. → Few-shot is worth it; ship the
  **smallest k** that captures the gain (compare 3 vs 6; if equal, use 3).
- **(B) guided+reasoning already strong; few-shot adds little** — CIs overlap. → Skip the
  exemplar machinery; keep guided_json for reliability. A fine result.
- **(C) guided_json *hurts* vs freetext** — `fmt%`→100% but `balAcc%` drops. → Schema is
  over-constraining; ensure reasoning is generated first, loosen the schema, re-run.
- **(D) Perception-bound everywhere** — `balAcc%` near 50% (chance) across all rungs. →
  Few-shot won't save it. Use **CV**: difference image (def−ref) → threshold → connected
  components → classify by blob count/area/compactness. Revisit the VLM on the next model.
- **(E) High variance at temp 0** — non-trivial `±sd` on `balAcc%`. → vLLM nondeterminism;
  consider majority-vote over N calls in production, or a stability gate.

The minimum viable experiment is just **(A)/(B)/(D)**: does few-shot beat zero-shot, and is
either good enough to use? Everything else is rigor on top of that question.

---

## 8. Relationship to yJarvis (same as the sibling)

yJarvis stays a stateless relay: it forwards `messages` + `guided_json` to vLLM and returns
the response verbatim. All task logic — prompt, schema, the paired-image assembly, exemplar
selection, `k` — lives client-side in git (the consuming module). That is exactly what lets one
`/api/call-chat/` endpoint serve `sem_corner_judge`, this patch judge, and every future judge
without backend changes. This experiment **decides whether the few-shot machinery is worth
building** for this task before any backend work — measure first, build second.
