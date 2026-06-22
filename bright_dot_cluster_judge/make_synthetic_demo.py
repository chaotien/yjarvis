#!/usr/bin/env python3
"""
make_synthetic_demo.py — generate a tiny synthetic def/ref patch dataset for sanity checks.

This is NOT real data. It exists so you can verify the whole pipeline (manifest loading,
paired-image message assembly, guided_json call, scoring) end-to-end *before* you have your
real labeled patches. Each pair shares a near-identical noisy gray background (the "GLV mostly
overlaps" property); the def_patch then adds either:

    single  — one small bright round dot
    cluster — several nearby bright dots, OR one large blob, OR a broad bright band

Run:
    python make_synthetic_demo.py --out-dir ./synthetic_demo
    # then run the experiment against it (will call vLLM if VLLM_BASE_URL is reachable):
    python run_fewshot_experiment.py --data-dir ./synthetic_demo --repeats 1

The generated <out-dir> contains eval_manifest.json, exemplar_manifest.json and the PNGs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

from PIL import Image, ImageDraw

PATCH = 64       # patch side length in px
BG_MEAN = 70     # background gray level
NOISE = 10       # background noise sigma


def _base_pair(rng: random.Random):
    """A def/ref pair sharing the same noisy background (so GLV mostly overlaps; only the
    added bright region is the real diff)."""
    px = [min(255, max(0, int(rng.gauss(BG_MEAN, NOISE)))) for _ in range(PATCH * PATCH)]
    ref = Image.new("L", (PATCH, PATCH)); ref.putdata(px)
    px2 = [min(255, max(0, v + int(rng.gauss(0, 3)))) for v in px]  # near-identical, not a clone
    deff = Image.new("L", (PATCH, PATCH)); deff.putdata(px2)
    return deff, ref


def _draw(img: Image.Image, morphology: str, rng: random.Random) -> None:
    """Draw the extra-bright region into the def_patch according to `morphology`."""
    d = ImageDraw.Draw(img)
    if morphology == "single_dot":
        cx, cy, r = rng.randint(20, 44), rng.randint(20, 44), rng.randint(2, 4)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rng.randint(210, 255))
    elif morphology == "multi_dots":
        cx, cy = rng.randint(22, 42), rng.randint(22, 42)
        for _ in range(rng.randint(3, 6)):
            ox, oy, r = rng.randint(-8, 8), rng.randint(-8, 8), rng.randint(2, 3)
            d.ellipse([cx + ox - r, cy + oy - r, cx + ox + r, cy + oy + r], fill=rng.randint(200, 255))
    elif morphology == "large_blob":
        cx, cy, r = rng.randint(24, 40), rng.randint(24, 40), rng.randint(8, 13)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rng.randint(190, 240))
    else:  # broad_area
        y0 = rng.randint(18, 30)
        d.rectangle([6, y0, PATCH - 6, y0 + rng.randint(12, 20)], fill=rng.randint(170, 220))


def _make_one(prefix: str, idx: int, label: str, out_dir: str, rng: random.Random) -> Dict[str, Any]:
    morph = "single_dot" if label == "single" else rng.choice(["multi_dots", "large_blob", "broad_area"])
    deff, ref = _base_pair(rng)
    _draw(deff, morph, rng)
    dname, rname = f"{prefix}_{label}_{idx:02d}_def.png", f"{prefix}_{label}_{idx:02d}_ref.png"
    deff.save(os.path.join(out_dir, dname))
    ref.save(os.path.join(out_dir, rname))
    return {"def_patch": dname, "ref_patch": rname, "answer": {
        "reasoning": f"def 相對 ref 多出的亮區形態為 {morph};依準則判為 {label}。",
        "brighter_region_found": True, "morphology": morph, "label": label}}


def _gen(prefix: str, n: int, out_dir: str, rng: random.Random) -> List[Dict[str, Any]]:
    # alternate labels so both classes are balanced
    return [_make_one(prefix, i, "single" if i % 2 == 0 else "cluster", out_dir, rng) for i in range(n)]


def build(out_dir: str, n_eval: int, n_exemplar: int, seed: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    exemplars = _gen("ex", n_exemplar, out_dir, rng)
    eval_items = _gen("ev", n_eval, out_dir, rng)
    with open(os.path.join(out_dir, "exemplar_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(exemplars, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "eval_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(eval_items, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(exemplars)} exemplars + {len(eval_items)} eval pairs to {out_dir}/")
    print("NOTE: synthetic data — for pipeline validation only, not for real metrics.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic def/ref single-vs-cluster demo data")
    ap.add_argument("--out-dir", default="./synthetic_demo")
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--n-exemplar", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.out_dir, a.n_eval, a.n_exemplar, a.seed)


if __name__ == "__main__":
    main()
