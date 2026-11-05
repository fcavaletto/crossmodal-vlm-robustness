# Phase A verification — clean SmolVLM2 baselines on MPS

**Date:** 2026-11-05  
**Branch:** `cursor/phase-a-baseline-dabf`  
**Checkout:** `/Users/fedecava/Development/crossmodal-vln-robustness` (cloned into empty worker-labeled dir)  
**Model:** `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` @ `7b375e1b73b11138ff12fe22c8f2822d8fe03467`  
**Device:** Apple M4 MPS, 24 GB unified memory · dtype float32 · `attn_implementation=eager` · `do_sample=False`

---

## Verdict

Phase A **passed**. Local inference works on MPS; vision inputs change predictions; development-set clean baselines under Neutral / Supportive / Misleading are measured with strict scoring. **Held-out test was not evaluated. No attacks run.**

---

## Files changed

| Path | Change |
| --- | --- |
| `pyproject.toml` | Added `torchvision>=0.19`, `num2words>=0.5.14` (required by SmolVLM2 processor) |
| `src/crossmodal_robustness/evaluate.py` | Prefer `dtype=torch.float32` with `torch_dtype` fallback |
| `reports/phase-a-verification.md` | This report (repo mirror) |
| `reports/phase-a-baseline-summary.json` | Aggregated development metrics |

Untracked locally (gitignored `results/`): full JSONL + smoke/vision logs used as evidence.

---

## Commands executed

```bash
git clone https://github.com/fcavaletto/crossmodal-vlm-robustness.git .
git checkout -b cursor/phase-a-baseline-dabf
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,plot]'
python -m pip install torchvision num2words   # then pinned in pyproject
python -m pytest -q
python -m crossmodal_robustness.evaluate --dry-run --task both --max-examples 2
# dataset integrity script (96 images, hashes, splits, balance)
# MPS matmul sanity check
# smoke: 3 contrasting images × color+shape (neutral)
# vision check: same prompt, red-circle vs blue-square vs blank
python -m crossmodal_robustness.evaluate \
  --device mps --task both --max-examples 0 --split development \
  --output results/development_baseline.jsonl
python -m pytest -q  # recheck after code edits
```

---

## Tests

| Suite | Result |
| --- | --- |
| `python -m pytest -q` (7 tests) | **7 passed** (before and after compat edits) |
| Dry-run prompts | OK — paired contexts, fixed question wording |
| Dataset integrity | **96** PNGs; **64** development / **32** test; 4×4×6 balance; SHA-256 match; wrong≠true labels |
| Held-out test inference | **Not run** (by design) |

---

## Model-loading verification

| Check | Result |
| --- | --- |
| `torch.backends.mps.is_available()` | True |
| Processor | `SmolVLMProcessor` loads after torchvision + num2words |
| Model class | `SmolVLMForConditionalGeneration` via `AutoModelForImageTextToText` |
| Load (cold download) | ~239 s first time; ~4 s warm from cache |
| Packages | Python 3.12.14 · torch 2.15.0 · torchvision 0.29.2 · transformers 5.22.1 · numpy 2.5.4 · Pillow 13.0.0 · accelerate 1.15.0 |

Missing deps were the only Phase A code/env fixes required; chat-template inference path worked as written.

---

## Evidence vision affects predictions

Same neutral color prompt, different images:

| Image | Raw answer |
| --- | --- |
| `item_000` red circle | `red` |
| `item_030` blue square | `blue` |
| `item_090` yellow star | `yellow.` |
| Blank 256×256 gray | `blue` (default-ish; not image-copy of red) |

Shape answers also differed (`circle…` echo vs `square`). Color accuracy on clean neutral/supportive development items is **64/64**, which is incompatible with text-only ignoring of the image.

---

## Memory and latency (approx.)

| Metric | Value |
| --- | --- |
| Process RSS after load (smoke) | ~401 MB (CPU RSS; undercounts unified GPU) |
| MPS `driver_allocated_memory` during infer | ~5.5 GB |
| MPS `current_allocated_memory` | ~2.0 GB |
| Smoke per-call latency | ~2.9–4.4 s |
| Full development mean / median / max | **4.84 s / 4.62 s / 9.17 s** (384 calls) |
| Full development wall time | ~31 minutes |

Feasible on 24 GB M4; float32 + eager attention leaves headroom for Phase B gradient experiments if careful.

---

## Development-set baseline (strict exact match)

**384** evaluations = 64 images × 2 tasks × 3 conditions. Invalid = parser returned `None` (counts as incorrect).

### By task × condition

| Task | Condition | Correct | Accuracy | Invalid | Invalid rate |
| --- | --- | --- | --- | --- | --- |
| color | neutral | 64/64 | **100.0%** | 0 | 0.0% |
| color | supportive | 64/64 | **100.0%** | 0 | 0.0% |
| color | misleading | 3/64 | **4.7%** | 13 | 20.3% |
| shape | neutral | 52/64 | **81.2%** | 12 | 18.8% |
| shape | supportive | 56/64 | **87.5%** | 8 | 12.5% |
| shape | misleading | 33/64 | **51.6%** | 15 | 23.4% |

### By condition (tasks pooled)

| Condition | Correct | Accuracy | Invalid |
| --- | --- | --- | --- |
| neutral | 116/128 | 90.6% | 12 (9.4%) |
| supportive | 120/128 | 93.8% | 8 (6.2%) |
| misleading | 36/128 | 28.1% | 28 (21.9%) |
| **overall** | **272/384** | **70.8%** | **48 (12.5%)** |

### Qualitative notes

- **Color is visually solved** under neutral/supportive; **misleading color context collapses accuracy** (48/64 parsed answers equal the false claimed color).
- Shape is harder: frequent invalid echo of the option list (`circle, square, triangle, or star.`).
- Supportive ≥ neutral for shape; misleading hurts both tasks but color far more.

Artifacts: `results/development_baseline.jsonl` (local), `reports/phase-a-baseline-summary.json` (committed).

---

## Remaining technical / scientific risks

1. **Strict parser vs verbose echoes** inflates “error” that is format failure, not wrong class — report invalid rate always; optional Phase B/C soft-score would be a protocol change.
2. **Text dominance on color** under misleading context may leave little room for *additional* PGD amplification (ceiling/floor effects on interaction DiD).
3. **MPS autograd** through SmolVLM2 still unproven (Phase B gate).
4. Synthetic domain; shared images across tasks → non-independent observations.
5. Unauthenticated HF Hub downloads; pin revision in scripts for full reproducibility.
6. `dtype` / Transformers 5.x API churn — keep the fallback path.

---

## Recommendation for Phase B (do not implement yet)

**Proceed to Phase B** with a single-image float32 MPS gradient feasibility gate:

1. Differentiable pixel→processor path (no PIL inside autograd).
2. NLL (or justified target) for the correct short answer token(s).
3. Confirm nonzero finite `∂L/∂image`; if MPS backward fails, document and propose CLIP-source transfer (different threat model).
4. Keep development-only; freeze prompts/parser as in Phase A unless Federico approves a format-handling change.

**Ask Federico before Phase B** only if wanting to (a) loosen the strict parser / prompt to cut shape invalids, or (b) change primary model — otherwise Phase B can start as scoped in `CURSOR_HANDOFF.md`.
