# Phase B verification — image→loss gradient feasibility (MPS)

**Date:** 2026-11-05  
**Branch:** `cursor/phase-b-gradient-gate-dabf`  
**Model:** `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (frozen, float32, eager attention)  
**Device:** Apple M4 MPS, 24 GB  
**Scope:** Feasibility gate only — **no PGD/FGSM/noise attacks**. Phase A parser, prompts, and baseline numbers unchanged (aside from a reporting typo fix).

---

## Verdict

**PASS.** A differentiable CHW `[0,1]` image tensor receives a **finite, nonzero** gradient from teacher-forced correct-answer NLL through frozen SmolVLM2 on MPS. Preprocessing uses the official SmolVLM image processor on torch tensors (**no PIL on the autograd path**). Peak observed MPS driver allocation ≈ **4.0–4.1 GB** — practical on 24 GB.

---

## Minor non-experimental fix

`reports/phase-a-baseline-summary.json`: `overall.invalid` corrected **15 → 48** to match the sum of per-condition invalid counts (28+12+8). **No baseline re-run**; experimental JSONL untouched.

---

## Implementation summary

New module: `src/crossmodal_robustness/gradient_gate.py`

1. Load RGB as float32 CHW in `[0,1]` (PIL for disk I/O only).
2. Freeze all model parameters.
3. Build prompt tokens via `apply_chat_template` with `processor_kwargs={"do_image_splitting": False}` (structure only).
4. Autograd path: `processor.image_processor(images=chw_tensor, do_rescale=False, do_image_splitting=False)` → `pixel_values`.
5. Teacher-forced NLL on the answer token sequence (`labels=-100` on the prompt prefix).
6. `loss.backward()`; inspect `image.grad`.
7. Dependence check: same prompt/answer text, two different images → losses and grad magnitudes differ.

Tests: `tests/test_gradient_gate.py` (tokenizer labels + preprocess keeps grad). Suite: **9 passed**.

### Why `do_image_splitting=False`

Default SmolVLM splitting yields **17×512×512** patches. A first MPS attempt with splitting **SIGKILL / OOM** on 24 GB during forward/backward. Disabling splitting yields **1×3×512×512** and succeeds.

Phase A *generation* baselines used the processor default (**splitting on**). Phase B therefore proves gradients through the vision stack under a **memory-feasible preprocess**. Phase C must choose whether to (a) attack under no-split and evaluate consistently, (b) seek a memory-efficient split path, or (c) accept a documented train/eval preprocess mismatch — **needs Federico’s call**.

### Preprocess fidelity

Torch path with `[0,1]` + `do_rescale=False` vs PIL reference (no-split): mean abs diff ≈ **9e-4**, max ≈ **0.27** (LANCZOS→BICUBIC fallback). Documented; gradients still flow.

---

## Gradient diagnostics

### Color / neutral / `item_000` → answer `red`

| Metric | Value |
| --- | --- |
| Loss (NLL) | 12.552 |
| Grad finite | yes |
| Grad nonzero | yes |
| Grad \|mean\| / \|max\| | 5.85e-3 / 0.826 |
| Nonzero fraction | > 0 (see JSON) |
| Forward / backward | 1.17 s / 5.04 s |
| MPS driver alloc | **4022 MB** |
| Prompt tokens (no-split) | 112 |
| Answer tokens | `['red']` |

### Dependence (`item_000` vs `item_030`, same prompt/answer `"red"`)

| | A (red circle) | B (blue square) |
| --- | --- | --- |
| Loss | 12.552 | 14.155 |
| Grad \|mean\| | 5.85e-3 | 7.75e-3 |

Losses and grad means **differ** → gradient depends on the image.

### Shape checks

| Case | Loss | Grad \|mean\| / \|max\| | Tokens | Mem |
| --- | --- | --- | --- | --- |
| `item_000` → `circle` | 9.790 | 4.01e-3 / 0.338 | `['circle']` | ~4052 MB |
| triangle image → `triangle` | 5.337 | 2.85e-3 / 0.321 | `['tri','angle']` | ~4052 MB |

Multi-token answers work.

---

## Commands

```bash
python -m pytest -q
python -m crossmodal_robustness.gradient_gate \
  --device mps --no-do-image-splitting \
  --output results/phase_b_gradient_gate.json
python -m crossmodal_robustness.gradient_gate \
  --device mps --no-do-image-splitting --task shape \
  --output results/phase_b_gradient_gate_shape.json
```

Artifacts (gitignored `results/` + committed `reports/` mirrors):  
`phase_b_gradient_gate.json`, `phase_b_gradient_gate_shape.json`, triangle diagnostics.

---

## Remaining risks for Phase C

1. **Image-splitting mismatch** vs Phase A eval preprocess (see above) — primary design decision.
2. Resize LANCZOS vs BICUBIC → epsilon measured in raw `[0,1]` remains well-defined; reported pixel maps should note interpolate mode.
3. `triangle` is two tokens — score full sequences (already done in the gate).
4. Color misleading floor from Phase A may limit DiD dynamic range once attacks exist.
5. Full PGD still unproven (step loop, clipping, paired reuse) — only the loss/grad adapter is gated.

---

## Recommendation for Phase C

See `reports/phase-b-verification.md`.
