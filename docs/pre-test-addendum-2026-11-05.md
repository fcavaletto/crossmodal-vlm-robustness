# Pre-test technical addendum — 2026-11-05

**Status:** Phase E **preparation** only. Held-out inference **not** executed.  
**Scope:** engineering safeguards required before one-shot authorization.  
**Does not change:** hypothesis hierarchy (D1), ε grid / PGD / FGSM / noise threat model (D2), prompts, parser, preprocess, optimize condition, bootstrap settings.

---

## 1. Process-stable noise seeds (departure vs Phase D noise draws)

**Issue.** Phase D seeded equal-budget L∞ noise with Python’s built-in `hash()`:

```text
seed = base + index * 131 + hash((task, attack, f"{ε:.8f}")) % 10007
```

`hash()` is process-randomized unless `PYTHONHASHSEED` is fixed, so identical CLI flags can yield different noise across interpreter invocations.

**Correction.** Noise seeds now use MD5-derived `attack_noise_seed(...)` in
`crossmodal_robustness.seeds` (tag `linf_noise_v1`). Distribution unchanged:
uniform L∞ noise in `[−ε, ε]` via `random_linf_noise`. Only the **concrete seed
integers** differ from Phase D.

**Implication.** Held-out (and any re-run of development noise rows) will **not**
bit-match Phase D noise images. Gradient attacks (FGSM/PGD) are unaffected
(deterministic under frozen `random_start=false`). Report noise results as the
same control family, not as a pixel-identical replay of Phase D noise.

**Not changed:** ε ∈ {2,4,8}/255, steps=7, α=ε/4, opt@neutral, no-split.

---

## 2. Frozen HF model **and** processor revision enforcement

Held-out entrypoint loads both processor and model with
`revision=7b375e1b73b11138ff12fe22c8f2822d8fe03467` and verifies resolved
`_commit_hash` against `docs/freeze-config.md` / `PROTOCOL.md`. Mismatch aborts.

---

## 3. Floor diagnostics no longer hardcode development 6/64

`phase_d_analyze.build_metrics` computes misleading-color clean accuracy from
**the JSONL being analyzed**. Held-out analysis must never import the
development 6/64 ≈ 9.4% constant as a test statistic. Development vs held-out
floors are reported in separate blocks (`heldout_analyze.compare_splits`).

---

## 4. Dedicated held-out entry + unique outputs

- Module: `python -m crossmodal_robustness.heldout_eval`
- Split: explicit `split="test"`; validates **32** images
- Development path (`phase_d_eval`) left intact
- Default output: `results/phase_e_heldout.json(.jsonl)` — refuses `phase_d_*` stems
- Inference requires `--i-authorize-heldout-oneshot`; `--validate-only` is the safe prep check

---

## 5. Exact held-out execution command (authorized one-shot only)

```bash
python -m crossmodal_robustness.heldout_eval \
  --manifest data/synthetic_v1/manifest.jsonl \
  --model HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --device mps \
  --task both \
  --epsilons 0.00784313725490196 0.01568627450980392 0.03137254901960784 \
  --steps 7 \
  --attacks noise fgsm pgd \
  --seed 20261105 \
  --output results/phase_e_heldout.json \
  --i-authorize-heldout-oneshot
```

Post-run analysis (after inference completes):

```bash
python -m crossmodal_robustness.heldout_analyze \
  --jsonl results/phase_e_heldout.jsonl \
  --development-metrics reports/phase-d-full64-metrics.json \
  --output results/phase_e_heldout_metrics.json \
  --figure-dir reports/figures/phase-e
```

Prep validation (no inference):

```bash
python -m crossmodal_robustness.heldout_eval --validate-only
```
