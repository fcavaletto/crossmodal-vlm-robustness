# Phase E verification — held-out one-shot (frozen Checkpoint 1)

**Date:** 2026-11-05 → 2026-11-06  
**Branch:** `cursor/phase-e-prep-dabf`  
**Authorized tip (command/config):** `34d85d4`  
**Runtime tip (revision-verify hotfix only):** `d1ef71e`  
**PR:** https://github.com/fcavaletto/crossmodal-vlm-robustness/pull/7  
**Scope:** Sealed **test** split, **32** images × color+shape; frozen ε ∈ {2,4,8}/255; PGD 7 / α=ε/4; FGSM; noise; opt@neutral; N/S/M eval; no-split; leading-space tokens.  
**Not done:** retune · merge to `main` · technical-report rewrite · fusion / extra experiments

---

## Verdict (held-out confirmatory)

1. **Primary DiD (misleading − supportive) does not support conflict amplification.** Color estimates are **negative** at all three ε with 95% CIs excluding 0 (replicates development). Shape estimates are negative/near-zero; CI includes 0 at ε=2/255 and ε=8/255.
2. **Secondary DiD (neutral − supportive) remains the informative contrast.** Color: positive at all ε (CIs exclude 0 except the lower edge touches 0 at ε=8/255). Shape: positive point estimates; CIs include 0 at ε=2 and 4/255.
3. **Misleading-color floor persists on test** (clean 4/32 = **12.5%**) — same identification problem as development (6/64 ≈ 9.4%).
4. **PGD** collapses neutral accuracy; **supportive** partially mitigates; **noise** ≈ null; **FGSM** weak vs PGD.
5. **L∞ / [0,1] bounds** hold before and after uint8 round-trip (576/576 attacked rows).

**Scientific status unchanged (D1):** primary = misleading−supportive (preregistered; report even if negative / floor-dominated). Secondary = neutral−supportive (strongest secondary; not a validated defense). No retrospective hypothesis rewrite.

---

## Interruptions (documented; no config change)

| Attempt | Outcome |
| --- | --- |
| **1** | Aborted **before any test inference**. Transformers 5.22.1 did not populate object `_commit_hash`; revision assert raised. Log preserved: `results/phase_e_heldout_attempt1_revision_verify_fail.log`. |
| **Hotfix** | Commit `d1ef71e` — verify frozen revision via `huggingface_hub.model_info(...).sha` while still loading with `revision=7b375e1b…`. **No** change to ε, attacks, prompts, parser, preprocess, metrics, or seeds. |
| **2 (authorized one-shot)** | Started under identical CLI. Agent session / Mac sleep interrupted monitoring mid-run (~9/32 → later resumed monitoring at ~26/32). The **same process (PID 25174) kept running** and completed; **no second launch**, no `--resume` relaunch required. |

Wall clock (attempt 2): **11149.1 s** (~3.1 h) → `status=complete`, **640** JSONL rows, **32** images.

---

## Exact command (as run)

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

Analyze:

```bash
python -m crossmodal_robustness.heldout_analyze \
  --jsonl results/phase_e_heldout.jsonl \
  --development-metrics reports/phase-d-full64-metrics.json \
  --output results/phase_e_heldout_metrics.json \
  --figure-dir reports/figures/phase-e
```

---

## Artifacts

| Path | Role |
| --- | --- |
| `results/phase_e_heldout.jsonl` | Raw predictions (local; gitignored) |
| `results/phase_e_heldout.log` | Run log (local) |
| `results/phase_e_heldout_attempt1_revision_verify_fail.log` | Attempt-1 abort (local) |
| `results/phase_e_heldout_meta.json` → `reports/phase-e-heldout-meta.json` | Frozen pins + wall time |
| `results/phase_e_heldout_metrics.json` → `reports/phase-e-heldout-metrics.json` | Held-out metrics + side-by-side vs Phase D |
| `reports/figures/phase-e/*.png` | Accuracy / additional-error / DiD plots |
| `reports/phase-e-verification.md` | This document |

---

## Clean accuracy (held-out vs development)

| Split | Task | Neutral | Supportive | Misleading |
| --- | --- | --- | --- | --- |
| **Test (32)** | color | 29/32 (90.6%) | 32/32 (100%) | **4/32 (12.5%)** ← floor |
| **Test (32)** | shape | 24/32 (75.0%) | 32/32 (100%) | 8/32 (25.0%) |
| Dev (64) | color | 63/64 (98.4%) | 64/64 (100%) | 6/64 (9.4%) |
| Dev (64) | shape | 48/64 (75.0%) | 64/64 (100%) | 29/64 (45.3%) |

Clean invalid rate (test): **0**.  
**Unexpected / note:** test misleading-shape clean (25%) is lower than development (45.3%); test neutral-color clean (90.6%) is slightly lower than development (98.4%). Floor pattern for misleading-color **replicates**.

---

## Bounds / round-trip (held-out)

Among **576** attacked rows: before-roundtrip OK rate **1.0**; after-roundtrip OK rate **1.0**; max L∞ after ≈ 8/255; mean |float adv − uint8 reconstr| ≈ 8.7e-4.

---

## PGD conditional ASR (held-out; flips among clean-correct)

| ε | Task | Neutral ASR | Supportive ASR | Misleading ASR* |
| --- | --- | --- | --- | --- |
| 2/255 | color | 29/29 = 100% | 16/32 = 50.0% | 4/4 = 100% |
| 2/255 | shape | 22/24 = 91.7% | 14/32 = 43.8% | 8/8 = 100% |
| 4/255 | color | 29/29 = 100% | 16/32 = 50.0% | 4/4 |
| 4/255 | shape | 24/24 = 100% | 17/32 = 53.1% | 8/8 |
| 8/255 | color | 27/29 = 93.1% | 20/32 = 62.5% | 4/4 |
| 8/255 | shape | 24/24 = 100% | 14/32 = 43.8% | 8/8 |

\*Misleading ASR denominators are tiny/floored — not a reliable amplification readout.

### PGD robust accuracy @ ε=8/255 (held-out)

| Task | Neutral | Supportive | Misleading |
| --- | --- | --- | --- |
| color | 2/32 (6.2%), inv 28.1% | 12/32 (37.5%), inv 31.2% | 0/32 (0%), inv 25.0% |
| shape | 0/32 (0%), inv 0% | 18/32 (56.2%), inv 0% | 0/32 (0%), inv 0% |

---

## Primary DiD — misleading − supportive (image bootstrap, n=2000, seed 20261105)

| Attack | ε | Task | Held-out estimate [CI95] | Development estimate [CI95] |
| --- | --- | --- | --- | --- |
| PGD | 2/255 | color | **−0.375** [−0.562, −0.156] | −0.531 [−0.656, −0.391] |
| PGD | 4/255 | color | **−0.375** [−0.594, −0.156] | −0.547 [−0.672, −0.406] |
| PGD | 8/255 | color | **−0.500** [−0.688, −0.344] | −0.516 [−0.656, −0.375] |
| PGD | 2/255 | shape | −0.188 [−0.469, 0.062] | −0.078 [−0.281, 0.125] |
| PGD | 4/255 | shape | −0.281 [−0.531, −0.031] | −0.016 [−0.234, 0.203] |
| PGD | 8/255 | shape | −0.188 [−0.469, 0.062] | −0.016 [−0.219, 0.188] |

**Reading:** Held-out **replicates** the development conclusion that primary DiD is **not positive** for color (floor-dominated). Shape remains near-null / slightly negative; one ε (4/255) has CI excluding 0 on test but development CI included 0 — small-n instability, not a claim flip to amplification.

---

## Secondary DiD — neutral − supportive

| Attack | ε | Task | Held-out estimate [CI95] | Development estimate [CI95] |
| --- | --- | --- | --- | --- |
| PGD | 2/255 | color | **0.406** [0.188, 0.656] | 0.359 [0.250, 0.500] |
| PGD | 4/255 | color | **0.406** [0.188, 0.625] | 0.297 [0.172, 0.422] |
| PGD | 8/255 | color | **0.219** [0.000, 0.438] | 0.375 [0.250, 0.500] |
| PGD | 2/255 | shape | 0.250 [−0.062, 0.531] | 0.219 [0.016, 0.406] |
| PGD | 4/255 | shape | 0.219 [−0.094, 0.500] | 0.281 [0.062, 0.469] |
| PGD | 8/255 | shape | 0.312 [0.000, 0.594] | 0.250 [0.047, 0.453] |

**Reading:** Supportive context still reduces additional PGD error relative to neutral on color (secondary claim direction replicates). Shape secondary CIs are wider on n=32 and often include 0 — weaker held-out confirmation than development.

---

## Controls (held-out, ε=8/255 DiD snapshot)

| Attack | Task | Primary mis−sup | Secondary neu−sup |
| --- | --- | --- | --- |
| noise | color | 0.000 [0, 0] | −0.094 [−0.219, 0] |
| noise | shape | 0.000 [0, 0] | 0.000 [0, 0] |
| FGSM | color | 0.062 [0, 0.156] | 0.031 [0, 0.094] |
| FGSM | shape | 0.125 [0.031, 0.250] | 0.188 [0.062, 0.344] |

Noise remains a near-null control. FGSM remains much weaker than PGD (same as development).

---

## Unexpected / honest caveats

1. **No primary-claim reversal:** held-out does **not** rescue misleading-amplification; color primary DiD stays negative.  
2. **Floor non-replication of magnitude only:** misleading-color clean 12.5% (test) vs 9.4% (dev) — still a floor. Misleading-shape clean drops on test (25% vs 45%).  
3. **Secondary CI edge case:** color neu−sup at ε=8/255 CI lower bound = 0.000 (touches null).  
4. **Shape secondary weaker on test** (several CIs include 0) — do not over-claim supportive “defense.”  
5. **Invalids under PGD (color)** ~19–31% — same strict-parser failure mode as development; many errors are invalids, not clean class swaps.  
6. **Noise seeds** use MD5-stable scheme (pre-test addendum); distribution matches Phase D family but tensors are not bit-identical to development noise.  
7. **Sleep / agent disconnect** delayed monitoring only; completed JSONL has exactly 640 unique keys and meta `status=complete`.

---

## STOP

Phase E held-out evaluation complete under frozen Checkpoint 1. **No retune. No merge.** Next work (technical report / README) is out of scope for this stop.
