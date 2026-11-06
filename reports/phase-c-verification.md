# Phase C verification — no-split re-baseline, loss fix, attack pilot

**Date:** 2026-11-05  
**Branch:** `cursor/phase-c-nosplit-attacks-dabf`  
**Policy:** Federico option 1 — `do_image_splitting=False` for adversarial optimization **and** evaluation  
**Model:** frozen `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, float32, eager attention, MPS  
**Not done:** held-out test · parser/prompt changes · Phase D metrics/plots

---

## Verdict

Phase C **completed with a successful pilot**.

1. Full no-split development baseline (384) did **not** substantially degrade clean accuracy vs Phase A.  
2. Loss validation initially **failed** due to bare-word tokenization; after scoring leading-space BPE pieces (`Ġred`), validation **passed**.  
3. FGSM / PGD / random-noise implemented; 8-image pilot shows **PGD** flips answers under ε=8/255 while **noise/FGSM** do not (at this budget/steps). All perturbations respect L∞ and [0,1] bounds.

---

## 1. No-split development baseline (preserve Phase A)

Phase A artifacts untouched. New files:

| Artifact | Path |
| --- | --- |
| JSONL | `results/development_baseline_nosplit.jsonl` (gitignored) |
| Summary | `reports/phase-c-nosplit-baseline-summary.json` |

| Task × condition | No-split correct | Phase A (split) |
| --- | --- | --- |
| color neutral | **63/64 (98.4%)** | 64/64 |
| color supportive | **64/64 (100%)** | 64/64 |
| color misleading | 6/64 (9.4%) | 3/64 |
| shape neutral | 48/64 (75.0%) | 52/64 |
| shape supportive | **64/64 (100%)** | 56/64 |
| shape misleading | 29/64 (45.3%) | 33/64 |
| **Overall** | **274/384 (71.4%)**, invalid **0** | 272/384 (70.8%), invalid 48 |

Mean latency **0.31 s** (vs ~4.8 s with splitting). Gate: no substantial degradation → **continue**.

---

## 2. Loss validation & tokenization fix

Probe: `item_000`, color, neutral, no-split.

| Check | Before fix (bare `red`) | After fix (`Ġred`) |
| --- | --- | --- |
| Greedy generation | `red` | `red` |
| Correct NLL (PIL) | **12.62** | **0.266** |
| NLL ranking | yellow < blue < red | **red ≪ blue < yellow < green** |
| Margin (best wrong − correct) | −0.60 (fail) | **+1.90** (pass) |
| PIL vs torch generation | — | **agree** |

**Root cause:** After `Assistant:`, SmolVLM emits leading-space tokens (`Ġred` id 2382), not bare `red` (id 455). Phase B’s NLL≈12.55 was the near-zero-probability bare token. `answer_token_ids(..., leading_space=True)` is now the default for gates/attacks. Parser/prompts unchanged.

Artifact: `reports/phase-c-loss-validation.json`.

---

## 3. Preprocessing consistency (PIL vs torch)

| Metric | Value |
| --- | --- |
| Max \|PIL − torch\| pixel_values | ~0.266 (BICUBIC vs LANCZOS) |
| Mean abs diff | ~9e-4 |
| Generation with swapped `pixel_values` | **identical** on probe |
| Attack path | torch CHW `[0,1]`, `do_rescale=False`, `do_image_splitting=False` |
| Eval path for adv images | tensor → uint8 PIL → same no-split `apply_chat_template` |

Residual risk: 8-bit round-trip for generation eval; L∞ measured in float `[0,1]` before quantization. Did not block the pilot.

---

## 4. Attack pilot (8 development images × color+shape)

Settings: ε=**8/255**, FGSM 1-step, PGD **7** steps step **2/255**, noise U[-ε,ε], optimize under **neutral**, reuse pixels across N/S/M. Wall ~**763 s**. MPS driver ~**4.0–4.1 GB**.

### Epsilon / bounds

**100%** of clean/noise/FGSM/PGD outputs satisfy L∞≤ε and pixels∈[0,1] (unit-tested primitives + pilot checks).

### Source NLL (ascent objective)

| Attack | Fraction NLL increased |
| --- | --- |
| noise | 25% |
| FGSM | 93.8% |
| PGD | **100%** |

### Accuracy (16 task-instances per condition cell = 8 images × {color,shape} pooled in summary JSON; task breakdown below)

**Color neutral (clean-correct 8/8):** noise ASR 0% · FGSM ASR 0% · **PGD ASR 8/8 = 100%**  
**Color supportive:** PGD ASR **4/8 = 50%**  
**Shape neutral (clean-correct 4/8):** PGD ASR **4/4 = 100%** on those correct; noise/FGSM 0%  
**Shape supportive:** PGD ASR **5/8 = 62.5%**

Pooled accuracy after PGD: neutral **0/16**, supportive **7/16**, misleading **0/16** (clean was 12/16, 16/16, 2/16).

Noise matched clean accuracy everywhere in this pilot (equal-budget control ineffective at ε=8/255). FGSM raised NLL but did not flip greedy answers.

Artifacts: `reports/phase-c-attack-pilot.json`, `reports/phase-c-attack-pilot-summary.json`.

---

## Files changed (high level)

- `evaluate.py` — `--do-image-splitting` / `--no-do-image-splitting`
- `gradient_gate.py` — leading-space answer tokens
- `loss_validation.py`, `attack_pilot.py`
- `attacks/{fgsm,noise,vlm_loss}.py` + unit tests
- Reports under `reports/phase-c-*` (Phase A files not overwritten)

Tests: **12 passed**.

---

## Limitations

- Pilot n=8 images only; no CIs / DiD interaction stats (Phase D).  
- Single ε and PGD step count.  
- Optimize-under-neutral only.  
- Supportive text still recovers some PGD-broken color cases (50% remain correct) — interaction hypothesis is plausible but untested at scale.  
- Phase B report’s NLL≈12.55 is superseded by the tokenization fix (gradients were real but on the wrong token).

---

## Recommendation for Phase D

**Proceed** with Phase D on the no-split stack: scale attacks to the full development set (or a larger stratified subset), compute paired DiD (misleading vs supportive) with image-level bootstrap CIs, add plots/diff maps, freeze hyperparameters, then (only after freeze) consider held-out test.

**Ask Federico before Phase D** if he wants: (a) additional ε∈{2,4,8}/255 or more PGD steps in the scale-up, (b) optimize-under-supportive instead of/in addition to neutral, (c) whether to note Phase B NLL erratum in the final report.
