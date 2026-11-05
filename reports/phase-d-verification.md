# Phase D verification — development ε-grid evaluation (no freeze / no test)

**Date:** 2026-11-05  
**Branch:** `cursor/phase-d-dev-eval-dabf`  
**Stack (frozen):** SmolVLM2-500M, float32, eager attention, MPS, **`do_image_splitting=False`**, Phase C parser/prompts, leading-space answer tokens (`Ġred`, …)  
**Scope:** All **64** development images × color+shape; ε ∈ {2,4,8}/255; PGD 7 steps α=ε/4; FGSM; random noise; optimize **neutral only**; evaluate same pixels under N/S/M.  
**Not done:** held-out test · protocol freeze · Phase E report package

---

## Verdict (preliminary, development only)

1. **PGD works** at all three budgets: near-total collapse of neutral accuracy on baseline-correct color/shape items; supportive text **partially mitigates** attack-induced errors.  
2. **Conflict-amplification DiD (misleading − supportive) is negative for color** and near-zero for shape — **not** evidence that misleading amplifies attacks. This is dominated by the **misleading-color floor** (clean accuracy 6/64 ≈ 9.4%).  
3. The informative contrast is **neutral vs supportive**: positive DiD (more additional error under neutral than supportive), with 95% image-level bootstrap CIs excluding 0 for color at all ε.  
4. Noise ≈ ineffective; FGSM weak vs PGD at matching ε.  
5. L∞ / [0,1] bounds hold **before and after** uint8 round-trip used for generation eval.

**STOP for Federico’s review before freezing hyperparameters or touching the test split.**

---

## Phase B tokenization lesson (explicit)

Phase B reported teacher-forced NLL ≈ 12.55 for label `red` because scoring used the **bare** BPE id (`red`, 455). Greedy decoding emits the **leading-space** piece (`Ġred`, 2382) after `Assistant:`. Phase C fixed `answer_token_ids(..., leading_space=True)`; Phase D attacks/margins use that fix. Phase B’s nonzero gradient was real but **misaligned to the wrong token**. Do not treat Phase B NLL magnitude as comparable to Phase C/D.

---

## Experimental protocol (as run)

| Setting | Value |
| --- | --- |
| Images | Balanced 16-image smoke, then all 64 development (resume) |
| Tasks | color, shape |
| Optimize | neutral prompt only |
| Evaluate | neutral / supportive / misleading (same perturbation) |
| ε | 2/255, 4/255, 8/255 |
| PGD | 7 iterations, step α=ε/4, no random start |
| Controls | FGSM (1-step), uniform L∞ noise |
| Preprocess | no-split; attack on torch CHW [0,1]; eval via uint8 PIL |
| Wall | ~1.4 h balanced-16 + ~4.4 h remaining-48 (resume); MPS ~4 GB |

Artifacts: `results/phase_d_full64.jsonl` (local), `reports/phase-d-full64-metrics.json`, figures under `reports/figures/phase-d/`.

---

## Clean (no-attack) development accuracy

Matches Phase C no-split baseline on this stack:

| Task | Neutral | Supportive | Misleading |
| --- | --- | --- | --- |
| color | 63/64 (98.4%) | 64/64 (100%) | **6/64 (9.4%)** ← floor |
| shape | 48/64 (75.0%) | 64/64 (100%) | 29/64 (45.3%) |

Invalid rate clean: **0**.

---

## Bounds / round-trip

Among 1152 attacked rows: **100%** within ε and in [0,1] **before** round-trip; **100%** after round-trip with ≤1 grey-level tolerance (`ε + 1.5/255`). Mean |float adv − uint8 reconstr| ≈ 8.7e-4.

---

## Attack effectiveness (PGD highlight)

### Conditional ASR (flips among clean-correct)

| ε | Task | Neutral ASR | Supportive ASR | Misleading ASR* |
| --- | --- | --- | --- | --- |
| 2/255 | color | **63/63 = 100%** | 40/64 = 62.5% | 6/6 = 100% (n=6) |
| 2/255 | shape | **48/48 = 100%** | 34/64 = 53.1% | 29/29 = 100% |
| 4/255 | color | 60/63 = 95.2% | 41/64 = 64.1% | 6/6 |
| 4/255 | shape | 48/48 = 100% | 30/64 = 46.9% | 29/29 |
| 8/255 | color | **63/63 = 100%** | 39/64 = 60.9% | 6/6 |
| 8/255 | shape | 46/48 = 95.8% | 30/64 = 46.9% | 29/29 |

\*Misleading ASR denominators are tiny/floored for color — not a reliable amplification readout.

**Controls at ε=8/255, color neutral:** noise ASR **0/63**; FGSM ASR **4/63 ≈ 6.3%**.

### Mean additional error (error_adv − error_clean), PGD

| ε | Task | Neutral Δ | Supportive Δ | Misleading Δ |
| --- | --- | --- | --- | --- |
| 2/255 | color | **0.984** | 0.625 | **0.094** (floor) |
| 8/255 | color | **0.984** | 0.609 | **0.094** (floor) |
| 2/255 | shape | 0.750 | 0.531 | 0.453 |
| 8/255 | shape | 0.719 | 0.469 | 0.453 |

PGD raises correct-answer NLL on **100%** of image–task pairs; margins shrink (mean Δmargin ≈ −3 to −5 nats).

PGD also increases **invalid** (unparseable) color responses (~16–42% depending on ε/condition) — counted as errors under the frozen strict parser.

---

## Conflict-amplification DiD (image-level bootstrap, 2000 resamples)

Definition per image:  
`DiD = (e_adv − e_clean)_primary − (e_adv − e_clean)_control`.

### Misleading vs supportive (protocol primary)

| ε | Task | Estimate | 95% CI | Reading |
| --- | --- | --- | --- | --- |
| 2/255 | color | **−0.531** | [−0.656, −0.391] | Negative; floor-dominated |
| 4/255 | color | **−0.547** | [−0.672, −0.406] | Negative |
| 8/255 | color | **−0.516** | [−0.656, −0.375] | Negative |
| 2–8/255 | shape | ≈ −0.02 to −0.08 | CIs cover 0 | No clear amplification |

**Do not interpret low misleading robust accuracy as amplification.** Clean misleading-color error is already ~91%; attack can add at most ~0.09 on average, while supportive starts at 0% clean error and absorbs large additional PGD error — mechanically driving DiD negative.

### Neutral vs supportive (secondary; better identified)

| ε | Task | Estimate | 95% CI |
| --- | --- | --- | --- |
| 2/255 | color | **+0.359** | [0.250, 0.500] |
| 8/255 | color | **+0.375** | [0.250, 0.500] |
| 2/255 | shape | **+0.219** | [0.016, 0.406] |
| 8/255 | shape | **+0.250** | [0.047, 0.453] |

Supportive context **reduces** attack-caused additional error relative to neutral on this synthetic benchmark — a cross-modal interaction, but **not** the preregistered “misleading amplifies PGD” claim for color.

---

## Figures (absolute paths)

- `/Users/fedecava/Development/crossmodal-vln-robustness/reports/figures/phase-d/accuracy_vs_eps_color.png`
- `/Users/fedecava/Development/crossmodal-vln-robustness/reports/figures/phase-d/accuracy_vs_eps_shape.png`
- `/Users/fedecava/Development/crossmodal-vln-robustness/reports/figures/phase-d/pgd_additional_error.png`
- `/Users/fedecava/Development/crossmodal-vln-robustness/reports/figures/phase-d/pgd_did_bootstrap.png`
- `/Users/fedecava/Development/crossmodal-vln-robustness/reports/figures/phase-d/examples/item_000_color_eps8.png`
- `/Users/fedecava/Development/crossmodal-vln-robustness/reports/figures/phase-d/examples/item_030_color_eps8.png`

Store mirrors: `/Users/fedecava/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-ecd85e51-24e7-4901-84e2-c396123aadd5/files/media/phase-d/`

---

## Limitations

- Synthetic domain; small n=64 images; color/shape share images → non-independent tasks.  
- Misleading-color floor blocks the primary DiD.  
- Optimize-under-neutral only (as approved).  
- Strict parser treats many PGD invalids as failures.  
- No held-out evaluation; results are **development / preliminary**.  
- ε grid shows PGD already strong at 2/255 — limited graded dose–response on neutral ASR (ceiling).

---

## Recommendation before freeze / Phase E

**Ask Federico to decide:**

1. **Primary scientific claim:** keep misleading–supportive DiD (with hard floor caveat) vs elevate neutral–supportive mitigation as the main reported interaction.  
2. **Whether to freeze** ε∈{2,4,8}/255, PGD 7 / α=ε/4, opt@neutral, no-split, leading-space scoring — or adjust (e.g., milder attacks, different objective) given ceiling at ε=2/255.  
3. **Only after freeze:** Phase E held-out test + technical report. **Do not run test yet.**
