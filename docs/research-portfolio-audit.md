# Research portfolio audit — crossmodal-vlm-robustness

**Audit date:** 2026-11-05  
**Canonical tip:** `cursor/phase-d-dev-eval-dabf` @ `17d64b2` ([PR #4](https://github.com/fcavaletto/crossmodal-vlm-robustness/pull/4))  
**`main`:** still starter scaffold @ `c60c8a4` — **do not restart from main**  
**Checkout:** `/Users/fedecava/Development/crossmodal-vln-robustness`  
**Hardware confirmed:** MacBook Air M4, 24 GB, PyTorch MPS  

---

## 1. Git / PR landscape

| Branch | Tip | PR | Status |
| --- | --- | --- | --- |
| `main` | starter only | — | outdated vs experiments |
| `cursor/phase-a-baseline-dabf` | Phase A docs + deps | [#1](https://github.com/fcavaletto/crossmodal-vlm-robustness/pull/1) | OPEN (superseded by later tips) |
| `cursor/phase-b-gradient-gate-dabf` | gradient gate | [#2](https://github.com/fcavaletto/crossmodal-vlm-robustness/pull/2) | OPEN |
| `cursor/phase-c-nosplit-attacks-dabf` | no-split + pilot | [#3](https://github.com/fcavaletto/crossmodal-vlm-robustness/pull/3) | OPEN |
| **`cursor/phase-d-dev-eval-dabf`** | **full ε-grid + DiD** | **[#4](https://github.com/fcavaletto/crossmodal-vlm-robustness/pull/4)** | **OPEN — current validated tip** |

Lineage is linear (A→B→C→D). Four open stacked PRs; nothing merged. Working tree clean on Phase D tip.

---

## 2. Implementation status (code)

### Confirmed working

| Component | Location | Evidence |
| --- | --- | --- |
| Synthetic dataset + hashes | `data/synthetic_v1/`, `generate_dataset.py` | 96 PNGs; tests green |
| Prompts + strict parser | `dataset.py` | unchanged since Phase A (by design) |
| MPS SmolVLM2 load/generate | `evaluate.py` | Phase A/C/D runs |
| `do_image_splitting` flag | `evaluate.py` | Phase C no-split baselines |
| Leading-space answer tokens | `gradient_gate.answer_token_ids` | Phase C loss validation |
| Diff. preprocess + NLL loss | `attacks/vlm_loss.py`, `gradient_gate.py` | Phase B/C/D |
| FGSM / PGD / noise | `attacks/{fgsm,pgd,noise}.py` | unit tests + Phase D |
| ε-grid eval runner | `phase_d_eval.py` | 1280-row JSONL |
| Metrics + DiD bootstrap + plots | `phase_d_analyze.py` | metrics JSON + figures |
| Example diff maps | `phase_d_examples.py` | `reports/figures/phase-d/examples/` |
| Unit tests | `tests/` | **14 passed** (2026-11-05 audit) |

### Experimental / provisional (not frozen)

- Hyperparameters: ε∈{2,4,8}/255, PGD 7, α=ε/4, opt@neutral, no-split  
- Primary scientific claim framing (misleading−supportive vs neutral−supportive)  
- Phase D development metrics (await Federico freeze decision)

### Not implemented / not started

- Held-out **test** evaluation (32 images)  
- Protocol freeze document  
- `reports/technical_report.md`  
- README rewrite to match reality  
- Reproducibility pin script (exact HF revision in a one-command runner)  
- Qwen3-VL cross-model (optional; out of critical path)  
- Supportive-optimized attacks (explicitly deferred)

---

## 3. Results collected (measured, development only)

### Phase A — default **image-splitting ON**

- Dev baseline 384 evals; overall 272/384 = 70.8%; invalid 12.5%  
- Color N/S 100%; color misleading 4.7%  
- Vision sensitivity confirmed  
- **Note:** later stack uses **no-split**; Phase A is historical / not the attack stack

### Phase B — gradient gate

- Finite nonzero `∂NLL/∂image` on MPS (~4 GB)  
- **Erratum:** NLL≈12.55 used bare BPE (`red`); wrong vs greedy `Ġred` (fixed Phase C)

### Phase C — no-split re-baseline + pilot

- No-split clean 274/384 = 71.4%; **invalid 0%**  
- Loss validation PASS after leading-space fix (NLL≈0.27 for `Ġred`)  
- 8-image pilot: PGD strong; FGSM/noise weak

### Phase D — full development ε-grid (primary evidence)

- 64 images × 2 tasks × {clean + 3ε × 3 attacks} = **1280 rows**  
- PGD color-neutral ASR ≈ **100%** even at ε=2/255; supportive ASR ≈ 47–64%  
- DiD misleading−supportive (color, ε=8/255): **−0.52** [−0.66, −0.38] — **floor**, not amplification  
- DiD neutral−supportive (color, ε=8/255): **+0.38** [0.25, 0.50] — supportive mitigates  
- Bounds 100% OK before/after uint8 round-trip  
- Figures under `reports/figures/phase-d/`

Local raw JSONL: `results/phase_d_full64.jsonl` (gitignored). Committed summaries: `reports/phase-d-full64-metrics.json`.

---

## 4. Held-out test status

| Check | Result |
| --- | --- |
| Protocol intent | Test sealed until freeze |
| Phase D / C / A reports | Explicitly “not done” |
| Baseline JSONL splits | development only (384+384) |
| Phase D image IDs ∩ test IDs | **∅** (0 overlap; 64 ⊆ development) |
| Any `evaluate --split test` artifacts found | **None** |

**Held-out has NOT been run.** Do not run until Federico freezes the study.

---

## 5. Methodological risks (active)

1. **Misleading-color floor (9.4% clean)** — primary preregistered DiD is poorly identified; negative DiD is mechanical.  
2. **Neutral ASR ceiling at ε=2/255** — weak dose–response on that margin; ε grid mostly informative for supportive residual accuracy.  
3. **Stack drift vs Phase A** — attacks use no-split; README still describes split-default unverified path.  
4. **Shared images across color/shape** — non-independent tasks; image-level bootstrap helps DiD but dual-task pooling can still mislead.  
5. **Strict parser + PGD invalids** — many PGD failures are format/invalid, not clean wrong-class flips; report both.  
6. **PIL↔torch resize mismatch** (~0.27 max) — generations agreed on probe; residual attack/eval mismatch risk remains.  
7. **Opt@neutral only** — supportive-optimized threat model untested.  
8. **Synthetic domain** — fine for methodology demo; weak for external validity claims.  
9. **Four unmerged PRs + stale README/HANDOFF** — portfolio readers landing on `main` see “no model run yet.”  
10. **Reproducibility gap** — HF revision not pinned in a single run script; large JSONL only local.

---

## 6. Outdated / misleading docs & files

| Path | Issue |
| --- | --- |
| `README.md` | Still “no VLM run”; “attack adapter not implemented”; omit no-split / Phase D |
| `CURSOR_HANDOFF.md` | Written as pre-A instructions; success criteria partially obsolete |
| `docs/PROTOCOL.md` | Still draft; not updated with freeze candidates or floor caveat |
| `docs/phase-{b,c,d}-*.md` | Thin pointers — OK, but README doesn’t link the real reports |
| Phase A reports | Valid historically; easy to confuse with no-split attack stack |
| `main` | Far behind tip — risky if used as “source of truth” |

Keep Phase A–D verification reports as immutable experiment logs; rewrite top-level README/PROTOCOL for the portfolio narrative.

---

## 7. Bottom line

The **experimental engine is built and development evidence exists**. The portfolio is **not submission-ready**: docs are stale, protocol unfrozen, held-out unused, technical report missing, and the original “misleading amplifies PGD” claim is **not supported** for color under the current design (floor). Credible path = honest negative/alternative finding + freeze + one held-out confirm + rewritten report/README before 16 Nov.
