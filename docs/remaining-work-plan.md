# Remaining work plan — through 16 Nov 2026

**Basis tip:** `cursor/phase-d-dev-eval-dabf` (PR #4)  
**Calendar:** ~11 days from 2026-11-05 → deadline 2026-11-16  
**Constraint this checkpoint:** plan only — **no** new expensive runs, **no** held-out, **no** merge until Federico approves.

Maps to handoff Parts / Phases: **I freeze → II held-out → III technical report → IV reproducibility/README → V portfolio packaging**.

---

## Priority order (do in sequence)

1. **Federico decisions (blocking)** — claim framing + freeze vs soften  
2. **Part I — Freeze protocol** (docs only, or light re-analysis; no test)  
3. **Part II — Held-out evaluation** (once, after freeze)  
4. **Part III — Technical report**  
5. **Part IV — README + reproduce script**  
6. **Part V — PR consolidation / portfolio polish**  
7. Optional only if time: milder-ε appendixendum, Qwen transfer, supportive-opt attack

---

## Part I — Protocol freeze (after approval)

**Goal:** Lock the study definition so held-out is not contaminated by tuning.

**Decisions Federico must make first**

| # | Decision | Options | Recommendation |
| --- | --- | --- | --- |
| D1 | Primary reported interaction | (a) misleading−supportive DiD with floor caveat; (b) **neutral−supportive mitigation** as primary; (c) both, ranked | **(b) primary, (a) secondary with floor** |
| D2 | Freeze attack grid | Keep ε∈{2,4,8}/255, PGD7, α=ε/4, opt@neutral, no-split | **Freeze as-run** (honest ceiling note) unless he wants a milder appendixendum |
| D3 | Soften attacks? | New smaller ε / fewer steps on **dev only** then re-freeze | Only if he rejects ceiling narrative; costs ≥0.5–1 day |
| D4 | Merge strategy | Squash/stack PRs #1–4 into tip then merge once; or keep tip branch only | Defer merge to end of Part V |

**Deliverables**

- `docs/PROTOCOL.md` → freeze revision (stack, tokens, preprocess, metrics, floor policy, sealed test)  
- `docs/freeze-config.md` (short machine-readable summary: model rev, ε, steps, seeds, flags)  
- Explicit “no further hyperparameter search” statement  

**Effort:** 0.5–1 day docs; **no GPU** unless D3.

---

## Part II — Held-out evaluation (Checkpoint 1 execution — not this turn)

**Only after Part I freeze.**

- Run same no-split stack on **32 test images** (both tasks; clean + PGD±controls at frozen ε)  
- Optimize under **neutral**; evaluate N/S/M with same pixels  
- Compute the **predeclared** metrics only (ASR, additional error, DiD + image bootstrap, invalid rates, bounds)  
- Store: `results/phase_e_heldout.jsonl`, `reports/phase-e-heldout-*.json`, figures  
- **One shot** — no peek-then-retune  

**Effort:** ~2–3 hours wall (32/64 of Phase D attack cost ≈ half day with analysis).

**Hard stop if anything in freeze is still ambiguous.**

---

## Part III — Technical report

`reports/technical_report.md` (and optional PDF later):

1. Abstract (methods demo; synthetic; negative/alternative finding OK)  
2. Hypothesis vs what was identifiable  
3. Threat model  
4. Methods (model, no-split, leading-space tokens, attacks, metrics)  
5. Development results (Phase D) + held-out (Part II)  
6. **Negative findings** + floor analysis  
7. Phase B tokenization validation lesson  
8. Limitations + future work  
9. Related work (honest non-novelty)

Pull figures from `reports/figures/phase-d/` (+ held-out figs).

**Effort:** 1–2 days writing after numbers exist.

---

## Part IV — Reproducibility & README rewrite

- Rewrite `README.md`: status truthful; no-split; how to reproduce; link Phase A–E reports  
- Add `scripts/reproduce_phase_d.sh` / `reproduce_heldout.sh` with pinned model id + recorded commit hash  
- Record env: Python 3.12, torch 2.15, transformers 5.22.1, HF rev used  
- Refresh `CURSOR_HANDOFF.md` to “completed through D; E pending” or archive as historical  

**Effort:** 0.5–1 day.

---

## Part V — Portfolio packaging

- Prefer **one** tip branch / one merge-ready PR into `main` after report (avoid four stale PRs confusing readers)  
- Ensure `reports/` + key figures committed; large JSONL stays gitignored or released as archive note  
- Optional short `docs/portfolio-readme.md` for AI4I application narrative  
- Close/supersede PRs #1–3 pointing to tip  

**Effort:** 0.5 day.

---

## Documentation file list

### Create

| File | Part |
| --- | --- |
| `docs/freeze-config.md` | I |
| `reports/technical_report.md` | III |
| `reports/phase-e-heldout-verification.md` | II |
| `scripts/reproduce_*.sh` (or `Makefile` targets) | IV |
| `docs/portfolio-narrative.md` (optional) | V |

### Rewrite

| File | Why |
| --- | --- |
| `README.md` | Currently false (“no model run”) |
| `docs/PROTOCOL.md` | Draft → frozen protocol |
| `CURSOR_HANDOFF.md` | Pre-A instructions → completion map |

### Keep as immutable logs

`reports/phase-{a,b,c,d}-verification.md` and metrics JSON/figures.

### Do not overwrite

Phase A split-ON summaries; they document a different preprocess.

---

## Schedule to 16 Nov 2026 (realistic)

| Window | Focus | Output |
| --- | --- | --- |
| **Nov 5–6** | Checkpoint 0 review (this plan) | Federico D1–D3 |
| **Nov 6–7** | Part I freeze docs | PROTOCOL + freeze-config |
| **Nov 7–8** | Part II held-out **once** | phase-e metrics + figs |
| **Nov 8–12** | Part III technical report | `technical_report.md` |
| **Nov 12–14** | Part IV README/reproduce | truthful entrypoint |
| **Nov 14–15** | Part V packaging / PR tidy | merge-ready tip |
| **Nov 16** | Buffer / final proofread | submit |

Slack: if held-out slips, **still ship** report with development-only + sealed-test statement (weaker but honest). Do **not** burn buffer on Qwen/LoRA.

---

## Risks Federico may have overlooked

1. **`main` is a liability** — anyone opening GitHub sees “no results”; tip is PR #4.  
2. **Claim mismatch** — application narrative still sounds like “misleading amplifies attacks”; data say floor + supportive mitigation. Align story early.  
3. **Ceiling at ε=2/255** — looks like overpowered attack; reviewers may ask for weaker ε; decide now (freeze vs appendixendum).  
4. **Invalid-as-error** under PGD inflates ASR vs “wrong color” story — report parse failures separately.  
5. **Dual-task dependence** — bootstrap is image-level for DiD, but tables still show both tasks; don’t over-pool.  
6. **Time sink:** re-running ε grids or new models before README/report.  
7. **Artifact persistence:** full JSONL only under local `results/` — backup before disk cleanup.  
8. **Stacked open PRs** — merge confusion; supersede explicitly.  
9. **Phase A vs C baselines** — cite the no-split numbers for attack claims.  
10. **Deadline buffer is thin** if D3 (soften + re-dev) is chosen — that consumes held-out week.

---

## Recommended next step after approval

**Single action for Federico:** answer **D1 (primary claim)** and **D2 (freeze as-run vs soften)** in one message.

Then authorize **Checkpoint 1 = Part I freeze docs only** (still no held-out). Held-out becomes Checkpoint 2 after freeze file exists.
