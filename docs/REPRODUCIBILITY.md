# Reproducibility and educational guide

This document explains **how the frozen study was run** and how a newcomer can re-check numbers **without inventing observations**. Phase F does not re-run model inference.

Companion beginner narrative: [`BEGINNER_WALKTHROUGH.md`](BEGINNER_WALKTHROUGH.md).

---

## 1. Conceptual walkthrough

1. **Stimuli.** Each PNG has a known color and shape. Prompts attach a neutral, supportive, or misleading context sentence, then ask for color or shape.  
2. **Clean eval.** Generate a short answer; strict parser → correct / incorrect / invalid.  
3. **Attack.** Under the **neutral** prompt, ascend teacher-forced NLL of the correct leading-space answer tokens w.r.t. pixels (FGSM or PGD), or add equal-budget noise.  
4. **Paired eval.** Reuse the **same** perturbed image under all three contexts.  
5. **Estimands.** Additional error \(\Delta = e_{\mathrm{adv}}-e_{\mathrm{clean}}\). Primary DiD averages \(\Delta_{\mathrm{mis}}-\Delta_{\mathrm{sup}}\) over images; secondary uses neutral instead of misleading. Bootstrap images for CIs.

---

## 2. Environment (recording host)

| Pin | Value |
| --- | --- |
| Python | 3.12.14 |
| torch | 2.15.0 |
| torchvision | 0.29.2 |
| transformers | 5.22.1 |
| Device | Apple MPS (MacBook Air M4, 24 GB) |
| Model | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| HF revision | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| Seed | 20261105 |
| ε | 2/255, 4/255, 8/255 |
| PGD | 7 steps, α=ε/4, random_start=false |
| Splitting | `do_image_splitting=False` |

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,plot]'
python -m pytest -q
```

---

## 3. Exact commands (as used)

### 3.1 Development (Phase D; already executed)

```bash
python -m crossmodal_robustness.phase_d_eval \
  --manifest data/synthetic_v1/manifest.jsonl \
  --subset all \
  --device mps \
  --task both \
  --epsilons 0.00784313725490196 0.01568627450980392 0.03137254901960784 \
  --steps 7 \
  --attacks noise fgsm pgd \
  --seed 20261105 \
  --output results/phase_d_full64.json
```

Wall ≈ **15853 s** (`reports/phase-d-full64-meta.json`).

### 3.2 Held-out one-shot (Phase E; already executed)

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

Wall ≈ **11149 s** (`reports/phase-e-heldout-meta.json`). Peak MPS memory on the order of ~4 GB in Phase D logs.

### 3.3 Analyze (no attack craft)

```bash
python -m crossmodal_robustness.heldout_analyze \
  --jsonl results/phase_e_heldout.jsonl \
  --development-metrics reports/phase-d-full64-metrics.json \
  --output results/phase_e_heldout_metrics.json \
  --figure-dir reports/figures/phase-e
```

---

## 4. Outputs and what is committed

| Path | Committed? | Role |
| --- | --- | --- |
| `results/phase_d_full64.jsonl` | **No** (gitignored) | Full development predictions |
| `results/phase_e_heldout.jsonl` | **No** (gitignored) | Full held-out predictions |
| `reports/phase-d-full64-metrics.json` | Yes | Development aggregate metrics |
| `reports/phase-e-heldout-metrics.json` | Yes | Held-out + side-by-side |
| `reports/artifacts/*_predictions_compact.jsonl` | Yes | Flattened prediction table derived from local JSONL |
| `reports/artifacts/checksums.json` | Yes | SHA-256 of committed + derived artifacts |
| `reports/artifacts/experiment_metadata.json` | Yes | Immutable experiment metadata |

**If a local `results/*.jsonl` file is missing:** say so and use committed metrics / compact tables. **Do not** regenerate observations for Phase F documentation. On this authoring host both JSONLs were present and hashed into `experiment_metadata.json`.

---

## 5. Worked DiD + bootstrap (educational)

See `reports/technical_report.md` §6.5 for `item_004` / color / PGD ε=8/255 from existing predictions:

- \(\Delta_{\mathrm{neu}}=1\), \(\Delta_{\mathrm{sup}}=0\), \(\Delta_{\mathrm{mis}}=0\)  
- Image contribution to primary DiD: \(0\); to secondary: \(1\).

Bootstrap: draw 32 image indices with replacement, recompute the mean DiD, repeat 2000 times, take 2.5/97.5 percentiles (seed 20261105). Implementation: `crossmodal_robustness.phase_d_analyze.bootstrap_did`.

---

## 6. Process disclosures

- Leading-space token fix after Phase B (`Ġred`).  
- Revision verification hotfix `d1ef71e` (hub SHA) after attempt-1 abort.  
- Noise seeds: MD5-stable `attack_noise_seed` (pre-test addendum); distribution unchanged vs Phase D, tensors not bit-identical.  
- Held-out authorization required `--i-authorize-heldout-oneshot`.

---

## 7. What not to do

- Do not change ε, steps, prompts, parser, preprocess, or primary estimand after seeing held-out numbers.  
- Do not present secondary DiD as the original primary claim.  
- Do not merge historical Phase A–D PRs separately if consolidating on the Phase F tip (editorial recommendation only; merge is Federico’s decision).
