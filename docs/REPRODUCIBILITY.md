# Reproducibility guide

Two jobs live here: (1) record **exactly** how the frozen runs were executed, and
(2) let a **fresh clone** re-check published statistics without gitignored
`results/*.jsonl` files and without downloading the model.

For the scientific story behind the estimands, prefer
[`BEGINNER_WALKTHROUGH.md`](BEGINNER_WALKTHROUGH.md) and the technical report.

---

## 1. Pipeline in five steps

Each PNG carries a known color and shape. Prompts attach a neutral, supportive,
or misleading context and then ask for color or shape. Clean evaluation uses
greedy generation and a strict parser (wrong or invalid → error). Attacks ascend
**mean** teacher-forced NLL of the correct leading-space answer tokens under the
**neutral** prompt (FGSM, PGD, or equal-budget noise), then reuse those pixels
under all three contexts. The headline estimands are additional error
\(\Delta = e_{\mathrm{adv}}-e_{\mathrm{clean}}\) and the image-mean DiDs
(primary: mis−sup; secondary: neu−sup), with image bootstrap confidence intervals.

---

## 2. Fresh clone: re-check main statistics (no GPU, no `results/`)

Committed inputs:

| Path | Role |
| --- | --- |
| `reports/artifacts/phase_e_heldout_predictions_compact.jsonl` | Flattened held-out predictions (1920 rows) |
| `reports/artifacts/phase_d_full64_predictions_compact.jsonl` | Flattened development predictions (3840 rows) |
| `reports/phase-e-heldout-metrics.json` | Published held-out aggregates (+ side-by-side) |
| `reports/phase-d-full64-metrics.json` | Published development aggregates |
| `reports/artifacts/checksums.json` | SHA-256 records |

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,plot]'
python -m pytest -q
python -m crossmodal_robustness.verify_published
```

`verify_published` rebuilds accuracy, conditional ASR, additional error, and DiD
(+ bootstrap CIs with seed `20261105`, \(B=2000\)) from the compact tables and
compares them to the published metrics JSON. Exit code 0 means match.

Optional: inspect published tables directly

```bash
python - <<'PY'
import json
m=json.load(open("reports/phase-e-heldout-metrics.json"))
print("heldout floor", m["heldout"]["floor_effect_color_misleading"])
print("primary PGD ε=8 color", m["did_side_by_side"]["pgd|0.03137255|color|mis_vs_sup"]["heldout"])
PY
```

**Do not** require `results/phase_e_heldout.jsonl` for statistic re-checks. That
path is gitignored. Compact tables + metrics JSON are the portable record.

---

## 3. Environment (recording host)

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
| Attack loss | Mean token NLL (`model(...).loss` on answer labels) |

---

## 4. Exact inference commands (historical; already executed)

These reproduce **observations** only if you intentionally re-run the one-shot
(expensive; not part of Phase F / Study 1 revisions). Prefer §2 for statistics.

### 4.1 Development (Phase D)

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

### 4.2 Held-out one-shot (Phase E)

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

Wall ≈ **11149 s** (`reports/phase-e-heldout-meta.json`). Peak MPS memory ~4 GB
in Phase D logs.

---

## 5. Outputs: committed vs local

| Path | In git? | Role |
| --- | --- | --- |
| `results/*.jsonl` | **No** | Full local prediction dumps |
| `reports/phase-*-metrics.json` | Yes | Published aggregates |
| `reports/artifacts/*_predictions_compact.jsonl` | Yes | Portable prediction tables |
| `reports/artifacts/checksums.json` | Yes | Hashes |
| `reports/artifacts/experiment_metadata.json` | Yes | Immutable metadata |

If a local `results/*.jsonl` is missing on a fresh clone: **expected**. Use §2.

---

## 6. Worked DiD + bootstrap

See `docs/BEGINNER_WALKTHROUGH.md` §§3–5 and `reports/technical_report.md` §6.5
(`item_004` / color / PGD ε=8/255).

---

## 7. Process disclosures

- Leading-space token fix after Phase B (`Ġred`).  
- Revision verification hotfix `d1ef71e` (hub SHA).  
- Noise seeds: MD5-stable `attack_noise_seed` (pre-test addendum).  
- Held-out requires `--i-authorize-heldout-oneshot`.  
- Attack objective is **mean** token NLL (not sum); see technical report §6.2.

---

## 8. Explicit caveats for readers

Small \(n\), multiple reported cells without formal multiplicity correction,
synthetic domain only, possible non-monotonic ASR vs ε, and **no mechanistic
evidence** for why supportive text helps — see technical report §9.

---

## 9. What not to do

- Do not change ε, steps, prompts, parser, preprocess, or primary estimand after
  seeing held-out numbers.  
- Do not present secondary DiD as the original primary claim.  
- Do not merge historical Phase A–D PRs separately if consolidating on the Phase F tip.
