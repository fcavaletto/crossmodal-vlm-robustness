#!/usr/bin/env bash
# Frozen Checkpoint-1 held-out one-shot. Requires Federico authorization.
# Do NOT run until --i-authorize-heldout-oneshot is intentionally added.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Validate sealed test split (no inference)..."
python -m crossmodal_robustness.heldout_eval --validate-only

cat <<'EOF'

To execute the authorized one-shot (ONLY after Federico approval):

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

python -m crossmodal_robustness.heldout_analyze \
  --jsonl results/phase_e_heldout.jsonl \
  --development-metrics reports/phase-d-full64-metrics.json \
  --output results/phase_e_heldout_metrics.json \
  --figure-dir reports/figures/phase-e
EOF
