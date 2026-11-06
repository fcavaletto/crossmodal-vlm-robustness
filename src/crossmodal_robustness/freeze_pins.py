"""Frozen study pins shared by Phase E held-out tooling (Checkpoint 1)."""

from __future__ import annotations

FROZEN_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
FROZEN_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
FROZEN_SEED = 20261105
FROZEN_EPSILONS = (2 / 255, 4 / 255, 8 / 255)
FROZEN_PGD_STEPS = 7
FROZEN_DO_IMAGE_SPLITTING = False
FROZEN_OPTIMIZE_CONDITION = "neutral"
EXPECTED_TEST_COUNT = 32
EXPECTED_DEVELOPMENT_COUNT = 64
