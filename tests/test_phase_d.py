"""Unit tests for Phase D helpers (no model download)."""

from __future__ import annotations

import unittest

from crossmodal_robustness.dataset import COLORS, SHAPES
from crossmodal_robustness.phase_d_analyze import bootstrap_did, did_for_image
from crossmodal_robustness.phase_d_eval import select_balanced_development


class PhaseDHelperTests(unittest.TestCase):
    def test_balanced_subset_covers_color_shape(self):
        records = []
        index = 0
        for color in COLORS:
            for shape in SHAPES:
                for variant in range(4):
                    records.append({
                        "id": f"item_{index:03d}",
                        "color": color,
                        "shape": shape,
                        "split": "development",
                    })
                    index += 1
        selected = select_balanced_development(records, 16)
        self.assertEqual(len(selected), 16)
        combos = {(r["color"], r["shape"]) for r in selected}
        self.assertEqual(len(combos), 16)

    def test_did_bootstrap_recovers_constant_effect(self):
        per_image = {
            f"img{i}": {
                "neutral": {"error_clean": 0, "error_adv": 1, "delta_error": 1},
                "supportive": {"error_clean": 0, "error_adv": 0, "delta_error": 0},
                "misleading": {"error_clean": 0, "error_adv": 1, "delta_error": 1},
            }
            for i in range(10)
        }
        self.assertEqual(did_for_image(per_image, "misleading", "supportive"), 1.0)
        boot = bootstrap_did(per_image, primary="misleading", control="supportive", n_boot=200)
        self.assertEqual(boot["estimate"], 1.0)
        self.assertAlmostEqual(boot["ci95"][0], 1.0)
        self.assertAlmostEqual(boot["ci95"][1], 1.0)


if __name__ == "__main__":
    unittest.main()
