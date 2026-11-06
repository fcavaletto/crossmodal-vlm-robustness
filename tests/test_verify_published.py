"""Offline reconstruction of published stats from compact prediction tables."""

from __future__ import annotations

import unittest
from pathlib import Path

from crossmodal_robustness.verify_published import (
    index_compact,
    load_compact,
    reconstruct_core,
    verify_default_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]


class VerifyPublishedTests(unittest.TestCase):
    def test_verify_default_artifacts_matches_published_metrics(self):
        report = verify_default_artifacts(ROOT)
        self.assertTrue(report["ok"], msg=report)
        self.assertEqual(report["n_mismatches"], 0)
        self.assertEqual(report["splits"]["heldout"]["n_images"], 32)
        self.assertEqual(report["splits"]["development"]["n_images"], 64)
        self.assertEqual(report["splits"]["heldout"]["n_prediction_rows"], 1920)
        self.assertEqual(report["splits"]["development"]["n_prediction_rows"], 3840)

    def test_reconstructed_primary_did_eps8_color(self):
        rows = load_compact(
            ROOT / "reports/artifacts/phase_e_heldout_predictions_compact.jsonl"
        )
        reconstructed = reconstruct_core(index_compact(rows))
        key = f"pgd|{8/255:.8f}|color|mis_vs_sup"
        stat = reconstructed["did_bootstrap"][key]
        self.assertEqual(stat["n_images"], 32)
        self.assertAlmostEqual(stat["estimate"], -0.5)
        self.assertAlmostEqual(stat["ci95"][0], -0.6875)
        self.assertAlmostEqual(stat["ci95"][1], -0.34375)

    def test_reconstructed_clean_floor(self):
        rows = load_compact(
            ROOT / "reports/artifacts/phase_e_heldout_predictions_compact.jsonl"
        )
        reconstructed = reconstruct_core(index_compact(rows))
        clean = reconstructed["accuracy"][f"clean|{0.0:.8f}|color"]["misleading"]
        self.assertEqual(clean["correct"], 4)
        self.assertEqual(clean["n"], 32)
        self.assertAlmostEqual(clean["accuracy"], 0.125)


if __name__ == "__main__":
    unittest.main()
