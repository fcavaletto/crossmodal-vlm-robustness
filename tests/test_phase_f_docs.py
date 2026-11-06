"""Doc/metric consistency checks for Phase F (no model inference)."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PhaseFDocConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = json.loads(
            (ROOT / "reports/phase-e-heldout-metrics.json").read_text(encoding="utf-8")
        )
        cls.report = (ROOT / "reports/technical_report.md").read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.artifacts = ROOT / "reports/artifacts"

    def test_heldout_clean_floor_matches_metrics(self):
        floor = self.metrics["heldout"]["floor_effect_color_misleading"]
        self.assertEqual(floor["correct"], 4)
        self.assertEqual(floor["n_images"], 32)
        self.assertAlmostEqual(floor["clean_accuracy_misleading_color"], 0.125)
        self.assertIn("4/32 (12.5%)", self.report)
        self.assertIn("4/32 = 12.5%", self.readme)

    def test_primary_did_eps8_color_in_report(self):
        key = f"pgd|{8/255:.8f}|color|mis_vs_sup"
        side = self.metrics["did_side_by_side"][key]["heldout"]
        self.assertAlmostEqual(side["estimate"], -0.5)
        self.assertIn("−0.500", self.report)
        self.assertIn("−0.500", self.readme)

    def test_secondary_did_eps2_color_positive(self):
        key = f"pgd|{2/255:.8f}|color|neu_vs_sup"
        side = self.metrics["did_side_by_side"][key]["heldout"]
        self.assertAlmostEqual(side["estimate"], 0.40625)
        self.assertIn("0.406", self.report)

    def test_dev_clean_misleading_color_6_of_64(self):
        # Development floor lives in nested development block (may use legacy key).
        acc = self.metrics["development"]["accuracy"]
        clean = next(v for k, v in acc.items() if k.startswith("clean|") and k.endswith("|color"))
        self.assertEqual(clean["misleading"]["correct"], 6)
        self.assertEqual(clean["misleading"]["n"], 64)
        self.assertIn("6/64 (9.4%)", self.report)

    def test_artifacts_exist_with_checksums(self):
        meta = json.loads((self.artifacts / "experiment_metadata.json").read_text(encoding="utf-8"))
        checks = json.loads((self.artifacts / "checksums.json").read_text(encoding="utf-8"))
        self.assertIn("committed_artifact_checksums", meta)
        self.assertIn("reports/phase-e-heldout-metrics.json", checks["committed"])
        e_path = self.artifacts / "phase_e_heldout_predictions_compact.jsonl"
        d_path = self.artifacts / "phase_d_full64_predictions_compact.jsonl"
        self.assertTrue(e_path.is_file())
        self.assertTrue(d_path.is_file())
        self.assertEqual(sum(1 for _ in e_path.open()), 1920)
        self.assertEqual(sum(1 for _ in d_path.open()), 3840)

    def test_readme_no_longer_claims_no_model_run(self):
        self.assertNotIn("No VLM model run or adversarial result is claimed yet", self.readme)
        self.assertIn("does **not** support the amplification claim", self.readme)

    def test_protocol_still_frozen_not_rewritten(self):
        protocol = (ROOT / "docs/PROTOCOL.md").read_text(encoding="utf-8")
        self.assertIn("FROZEN", protocol)
        self.assertIn("misleading textual context", protocol.lower())
        # Report must acknowledge freeze-after-development chronology.
        self.assertIn("frozen after phase d development review", self.report.lower())


if __name__ == "__main__":
    unittest.main()
