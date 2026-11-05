"""Phase E preparation safeguards (no model download, no held-out inference)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from crossmodal_robustness.freeze_pins import (
    EXPECTED_TEST_COUNT,
    FROZEN_EPSILONS,
    FROZEN_MODEL_ID,
    FROZEN_REVISION,
    FROZEN_SEED,
)
from crossmodal_robustness.heldout_analyze import compare_splits
from crossmodal_robustness.heldout_eval import (
    assert_unique_heldout_output,
    load_heldout_records,
    main as heldout_main,
)
from crossmodal_robustness.phase_d_analyze import build_metrics
from crossmodal_robustness.seeds import attack_noise_seed, stable_seed_int


class StableSeedTests(unittest.TestCase):
    def test_stable_seed_deterministic_across_calls(self):
        a = stable_seed_int("color", 0.03137255, "noise")
        b = stable_seed_int("color", 0.03137255, "noise")
        self.assertEqual(a, b)

    def test_stable_seed_differs_by_task(self):
        self.assertNotEqual(
            stable_seed_int("color", "x"),
            stable_seed_int("shape", "x"),
        )

    def test_attack_noise_seed_stable_and_independent_of_python_hash(self):
        seed_a = attack_noise_seed(
            base_seed=FROZEN_SEED,
            image_index=3,
            task="color",
            attack_name="noise",
            epsilon=8 / 255,
        )
        seed_b = attack_noise_seed(
            base_seed=FROZEN_SEED,
            image_index=3,
            task="color",
            attack_name="noise",
            epsilon=8 / 255,
        )
        self.assertEqual(seed_a, seed_b)
        # Must not depend on process-randomized hash() of tuples.
        with mock.patch("builtins.hash", side_effect=AssertionError("hash() must not be used")):
            seed_c = attack_noise_seed(
                base_seed=FROZEN_SEED,
                image_index=3,
                task="color",
                attack_name="noise",
                epsilon=8 / 255,
            )
        self.assertEqual(seed_a, seed_c)


class HeldoutSplitTests(unittest.TestCase):
    def test_load_heldout_records_validates_32(self):
        manifest = Path("data/synthetic_v1/manifest.jsonl")
        if not manifest.is_file():
            self.skipTest("manifest not present")
        records = load_heldout_records(manifest)
        self.assertEqual(len(records), EXPECTED_TEST_COUNT)
        self.assertTrue(all(r["split"] == "test" for r in records))

    def test_load_heldout_records_rejects_wrong_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            rows = []
            for i in range(3):
                rows.append({
                    "id": f"item_{i:03d}",
                    "split": "test",
                    "image": f"images/item_{i:03d}.png",
                    "color": "red",
                    "shape": "circle",
                })
            # Create dummy images so load_manifest path checks pass.
            img_dir = Path(tmp) / "images"
            img_dir.mkdir()
            for i in range(3):
                (img_dir / f"item_{i:03d}.png").write_bytes(b"not-a-real-png")
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            # load_manifest checks is_file(); fake png is fine. Count must fail.
            with self.assertRaises(RuntimeError):
                load_heldout_records(path)

    def test_unique_output_rejects_phase_d_stems(self):
        with self.assertRaises(ValueError):
            assert_unique_heldout_output(Path("results/phase_d_full64.json"))
        with self.assertRaises(ValueError):
            assert_unique_heldout_output(Path("results/misc_run.json"))
        assert_unique_heldout_output(Path("results/phase_e_heldout.json"))

    def test_validate_only_cli_no_inference(self):
        manifest = Path("data/synthetic_v1/manifest.jsonl")
        if not manifest.is_file():
            self.skipTest("manifest not present")
        with mock.patch("crossmodal_robustness.heldout_eval.load_model") as load_model:
            heldout_main(["--validate-only", "--manifest", str(manifest)])
            load_model.assert_not_called()

    def test_inference_refused_without_authorization(self):
        manifest = Path("data/synthetic_v1/manifest.jsonl")
        if not manifest.is_file():
            self.skipTest("manifest not present")
        with self.assertRaises(SystemExit):
            heldout_main(["--manifest", str(manifest)])


class AnalyzeNoHardcodedDevStatsTests(unittest.TestCase):
    def _toy_rows(self, *, n_images: int, misleading_correct: int) -> list[dict]:
        rows = []
        for i in range(n_images):
            correct = i < misleading_correct
            rows.append({
                "image_id": f"img_{i:03d}",
                "task": "color",
                "attack": "clean",
                "epsilon": 0.0,
                "evaluations": [
                    {"condition": "neutral", "correct": True, "invalid": False},
                    {"condition": "supportive", "correct": True, "invalid": False},
                    {"condition": "misleading", "correct": correct, "invalid": False},
                ],
            })
        return rows

    def test_floor_comes_from_rows_not_hardcoded_6_of_64(self):
        rows = self._toy_rows(n_images=8, misleading_correct=2)
        metrics = build_metrics(rows, n_boot=50)
        floor = metrics["floor_effect_color_misleading"]
        self.assertEqual(floor["n_images"], 8)
        self.assertEqual(floor["correct"], 2)
        self.assertAlmostEqual(floor["clean_accuracy_misleading_color"], 0.25)
        # Guard against resurrecting the development constant.
        blob = json.dumps(floor)
        self.assertNotIn("6/64", blob)
        self.assertNotIn("0.09375", blob)

    def test_compare_splits_keeps_dev_and_test_separate(self):
        held = build_metrics(self._toy_rows(n_images=4, misleading_correct=1), n_boot=20)
        dev = build_metrics(self._toy_rows(n_images=6, misleading_correct=1), n_boot=20)
        compared = compare_splits(heldout_metrics=held, development_metrics=dev)
        self.assertIs(compared["heldout"], held)
        self.assertIs(compared["development"], dev)
        self.assertEqual(
            compared["floor_side_by_side"]["heldout"]["n_images"], 4
        )
        self.assertEqual(
            compared["floor_side_by_side"]["development"]["n_images"], 6
        )


class FreezePinTests(unittest.TestCase):
    def test_frozen_pins_match_protocol_values(self):
        self.assertEqual(FROZEN_MODEL_ID, "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        self.assertEqual(FROZEN_REVISION, "7b375e1b73b11138ff12fe22c8f2822d8fe03467")
        self.assertEqual(FROZEN_SEED, 20261105)
        self.assertEqual(FROZEN_EPSILONS, (2 / 255, 4 / 255, 8 / 255))
        self.assertEqual(EXPECTED_TEST_COUNT, 32)


if __name__ == "__main__":
    unittest.main()
