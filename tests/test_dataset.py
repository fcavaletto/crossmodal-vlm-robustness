import tempfile
import unittest
from pathlib import Path

from crossmodal_robustness.dataset import build_prompt, load_manifest, parse_answer
from crossmodal_robustness.generate_dataset import generate_dataset


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.tmp.name) / "synth"
        cls.metadata = generate_dataset(cls.directory)
        cls.records = load_manifest(cls.directory / "manifest.jsonl")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_balanced_full_factorial(self):
        self.assertEqual(self.metadata["count"], 96)
        self.assertEqual(self.metadata["development_count"], 64)
        self.assertEqual(self.metadata["test_count"], 32)
        self.assertEqual(len({r["id"] for r in self.records}), 96)
        self.assertEqual(len({(r["color"], r["shape"]) for r in self.records}), 16)

    def test_misleading_context_never_matches_label(self):
        for record in self.records:
            self.assertNotEqual(record["wrong_color"], record["color"])
            self.assertNotEqual(record["wrong_shape"], record["shape"])

    def test_prompts_change_only_context(self):
        record = self.records[0]
        for task in ("color", "shape"):
            neutral = build_prompt(record, task, "neutral")
            misleading = build_prompt(record, task, "misleading")
            supportive = build_prompt(record, task, "supportive")
            self.assertEqual(neutral.split("Question: ")[1], misleading.split("Question: ")[1])
            self.assertEqual(neutral.split("Question: ")[1], supportive.split("Question: ")[1])
            self.assertIn(record[f"wrong_{task}"], misleading)

    def test_strict_parser(self):
        self.assertEqual(parse_answer("Blue.", "color"), "blue")
        self.assertEqual(parse_answer("triangle", "shape"), "triangle")
        self.assertIsNone(parse_answer("It seems blue", "color"))
        self.assertIsNone(parse_answer("red or blue", "color"))
        self.assertIsNone(parse_answer("unknown", "shape"))

    def test_regeneration_is_guarded(self):
        with self.assertRaises(FileExistsError):
            generate_dataset(self.directory)


if __name__ == "__main__":
    unittest.main()

