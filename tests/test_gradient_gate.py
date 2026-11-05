"""Lightweight tests for Phase B helpers (no model download required)."""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class GradientGateHelperTests(unittest.TestCase):
    def test_answer_token_ids_cover_label_set(self):
        from transformers import AutoTokenizer
        from crossmodal_robustness.gradient_gate import answer_token_ids

        # Tokenizer-only load is enough for this check.
        class _Proc:
            tokenizer = AutoTokenizer.from_pretrained(
                "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
            )

        processor = _Proc()
        for word in ("red", "blue", "green", "yellow", "circle", "square", "star", "triangle"):
            ids = answer_token_ids(processor, word)
            self.assertGreaterEqual(ids.numel(), 1)
            tokens = processor.tokenizer.convert_ids_to_tokens(ids[0])
            self.assertTrue(tokens[0].startswith("Ġ") or tokens[0].startswith("▁"))
        # Bare (no leading space) encoding differs and is the wrong attack target.
        bare = answer_token_ids(processor, "red", leading_space=False)
        spaced = answer_token_ids(processor, "red", leading_space=True)
        self.assertFalse(torch.equal(bare, spaced))

    def test_differentiable_pixel_values_keep_grad(self):
        from transformers import AutoProcessor
        from crossmodal_robustness.gradient_gate import differentiable_pixel_values

        processor = AutoProcessor.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        )
        image = torch.rand(3, 64, 64, dtype=torch.float32, requires_grad=True)
        pixel_values, mask = differentiable_pixel_values(
            processor, image, do_image_splitting=False
        )
        self.assertTrue(pixel_values.requires_grad or pixel_values.grad_fn is not None)
        loss = pixel_values.mean()
        loss.backward()
        self.assertIsNotNone(image.grad)
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertEqual(tuple(mask.shape[:2]), tuple(pixel_values.shape[:2]))


if __name__ == "__main__":
    unittest.main()
