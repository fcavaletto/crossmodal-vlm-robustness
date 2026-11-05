"""Unit tests for L-inf attack primitives (analytic objectives; no VLM)."""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class AttackPrimitiveTests(unittest.TestCase):
    def test_fgsm_respects_epsilon_and_ascends(self):
        from crossmodal_robustness.attacks import fgsm_linf

        source = torch.full((3, 8, 8), 0.5, dtype=torch.float32)
        adversarial = fgsm_linf(source, lambda image: image.mean(), epsilon=8 / 255)
        self.assertLessEqual((adversarial - source).abs().max().item(), 8 / 255 + 1e-6)
        self.assertGreater(adversarial.mean().item(), source.mean().item())
        self.assertTrue(bool(((adversarial >= 0) & (adversarial <= 1)).all()))

    def test_pgd_respects_epsilon(self):
        from crossmodal_robustness.attacks import pgd_linf

        source = torch.full((1, 3, 8, 8), 0.4, dtype=torch.float32)
        adversarial = pgd_linf(
            source, lambda image: image.mean(), epsilon=4 / 255, step_size=2 / 255, steps=5
        )
        self.assertLessEqual((adversarial - source).abs().max().item(), 4 / 255 + 1e-6)
        self.assertTrue(bool(((adversarial >= 0) & (adversarial <= 1)).all()))

    def test_random_noise_respects_epsilon(self):
        from crossmodal_robustness.attacks import random_linf_noise

        source = torch.full((3, 16, 16), 0.5, dtype=torch.float32)
        generator = torch.Generator().manual_seed(0)
        noisy = random_linf_noise(source, epsilon=8 / 255, generator=generator)
        self.assertLessEqual((noisy - source).abs().max().item(), 8 / 255 + 1e-6)
        self.assertTrue(bool(((noisy >= 0) & (noisy <= 1)).all()))
        self.assertFalse(torch.equal(noisy, source))


if __name__ == "__main__":
    unittest.main()
