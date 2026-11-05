import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed; optional PGD tests skipped")
class PGDTests(unittest.TestCase):
    def test_gradient_ascends_and_respects_bounds(self):
        from crossmodal_robustness.attacks.pgd import pgd_linf

        source = torch.full((1, 3, 8, 8), 0.5, dtype=torch.float32)
        def objective(image):
            return image.mean()
        adversarial = pgd_linf(source, objective, epsilon=8 / 255,
                               step_size=2 / 255, steps=10)
        self.assertLessEqual((adversarial - source).abs().max().item(), 8 / 255 + 1e-6)
        self.assertGreater(objective(adversarial).item(), objective(source).item())
        self.assertTrue(bool(((adversarial >= 0) & (adversarial <= 1)).all()))
        self.assertTrue(bool((source == 0.5).all()))

    def test_invalid_arguments(self):
        from crossmodal_robustness.attacks.pgd import pgd_linf

        with self.assertRaises(ValueError):
            pgd_linf(torch.zeros(2), lambda x: x.sum(), epsilon=-0.1)


if __name__ == "__main__":
    unittest.main()

