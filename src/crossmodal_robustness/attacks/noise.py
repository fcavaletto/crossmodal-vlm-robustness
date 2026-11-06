"""Equal-budget random L-infinity noise control in [0,1] RGB space."""

from __future__ import annotations

import torch


def random_linf_noise(
    image: torch.Tensor,
    *,
    epsilon: float = 8 / 255,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add uniform noise in ``[-eps, eps]`` and project to the valid pixel cube."""
    if not image.is_floating_point():
        raise TypeError("image must be a floating-point tensor in [0, 1]")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("image values must be finite and between 0 and 1")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")

    source = image.detach()
    noise = torch.empty_like(source).uniform_(-epsilon, epsilon, generator=generator)
    return (source + noise).clamp(0.0, 1.0)
