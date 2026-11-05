"""Fast Gradient Sign Method in L-infinity / [0,1] pixel space."""

from __future__ import annotations

from collections.abc import Callable

import torch


def fgsm_linf(
    image: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    epsilon: float = 8 / 255,
) -> torch.Tensor:
    """One-step signed-gradient ascent of ``loss_fn``, clipped to the epsilon ball."""
    if not image.is_floating_point():
        raise TypeError("image must be a floating-point tensor in [0, 1]")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("image values must be finite and between 0 and 1")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")

    source = image.detach()
    adversarial = source.clone().requires_grad_(True)
    loss = loss_fn(adversarial)
    if loss.ndim != 0:
        raise ValueError("loss_fn must return a scalar tensor")
    gradient = torch.autograd.grad(loss, adversarial)[0]
    candidate = source + epsilon * gradient.sign()
    return candidate.clamp(0.0, 1.0).detach()
