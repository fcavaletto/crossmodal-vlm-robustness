"""An independently testable L-infinity projected-gradient ascent primitive.

Input tensor must contain RGB values on [0,1] *before* model normalization.
The provided loss function must perform differentiable preprocessing and return
a scalar loss to MAXIMIZE. No attack against the VLM is claimed yet.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def pgd_linf(
    image: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    epsilon: float = 8 / 255,
    step_size: float = 2 / 255,
    steps: int = 10,
    random_start: bool = False,
) -> torch.Tensor:
    """Maximize a differentiable loss subject to pixel and L-infinity bounds."""
    if not image.is_floating_point():
        raise TypeError("image must be a floating-point tensor in [0, 1]")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("image values must be finite and between 0 and 1")
    if epsilon < 0 or step_size <= 0 or steps < 0:
        raise ValueError("epsilon >= 0, step_size > 0, and steps >= 0 required")

    source = image.detach().clone()
    adversarial = source.clone()
    if random_start and epsilon > 0:
        adversarial = (source + torch.empty_like(source).uniform_(-epsilon, epsilon)).clamp(0, 1)

    for _ in range(steps):
        adversarial = adversarial.detach().requires_grad_(True)
        loss = loss_fn(adversarial)
        if loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar tensor")
        gradient = torch.autograd.grad(loss, adversarial)[0]
        candidate = adversarial.detach() + step_size * gradient.sign()
        adversarial = torch.minimum(torch.maximum(candidate, source - epsilon), source + epsilon)
        adversarial = adversarial.clamp(0.0, 1.0)
    return adversarial.detach()

