"""Differentiable SmolVLM answer-NLL objective matching Phase C no-split eval."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image

from crossmodal_robustness.gradient_gate import (
    answer_token_ids,
    build_prompt_tensors,
    differentiable_pixel_values,
    teacher_forced_nll,
)


def make_answer_nll_loss(
    processor,
    model,
    *,
    image_pil: Image.Image,
    prompt: str,
    answer: str,
    device: str,
    do_image_splitting: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return ``loss_fn(image_01_chw) -> scalar`` for ascent-based attacks.

    Prompt tokens are fixed from ``image_pil`` (structure only). Pixel values
    always come from the differentiable torch preprocess of the attack tensor so
    optimization and the loss see the same vision path.
    """
    base = build_prompt_tensors(
        processor, image_pil, prompt, do_image_splitting=do_image_splitting
    )
    input_ids = base["input_ids"].to(device)
    attention_mask = base["attention_mask"].to(device)
    answer_ids = answer_token_ids(processor, answer)

    def loss_fn(image_01: torch.Tensor) -> torch.Tensor:
        if image_01.ndim == 4 and image_01.shape[0] == 1:
            image_01 = image_01[0]
        pixel_values, pixel_attention_mask = differentiable_pixel_values(
            processor, image_01, do_image_splitting=do_image_splitting
        )
        return teacher_forced_nll(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values.to(device),
            pixel_attention_mask=pixel_attention_mask.to(device),
            answer_ids=answer_ids,
        )

    return loss_fn


def tensor_to_pil(image_01: torch.Tensor) -> Image.Image:
    """Convert CHW ``[0,1]`` tensor to RGB PIL for generation-time evaluation."""
    array = (
        image_01.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")
