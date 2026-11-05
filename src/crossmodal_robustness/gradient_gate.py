"""Phase B: end-to-end image→answer-loss gradient feasibility on frozen SmolVLM2.

Perturbations live in raw RGB ``[0, 1]``. Preprocessing uses the official
SmolVLM image processor on a torch CHW tensor (no PIL on the autograd path).
This module is a *gate*, not an attack runner.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .dataset import build_prompt, load_manifest
from .evaluate import DEFAULT_MODEL, load_model, select_device


def rss_mb() -> float:
    """macOS: ``ru_maxrss`` is bytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def mps_mem_mb() -> dict[str, float | None]:
    if not torch.backends.mps.is_available():
        return {"driver_allocated_mb": None, "current_allocated_mb": None}
    driver = torch.mps.driver_allocated_memory() / (1024**2)
    try:
        current = torch.mps.current_allocated_memory() / (1024**2)
    except Exception:
        current = None
    return {"driver_allocated_mb": round(driver, 1), "current_allocated_mb":
            None if current is None else round(current, 1)}


def load_image_01(path: str | Path) -> torch.Tensor:
    """Load RGB as float32 CHW in ``[0, 1]`` (PIL only for disk I/O, not autograd)."""
    with Image.open(path) as image_file:
        array = np.asarray(image_file.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def freeze_model(model: torch.nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def build_prompt_tensors(processor, image_pil: Image.Image, prompt: str, *,
                         do_image_splitting: bool):
    """Tokenize the multimodal prompt; pixel values are discarded (structure only)."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_pil},
        {"type": "text", "text": prompt},
    ]}]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"do_image_splitting": do_image_splitting},
    )


def differentiable_pixel_values(
    processor,
    image_01: torch.Tensor,
    *,
    do_image_splitting: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official processor on CHW ``[0,1]`` torch input; keep the autograd graph.

    Inputs already in ``[0, 1]`` must use ``do_rescale=False``; otherwise the
    processor multiplies by ``1/255`` again and destroys the signal.
    """
    if image_01.ndim != 3 or image_01.shape[0] != 3:
        raise ValueError("image_01 must be CHW with 3 channels in [0, 1]")
    outputs = processor.image_processor(
        images=image_01,
        return_tensors="pt",
        do_image_splitting=do_image_splitting,
        do_rescale=False,
    )
    return outputs["pixel_values"], outputs["pixel_attention_mask"]


def answer_token_ids(processor, answer: str, *, leading_space: bool = True) -> torch.Tensor:
    """Token ids for a short answer as it appears after ``Assistant:``.

    SmolVLM greedy decoding emits leading-space BPE pieces (``Ġred``), not the
    bare wordpiece (``red``). Scoring/attacking the bare ids yields near-zero
    probability and a misaligned objective — always prefer ``leading_space=True``
    unless deliberately probing tokenization.
    """
    text = f" {answer}" if leading_space and not answer.startswith(" ") else answer
    ids = processor.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if ids.numel() == 0:
        raise ValueError(f"Answer {answer!r} produced no tokens")
    return ids


def teacher_forced_nll(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    pixel_attention_mask: torch.Tensor,
    answer_ids: torch.Tensor,
) -> torch.Tensor:
    """Mean NLL of the answer token sequence given the multimodal prompt."""
    device = pixel_values.device
    answer_ids = answer_ids.to(device)
    full_ids = torch.cat([input_ids, answer_ids], dim=-1)
    full_mask = torch.cat([attention_mask, torch.ones_like(answer_ids)], dim=-1)
    labels = full_ids.clone()
    labels[:, : input_ids.shape[-1]] = -100
    outputs = model(
        input_ids=full_ids,
        attention_mask=full_mask,
        pixel_values=pixel_values,
        pixel_attention_mask=pixel_attention_mask,
        labels=labels,
    )
    if outputs.loss is None or not torch.isfinite(outputs.loss):
        raise RuntimeError(f"Non-finite or missing loss: {outputs.loss}")
    return outputs.loss


def run_gradient_gate(
    processor,
    model,
    *,
    image_path: Path,
    prompt: str,
    answer: str,
    device: str,
    do_image_splitting: bool,
) -> dict[str, Any]:
    """Compute ``∂L/∂image`` for correct-answer NLL; return diagnostics."""
    freeze_model(model)
    with Image.open(image_path) as handle:
        image_pil = handle.convert("RGB")

    base = build_prompt_tensors(
        processor, image_pil, prompt, do_image_splitting=do_image_splitting
    )
    input_ids = base["input_ids"].to(device)
    attention_mask = base["attention_mask"].to(device)
    answer_ids = answer_token_ids(processor, answer)

    # Reference (no-grad) processor output for numerical closeness checks.
    with torch.no_grad():
        ref_pv = processor.image_processor(
            images=image_pil,
            return_tensors="pt",
            do_image_splitting=do_image_splitting,
        )["pixel_values"]

    if device == "mps":
        torch.mps.empty_cache()
        reset_peak = getattr(torch.mps, "reset_peak_memory_stats", None)
        if callable(reset_peak):
            reset_peak()

    image = load_image_01(image_path).to(dtype=torch.float32)
    image = image.detach().requires_grad_(True)

    t0 = time.perf_counter()
    pixel_values, pixel_attention_mask = differentiable_pixel_values(
        processor, image, do_image_splitting=do_image_splitting
    )
    preprocess_match = {
        "max_abs_diff_vs_pil_processor": float((pixel_values.detach().cpu() - ref_pv).abs().max()),
        "mean_abs_diff_vs_pil_processor": float((pixel_values.detach().cpu() - ref_pv).abs().mean()),
        "pixel_values_shape": list(pixel_values.shape),
        "do_image_splitting": do_image_splitting,
        "do_rescale": False,
        "note": "Torch path uses BICUBIC when LANCZOS is unavailable; small resize diffs expected.",
    }
    pixel_values = pixel_values.to(device)
    pixel_attention_mask = pixel_attention_mask.to(device)

    loss = teacher_forced_nll(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        pixel_attention_mask=pixel_attention_mask,
        answer_ids=answer_ids,
    )
    forward_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    loss.backward()
    backward_s = time.perf_counter() - t1

    grad = image.grad
    if grad is None:
        raise RuntimeError("image.grad is None — autograd path broken")

    grad_cpu = grad.detach().cpu()
    diagnostics = {
        "answer": answer,
        "answer_token_ids": answer_ids[0].tolist(),
        "answer_tokens": processor.tokenizer.convert_ids_to_tokens(answer_ids[0]),
        "prompt_tokens": int(input_ids.shape[-1]),
        "loss": float(loss.detach().cpu()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "grad_shape": list(grad_cpu.shape),
        "grad_finite": bool(torch.isfinite(grad_cpu).all().item()),
        "grad_abs_mean": float(grad_cpu.abs().mean()),
        "grad_abs_max": float(grad_cpu.abs().max()),
        "grad_nonzero_fraction": float((grad_cpu != 0).float().mean()),
        "grad_nonzero": bool((grad_cpu != 0).any().item()),
        "forward_seconds": round(forward_s, 3),
        "backward_seconds": round(backward_s, 3),
        "rss_mb": round(rss_mb(), 1),
        **{f"mps_{k}": v for k, v in mps_mem_mb().items()},
        "preprocess": preprocess_match,
    }
    return diagnostics


def dependency_check(
    processor,
    model,
    *,
    image_a: Path,
    image_b: Path,
    prompt: str,
    answer: str,
    device: str,
    do_image_splitting: bool,
) -> dict[str, Any]:
    """Show that gradients differ when the image changes (same prompt/answer)."""
    diag_a = run_gradient_gate(
        processor, model,
        image_path=image_a, prompt=prompt, answer=answer, device=device,
        do_image_splitting=do_image_splitting,
    )
    # Clear grads / free graph residue before second pass.
    if device == "mps":
        torch.mps.empty_cache()
    diag_b = run_gradient_gate(
        processor, model,
        image_path=image_b, prompt=prompt, answer=answer, device=device,
        do_image_splitting=do_image_splitting,
    )
    return {
        "image_a": diag_a,
        "image_b": diag_b,
        "losses_differ": abs(diag_a["loss"] - diag_b["loss"]) > 1e-6,
        "grad_means_differ": abs(diag_a["grad_abs_mean"] - diag_b["grad_abs_mean"]) > 1e-12,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/synthetic_v1/manifest.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="mps", choices=["auto", "mps", "cpu"])
    parser.add_argument("--image-id", default=None, help="Development image id (default: first red circle)")
    parser.add_argument("--compare-id", default=None, help="Second image for gradient dependence check")
    parser.add_argument("--task", default="color", choices=["color", "shape"])
    parser.add_argument("--condition", default="neutral",
                        choices=["neutral", "supportive", "misleading"])
    parser.add_argument("--do-image-splitting", action=argparse.BooleanOptionalAction, default=False,
                        help="Match default SmolVLM splitting (memory-heavy on 24GB). Default: off.")
    parser.add_argument("--output", type=Path, default=Path("results/phase_b_gradient_gate.json"))
    args = parser.parse_args()

    import transformers

    device = select_device(args.device)
    records = {record["id"]: record for record in load_manifest(args.manifest, split="development")}
    if args.image_id:
        record = records[args.image_id]
    else:
        record = next(r for r in records.values() if r["color"] == "red" and r["shape"] == "circle")
    if args.compare_id:
        compare = records[args.compare_id]
    else:
        compare = next(r for r in records.values() if r["color"] == "blue" and r["shape"] == "square")

    prompt = build_prompt(record, args.task, args.condition)
    answer = record[args.task]
    print(f"Loading {args.model} on {device}; splitting={args.do_image_splitting}", flush=True)
    t_load = time.perf_counter()
    processor, model = load_model(args.model, device)
    load_s = time.perf_counter() - t_load
    freeze_model(model)

    print(f"Gate on {record['id']} answer={answer!r}", flush=True)
    primary = run_gradient_gate(
        processor, model,
        image_path=Path(record["absolute_image"]),
        prompt=prompt,
        answer=answer,
        device=device,
        do_image_splitting=args.do_image_splitting,
    )
    print(json.dumps({k: primary[k] for k in (
        "loss", "grad_finite", "grad_nonzero", "grad_abs_mean", "grad_abs_max",
        "forward_seconds", "backward_seconds", "mps_driver_allocated_mb",
    )}, indent=2), flush=True)

    print(f"Dependence check vs {compare['id']} (same prompt/answer text)", flush=True)
    # Re-build prompt/answer from *record* so only the image changes.
    dependence = dependency_check(
        processor, model,
        image_a=Path(record["absolute_image"]),
        image_b=Path(compare["absolute_image"]),
        prompt=prompt,
        answer=answer,
        device=device,
        do_image_splitting=args.do_image_splitting,
    )

    payload = {
        "phase": "B",
        "gate": "image_to_answer_nll_gradient",
        "model": args.model,
        "device": device,
        "do_image_splitting": args.do_image_splitting,
        "task": args.task,
        "condition": args.condition,
        "image_id": record["id"],
        "compare_image_id": compare["id"],
        "prompt": prompt,
        "load_seconds": round(load_s, 2),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "primary": primary,
        "dependence": {
            "losses_differ": dependence["losses_differ"],
            "grad_means_differ": dependence["grad_means_differ"],
            "loss_a": dependence["image_a"]["loss"],
            "loss_b": dependence["image_b"]["loss"],
            "grad_abs_mean_a": dependence["image_a"]["grad_abs_mean"],
            "grad_abs_mean_b": dependence["image_b"]["grad_abs_mean"],
        },
        "success_criteria": {
            "loss_finite": primary["loss_finite"],
            "grad_finite": primary["grad_finite"],
            "grad_nonzero": primary["grad_nonzero"],
            "grad_depends_on_image": dependence["grad_means_differ"] or dependence["losses_differ"],
            "no_pil_on_autograd_path": True,
        },
    }
    payload["passed"] = all(payload["success_criteria"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"passed={payload['passed']} wrote {args.output}", flush=True)
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
