"""Validate teacher-forced answer NLL under no-split before running attacks.

Checks:
1. Greedy generation under ``do_image_splitting=False`` for the probe image.
2. Conditional NLL of the correct label vs incorrect color/shape alternatives.
3. PIL vs differentiable-torch preprocess: tensor diffs and generation agreement.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .dataset import COLORS, SHAPES, build_prompt, load_manifest
from .evaluate import DEFAULT_MODEL, load_model, run_inference, select_device
from .gradient_gate import (
    answer_token_ids,
    build_prompt_tensors,
    differentiable_pixel_values,
    freeze_model,
    load_image_01,
    teacher_forced_nll,
)


def score_answer(
    processor,
    model,
    *,
    image_pil: Image.Image,
    image_01: torch.Tensor | None,
    prompt: str,
    answer: str,
    device: str,
    do_image_splitting: bool,
    pixel_source: str,
) -> dict[str, Any]:
    """Teacher-forced mean NLL for ``answer`` with a chosen pixel path."""
    base = build_prompt_tensors(
        processor, image_pil, prompt, do_image_splitting=do_image_splitting
    )
    input_ids = base["input_ids"].to(device)
    attention_mask = base["attention_mask"].to(device)
    answer_ids = answer_token_ids(processor, answer)

    if pixel_source == "pil":
        with torch.no_grad():
            processed = processor.image_processor(
                images=image_pil,
                return_tensors="pt",
                do_image_splitting=do_image_splitting,
            )
        pixel_values = processed["pixel_values"].to(device)
        pixel_attention_mask = processed["pixel_attention_mask"].to(device)
    elif pixel_source == "torch":
        assert image_01 is not None
        pixel_values, pixel_attention_mask = differentiable_pixel_values(
            processor, image_01, do_image_splitting=do_image_splitting
        )
        pixel_values = pixel_values.to(device)
        pixel_attention_mask = pixel_attention_mask.to(device)
    else:
        raise ValueError(pixel_source)

    with torch.no_grad():
        loss = teacher_forced_nll(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            answer_ids=answer_ids,
        )
    return {
        "answer": answer,
        "token_ids": answer_ids[0].tolist(),
        "tokens": processor.tokenizer.convert_ids_to_tokens(answer_ids[0]),
        "nll": float(loss),
        "pixel_source": pixel_source,
        "prompt_tokens": int(input_ids.shape[-1]),
    }


def generate_with_pixel_values(
    processor,
    model,
    *,
    image_pil: Image.Image,
    prompt: str,
    pixel_values: torch.Tensor,
    pixel_attention_mask: torch.Tensor,
    device: str,
    do_image_splitting: bool,
    max_new_tokens: int = 12,
) -> str:
    """Greedy decode using caller-supplied ``pixel_values`` (preprocess A/B test)."""
    base = build_prompt_tensors(
        processor, image_pil, prompt, do_image_splitting=do_image_splitting
    )
    inputs = {
        "input_ids": base["input_ids"].to(device),
        "attention_mask": base["attention_mask"].to(device),
        "pixel_values": pixel_values.to(device),
        "pixel_attention_mask": pixel_attention_mask.to(device),
    }
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    prompt_size = inputs["input_ids"].shape[-1]
    return processor.batch_decode(generated[:, prompt_size:], skip_special_tokens=True)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/synthetic_v1/manifest.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="mps", choices=["auto", "mps", "cpu"])
    parser.add_argument("--image-id", default="item_000")
    parser.add_argument("--task", default="color", choices=["color", "shape"])
    parser.add_argument("--condition", default="neutral",
                        choices=["neutral", "supportive", "misleading"])
    parser.add_argument("--output", type=Path, default=Path("results/phase_c_loss_validation.json"))
    args = parser.parse_args()

    device = select_device(args.device)
    records = {r["id"]: r for r in load_manifest(args.manifest, split="development")}
    record = records[args.image_id]
    prompt = build_prompt(record, args.task, args.condition)
    correct = record[args.task]
    alternatives = list(COLORS if args.task == "color" else SHAPES)

    print(f"Loading {args.model} on {device}", flush=True)
    processor, model = load_model(args.model, device)
    freeze_model(model)

    with Image.open(record["absolute_image"]) as handle:
        image_pil = handle.convert("RGB")
    image_01 = load_image_01(record["absolute_image"])

    # 1) Generation under no-split (PIL eval path — same as baseline evaluator).
    t0 = time.perf_counter()
    generated = run_inference(
        processor,
        model,
        Path(record["absolute_image"]),
        prompt,
        12,
        do_image_splitting=False,
        image=image_pil,
    )
    gen_s = time.perf_counter() - t0

    # 2) NLL for all alternatives under PIL and torch pixel paths.
    scores_pil = []
    scores_torch = []
    for answer in alternatives:
        scores_pil.append(
            score_answer(
                processor, model,
                image_pil=image_pil, image_01=image_01, prompt=prompt, answer=answer,
                device=device, do_image_splitting=False, pixel_source="pil",
            )
        )
        scores_torch.append(
            score_answer(
                processor, model,
                image_pil=image_pil, image_01=image_01, prompt=prompt, answer=answer,
                device=device, do_image_splitting=False, pixel_source="torch",
            )
        )

    # 3) Preprocess tensor diff + generation with swapped pixel_values.
    with torch.no_grad():
        pil_pv = processor.image_processor(
            images=image_pil, return_tensors="pt", do_image_splitting=False
        )
        torch_pv, torch_mask = differentiable_pixel_values(
            processor, image_01, do_image_splitting=False
        )
    pv_diff = {
        "max_abs": float((pil_pv["pixel_values"] - torch_pv).abs().max()),
        "mean_abs": float((pil_pv["pixel_values"] - torch_pv).abs().mean()),
        "shape": list(torch_pv.shape),
    }
    gen_from_pil_pv = generate_with_pixel_values(
        processor, model, image_pil=image_pil, prompt=prompt,
        pixel_values=pil_pv["pixel_values"], pixel_attention_mask=pil_pv["pixel_attention_mask"],
        device=device, do_image_splitting=False,
    )
    gen_from_torch_pv = generate_with_pixel_values(
        processor, model, image_pil=image_pil, prompt=prompt,
        pixel_values=torch_pv, pixel_attention_mask=torch_mask,
        device=device, do_image_splitting=False,
    )

    def ranking(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered = sorted(scores, key=lambda row: row["nll"])
        return [{"answer": row["answer"], "nll": row["nll"]} for row in ordered]

    correct_pil = next(row for row in scores_pil if row["answer"] == correct)
    wrong_pil = [row for row in scores_pil if row["answer"] != correct]
    best_wrong_pil = min(wrong_pil, key=lambda row: row["nll"])
    margin_pil = best_wrong_pil["nll"] - correct_pil["nll"]

    correct_torch = next(row for row in scores_torch if row["answer"] == correct)
    wrong_torch = [row for row in scores_torch if row["answer"] != correct]
    best_wrong_torch = min(wrong_torch, key=lambda row: row["nll"])
    margin_torch = best_wrong_torch["nll"] - correct_torch["nll"]

    payload = {
        "phase": "C",
        "check": "loss_validation_nosplit",
        "image_id": record["id"],
        "task": args.task,
        "condition": args.condition,
        "correct_label": correct,
        "prompt": prompt,
        "do_image_splitting": False,
        "generation": {
            "raw": generated,
            "seconds": round(gen_s, 3),
            "matches_correct_label": generated.strip().lower().rstrip(".") == correct,
        },
        "nll_ranking_pil": ranking(scores_pil),
        "nll_ranking_torch": ranking(scores_torch),
        "correct_nll_pil": correct_pil["nll"],
        "correct_nll_torch": correct_torch["nll"],
        "best_wrong_pil": {"answer": best_wrong_pil["answer"], "nll": best_wrong_pil["nll"]},
        "best_wrong_torch": {"answer": best_wrong_torch["answer"], "nll": best_wrong_torch["nll"]},
        "margin_best_wrong_minus_correct_pil": margin_pil,
        "margin_best_wrong_minus_correct_torch": margin_torch,
        "correct_is_lowest_nll_pil": ranking(scores_pil)[0]["answer"] == correct,
        "correct_is_lowest_nll_torch": ranking(scores_torch)[0]["answer"] == correct,
        "preprocess_diff": pv_diff,
        "generation_from_pil_pixel_values": gen_from_pil_pv,
        "generation_from_torch_pixel_values": gen_from_torch_pv,
        "pil_torch_generation_agree": gen_from_pil_pv == gen_from_torch_pv,
        "token_alignment_correct": {
            "answer": correct,
            "ids": correct_pil["token_ids"],
            "tokens": correct_pil["tokens"],
        },
    }
    # Gate: generation should prefer the correct short label under neutral, and
    # correct NLL should be strictly below at least one wrong alternative's NLL
    # (ideally the lowest). Soft fail if correct is not uniquely best but still
    # generates correctly — report clearly.
    payload["passed"] = bool(
        payload["generation"]["matches_correct_label"]
        and payload["correct_is_lowest_nll_pil"]
        and payload["correct_is_lowest_nll_torch"]
        and margin_pil > 0
        and margin_torch > 0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "generated": generated,
        "correct_nll_pil": correct_pil["nll"],
        "correct_nll_torch": correct_torch["nll"],
        "ranking_pil": ranking(scores_pil),
        "margin_pil": margin_pil,
        "margin_torch": margin_torch,
        "pv_max_abs_diff": pv_diff["max_abs"],
        "gen_agree": payload["pil_torch_generation_agree"],
        "passed": payload["passed"],
    }, indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
