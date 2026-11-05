"""Phase C development-set attack pilot (no-split, frozen SmolVLM2).

Optimizes each image once under a fixed prompt/condition, then evaluates the
**same** perturbed image under neutral / supportive / misleading contexts.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .attacks import fgsm_linf, pgd_linf, random_linf_noise
from .attacks.vlm_loss import make_answer_nll_loss, tensor_to_pil
from .dataset import build_prompt, load_manifest, parse_answer
from .evaluate import DEFAULT_MODEL, load_model, run_inference, select_device
from .gradient_gate import freeze_model, load_image_01, mps_mem_mb, rss_mb


def evaluate_image(
    processor,
    model,
    *,
    image_pil: Image.Image,
    record: dict[str, Any],
    task: str,
    conditions: list[str],
    do_image_splitting: bool,
) -> list[dict[str, Any]]:
    rows = []
    for condition in conditions:
        prompt = build_prompt(record, task, condition)
        raw = run_inference(
            processor,
            model,
            Path(record["absolute_image"]),
            prompt,
            12,
            do_image_splitting=do_image_splitting,
            image=image_pil,
        )
        parsed = parse_answer(raw, task)
        rows.append({
            "condition": condition,
            "prompt": prompt,
            "raw_answer": raw,
            "parsed_answer": parsed,
            "expected": record[task],
            "correct": parsed == record[task],
            "invalid": parsed is None,
        })
    return rows


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    import transformers

    device = select_device(args.device)
    records = load_manifest(args.manifest, split="development")
    if args.max_images:
        records = records[: args.max_images]
    conditions = list(args.conditions)
    tasks = ["color", "shape"] if args.task == "both" else [args.task]

    print(
        f"Loading {args.model} on {device}; images={len(records)} "
        f"splitting={args.do_image_splitting} eps={args.epsilon}",
        flush=True,
    )
    processor, model = load_model(args.model, device)
    freeze_model(model)

    results: list[dict[str, Any]] = []
    t_all = time.perf_counter()
    for index, record in enumerate(records, start=1):
        with Image.open(record["absolute_image"]) as handle:
            clean_pil = handle.convert("RGB")
        clean_tensor = load_image_01(record["absolute_image"]).to(device)

        for task in tasks:
            # Optimize once under the attack condition/prompt; reuse pixels.
            attack_prompt = build_prompt(record, task, args.optimize_condition)
            answer = record[task]
            loss_fn = make_answer_nll_loss(
                processor,
                model,
                image_pil=clean_pil,
                prompt=attack_prompt,
                answer=answer,
                device=device,
                do_image_splitting=args.do_image_splitting,
            )

            with torch.no_grad():
                clean_loss = float(loss_fn(clean_tensor))

            attack_rows = []
            for attack_name in args.attacks:
                t0 = time.perf_counter()
                if attack_name == "clean":
                    adv = clean_tensor.detach()
                elif attack_name == "fgsm":
                    adv = fgsm_linf(clean_tensor, loss_fn, epsilon=args.epsilon)
                elif attack_name == "pgd":
                    adv = pgd_linf(
                        clean_tensor,
                        loss_fn,
                        epsilon=args.epsilon,
                        step_size=args.step_size,
                        steps=args.steps,
                        random_start=args.random_start,
                    )
                elif attack_name == "noise":
                    generator = torch.Generator(device=device)
                    # MPS generators may be limited — build noise on CPU then move.
                    generator = torch.Generator()
                    generator.manual_seed(args.seed + index * 17 + hash(task) % 997)
                    adv = random_linf_noise(
                        clean_tensor.cpu(), epsilon=args.epsilon, generator=generator
                    ).to(device)
                else:
                    raise ValueError(attack_name)
                attack_s = time.perf_counter() - t0

                linf = float((adv.cpu() - clean_tensor.cpu()).abs().max())
                within_eps = linf <= args.epsilon + 1e-5
                in_bounds = bool(((adv >= 0) & (adv <= 1)).all().item())
                with torch.no_grad():
                    adv_loss = float(loss_fn(adv))

                adv_pil = tensor_to_pil(adv.cpu())
                eval_rows = evaluate_image(
                    processor,
                    model,
                    image_pil=adv_pil if attack_name != "clean" else clean_pil,
                    record=record,
                    task=task,
                    conditions=conditions,
                    do_image_splitting=args.do_image_splitting,
                )
                attack_rows.append({
                    "attack": attack_name,
                    "optimize_condition": args.optimize_condition,
                    "optimize_prompt": attack_prompt,
                    "clean_nll": clean_loss,
                    "adv_nll": adv_loss,
                    "nll_increased": adv_loss > clean_loss + 1e-6,
                    "linf": linf,
                    "epsilon": args.epsilon,
                    "within_epsilon": within_eps,
                    "pixels_in_unit_interval": in_bounds,
                    "attack_seconds": round(attack_s, 3),
                    "evaluations": eval_rows,
                    "mps": mps_mem_mb(),
                    "rss_mb": round(rss_mb(), 1),
                })
                print(
                    f"{index}/{len(records)} {record['id']} {task} {attack_name}: "
                    f"linf={linf:.5f} nll {clean_loss:.3f}->{adv_loss:.3f} "
                    f"neutral_correct={eval_rows[0]['correct']}",
                    flush=True,
                )

            results.append({
                "image_id": record["id"],
                "task": task,
                "color": record["color"],
                "shape": record["shape"],
                "attacks": attack_rows,
            })

    summary = summarize(results, conditions)
    payload = {
        "phase": "C",
        "pilot": True,
        "model": args.model,
        "device": device,
        "do_image_splitting": args.do_image_splitting,
        "epsilon": args.epsilon,
        "step_size": args.step_size,
        "pgd_steps": args.steps,
        "optimize_condition": args.optimize_condition,
        "n_images": len(records),
        "tasks": tasks,
        "conditions": conditions,
        "attacks": args.attacks,
        "wall_seconds": round(time.perf_counter() - t_all, 1),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)
    return payload


def summarize(results: list[dict[str, Any]], conditions: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for attack in sorted({row["attack"] for item in results for row in item["attacks"]}):
        by_cond: dict[str, dict[str, int]] = {
            condition: {"n": 0, "correct": 0, "invalid": 0} for condition in conditions
        }
        nll_up = 0
        nll_n = 0
        eps_ok = 0
        eps_n = 0
        for item in results:
            for row in item["attacks"]:
                if row["attack"] != attack:
                    continue
                nll_n += 1
                nll_up += int(row["nll_increased"])
                eps_n += 1
                eps_ok += int(row["within_epsilon"] and row["pixels_in_unit_interval"])
                for evaluation in row["evaluations"]:
                    bucket = by_cond[evaluation["condition"]]
                    bucket["n"] += 1
                    bucket["correct"] += int(evaluation["correct"])
                    bucket["invalid"] += int(evaluation["invalid"])
        summary[attack] = {
            "by_condition": by_cond,
            "nll_increased_rate": nll_up / nll_n if nll_n else None,
            "epsilon_bound_ok_rate": eps_ok / eps_n if eps_n else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/synthetic_v1/manifest.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="mps", choices=["auto", "mps", "cpu"])
    parser.add_argument("--task", default="color", choices=["color", "shape", "both"])
    parser.add_argument("--conditions", nargs="+", default=["neutral", "supportive", "misleading"])
    parser.add_argument("--optimize-condition", default="neutral",
                        choices=["neutral", "supportive", "misleading"])
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--step-size", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["clean", "noise", "fgsm", "pgd"],
        choices=["clean", "noise", "fgsm", "pgd"],
    )
    parser.add_argument(
        "--do-image-splitting",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output", type=Path, default=Path("results/phase_c_attack_pilot.json"))
    args = parser.parse_args()
    run_pilot(args)


if __name__ == "__main__":
    main()
