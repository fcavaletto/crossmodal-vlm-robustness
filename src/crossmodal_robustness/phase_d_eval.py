"""Phase D development evaluation: ε-grid attacks under neutral optimization.

Optimize once per (image, task, ε, attack) with a neutral prompt; evaluate the
same perturbation under neutral / supportive / misleading. No held-out test.
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
from .dataset import COLORS, SHAPES, build_prompt, load_manifest, parse_answer
from .evaluate import DEFAULT_MODEL, load_model, run_inference, select_device
from .gradient_gate import freeze_model, load_image_01, mps_mem_mb, rss_mb
from .loss_validation import score_answer


def select_balanced_development(records: list[dict[str, Any]], n_images: int) -> list[dict[str, Any]]:
    """Predefined balanced subset: round-robin over color×shape, lowest ids first."""
    ordered = sorted(records, key=lambda r: r["id"])
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {
        (color, shape): [] for color in COLORS for shape in SHAPES
    }
    for record in ordered:
        buckets[(record["color"], record["shape"])].append(record)
    selected: list[dict[str, Any]] = []
    combos = [(color, shape) for color in COLORS for shape in SHAPES]
    round_index = 0
    while len(selected) < n_images:
        progress = False
        for combo in combos:
            if len(selected) >= n_images:
                break
            bucket = buckets[combo]
            if round_index < len(bucket):
                selected.append(bucket[round_index])
                progress = True
        if not progress:
            break
        round_index += 1
    return selected


def pil_to_01(image: Image.Image) -> torch.Tensor:
    import numpy as np

    array = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def roundtrip_metrics(clean: torch.Tensor, adv: torch.Tensor, epsilon: float) -> dict[str, Any]:
    """L∞ / bounds before and after the uint8 PIL round-trip used at eval time."""
    clean_cpu = clean.detach().float().cpu()
    adv_cpu = adv.detach().float().cpu()
    linf_before = float((adv_cpu - clean_cpu).abs().max())
    in_bounds_before = bool(((adv_cpu >= 0) & (adv_cpu <= 1)).all().item())
    pil = tensor_to_pil(adv_cpu)
    reconstructed = pil_to_01(pil)
    linf_after = float((reconstructed - clean_cpu).abs().max())
    in_bounds_after = bool(((reconstructed >= 0) & (reconstructed <= 1)).all().item())
    # Quantization can push L∞ slightly above ε; report both.
    return {
        "linf_before_roundtrip": linf_before,
        "linf_after_roundtrip": linf_after,
        "within_epsilon_before": linf_before <= epsilon + 1e-5,
        "within_epsilon_after": linf_after <= epsilon + 1.5 / 255 + 1e-5,  # allow 1-level quant
        "pixels_in_unit_interval_before": in_bounds_before,
        "pixels_in_unit_interval_after": in_bounds_after,
        "roundtrip_max_abs_vs_float_adv": float((reconstructed - adv_cpu).abs().max()),
    }


def alternative_labels(task: str) -> list[str]:
    return list(COLORS if task == "color" else SHAPES)


def likelihood_snapshot(
    processor,
    model,
    *,
    image_pil: Image.Image,
    image_01: torch.Tensor,
    record: dict[str, Any],
    task: str,
    device: str,
) -> dict[str, Any]:
    """Correct-answer NLL and margin vs best wrong label under neutral context."""
    prompt = build_prompt(record, task, "neutral")
    correct = record[task]
    scores = []
    for answer in alternative_labels(task):
        row = score_answer(
            processor,
            model,
            image_pil=image_pil,
            image_01=image_01.cpu(),
            prompt=prompt,
            answer=answer,
            device=device,
            do_image_splitting=False,
            pixel_source="torch",
        )
        scores.append({"answer": answer, "nll": row["nll"]})
    correct_nll = next(s["nll"] for s in scores if s["answer"] == correct)
    wrong = [s for s in scores if s["answer"] != correct]
    best_wrong = min(wrong, key=lambda s: s["nll"])
    return {
        "correct_nll": correct_nll,
        "best_wrong_answer": best_wrong["answer"],
        "best_wrong_nll": best_wrong["nll"],
        "margin_best_wrong_minus_correct": best_wrong["nll"] - correct_nll,
        "ranking": sorted(scores, key=lambda s: s["nll"]),
    }


def evaluate_conditions(
    processor,
    model,
    *,
    image_pil: Image.Image,
    record: dict[str, Any],
    task: str,
    path_hint: Path,
) -> list[dict[str, Any]]:
    rows = []
    for condition in ("neutral", "supportive", "misleading"):
        prompt = build_prompt(record, task, condition)
        raw = run_inference(
            processor,
            model,
            path_hint,
            prompt,
            12,
            do_image_splitting=False,
            image=image_pil,
        )
        parsed = parse_answer(raw, task)
        rows.append({
            "condition": condition,
            "raw_answer": raw,
            "parsed_answer": parsed,
            "expected": record[task],
            "correct": parsed == record[task],
            "invalid": parsed is None,
        })
    return rows


def craft_attack(
    name: str,
    clean_tensor: torch.Tensor,
    loss_fn,
    *,
    epsilon: float,
    steps: int,
    seed: int,
) -> torch.Tensor:
    if name == "noise":
        generator = torch.Generator().manual_seed(seed)
        return random_linf_noise(clean_tensor.cpu(), epsilon=epsilon, generator=generator).to(
            clean_tensor.device
        )
    if name == "fgsm":
        return fgsm_linf(clean_tensor, loss_fn, epsilon=epsilon)
    if name == "pgd":
        return pgd_linf(
            clean_tensor,
            loss_fn,
            epsilon=epsilon,
            step_size=epsilon / 4,
            steps=steps,
            random_start=False,
        )
    raise ValueError(name)


def run_phase_d(args: argparse.Namespace) -> dict[str, Any]:
    import transformers

    device = select_device(args.device)
    all_records = load_manifest(args.manifest, split="development")
    if args.subset == "balanced":
        records = select_balanced_development(all_records, args.n_images)
    elif args.subset == "all":
        records = all_records
    else:
        raise ValueError(args.subset)
    epsilons = [float(x) for x in args.epsilons]
    tasks = ["color", "shape"] if args.task == "both" else [args.task]
    attacks = [a for a in args.attacks if a != "clean"]

    print(
        f"Phase D: {len(records)} images subset={args.subset} tasks={tasks} "
        f"eps={epsilons} attacks={attacks} device={device}",
        flush=True,
    )
    print("selected_ids", [r["id"] for r in records], flush=True)

    processor, model = load_model(args.model, device)
    freeze_model(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output.with_suffix(".jsonl")
    # Resume support: skip completed (image_id, task, epsilon, attack) keys.
    done: set[tuple[str, str, str, str]] = set()
    if jsonl_path.exists() and args.resume:
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                done.add((row["image_id"], row["task"], f"{row['epsilon']:.8f}", row["attack"]))
        print(f"Resuming; {len(done)} rows already on disk", flush=True)
    elif jsonl_path.exists() and not args.resume:
        jsonl_path.unlink()

    meta = {
        "phase": "D",
        "model": args.model,
        "device": device,
        "do_image_splitting": False,
        "optimize_condition": "neutral",
        "subset": args.subset,
        "n_images": len(records),
        "image_ids": [r["id"] for r in records],
        "tasks": tasks,
        "epsilons": epsilons,
        "pgd_steps": args.steps,
        "step_size_rule": "epsilon/4",
        "attacks": ["clean", *attacks],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "phase_b_tokenization_note": (
            "Phase B scored bare BPE ids (e.g. 'red'); Phase C/D use leading-space "
            "tokens (e.g. 'Ġred') matching greedy decoding."
        ),
    }
    args.output.with_name(args.output.stem + "_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    t_all = time.perf_counter()
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            path = Path(record["absolute_image"])
            with Image.open(path) as handle_img:
                clean_pil = handle_img.convert("RGB")
            clean_tensor = load_image_01(path).to(device)

            for task in tasks:
                attack_prompt = build_prompt(record, task, "neutral")
                loss_fn = make_answer_nll_loss(
                    processor,
                    model,
                    image_pil=clean_pil,
                    prompt=attack_prompt,
                    answer=record[task],
                    device=device,
                    do_image_splitting=False,
                )

                # Clean row once (epsilon 0).
                clean_key = (record["id"], task, f"{0.0:.8f}", "clean")
                if clean_key not in done:
                    t0 = time.perf_counter()
                    evals = evaluate_conditions(
                        processor, model, image_pil=clean_pil, record=record, task=task, path_hint=path
                    )
                    likes = likelihood_snapshot(
                        processor, model,
                        image_pil=clean_pil, image_01=clean_tensor,
                        record=record, task=task, device=device,
                    )
                    with torch.no_grad():
                        clean_nll = float(loss_fn(clean_tensor))
                    row = {
                        "image_id": record["id"],
                        "task": task,
                        "color": record["color"],
                        "shape": record["shape"],
                        "attack": "clean",
                        "epsilon": 0.0,
                        "optimize_condition": "neutral",
                        "clean_nll": clean_nll,
                        "adv_nll": clean_nll,
                        "nll_increased": False,
                        "likelihood": likes,
                        "bounds": roundtrip_metrics(clean_tensor, clean_tensor, 0.0),
                        "attack_seconds": round(time.perf_counter() - t0, 3),
                        "evaluations": evals,
                        "mps": mps_mem_mb(),
                        "rss_mb": round(rss_mb(), 1),
                    }
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    done.add(clean_key)
                    print(
                        f"{index}/{len(records)} {record['id']} {task} clean "
                        f"neutral_correct={evals[0]['correct']}",
                        flush=True,
                    )

                for epsilon in epsilons:
                    for attack_name in attacks:
                        key = (record["id"], task, f"{epsilon:.8f}", attack_name)
                        if key in done:
                            continue
                        t0 = time.perf_counter()
                        seed = args.seed + index * 131 + hash((task, attack_name, f"{epsilon:.8f}")) % 10_007
                        adv = craft_attack(
                            attack_name,
                            clean_tensor,
                            loss_fn,
                            epsilon=epsilon,
                            steps=args.steps,
                            seed=seed,
                        )
                        bounds = roundtrip_metrics(clean_tensor, adv, epsilon)
                        with torch.no_grad():
                            adv_nll = float(loss_fn(adv))
                            clean_nll = float(loss_fn(clean_tensor))
                        adv_pil = tensor_to_pil(adv.cpu())
                        # Likelihood on attacked tensor (torch path).
                        likes = likelihood_snapshot(
                            processor, model,
                            image_pil=adv_pil, image_01=adv.detach(),
                            record=record, task=task, device=device,
                        )
                        evals = evaluate_conditions(
                            processor, model, image_pil=adv_pil, record=record, task=task, path_hint=path
                        )
                        row = {
                            "image_id": record["id"],
                            "task": task,
                            "color": record["color"],
                            "shape": record["shape"],
                            "attack": attack_name,
                            "epsilon": epsilon,
                            "optimize_condition": "neutral",
                            "clean_nll": clean_nll,
                            "adv_nll": adv_nll,
                            "nll_increased": adv_nll > clean_nll + 1e-6,
                            "likelihood": likes,
                            "bounds": bounds,
                            "attack_seconds": round(time.perf_counter() - t0, 3),
                            "evaluations": evals,
                            "mps": mps_mem_mb(),
                            "rss_mb": round(rss_mb(), 1),
                        }
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                        handle.flush()
                        done.add(key)
                        print(
                            f"{index}/{len(records)} {record['id']} {task} {attack_name} "
                            f"eps={epsilon:.5f} linf_after={bounds['linf_after_roundtrip']:.5f} "
                            f"nll {clean_nll:.3f}->{adv_nll:.3f} "
                            f"neutral_correct={evals[0]['correct']}",
                            flush=True,
                        )

    meta["wall_seconds"] = round(time.perf_counter() - t_all, 1)
    meta["jsonl"] = str(jsonl_path)
    args.output.with_name(args.output.stem + "_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Done wall={meta['wall_seconds']}s wrote {jsonl_path}", flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/synthetic_v1/manifest.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="mps", choices=["auto", "mps", "cpu"])
    parser.add_argument("--task", default="both", choices=["color", "shape", "both"])
    parser.add_argument("--subset", default="balanced", choices=["balanced", "all"])
    parser.add_argument("--n-images", type=int, default=16, help="Used with --subset balanced")
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=[2 / 255, 4 / 255, 8 / 255],
    )
    parser.add_argument("--steps", type=int, default=7, help="PGD iterations; α=ε/4")
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["noise", "fgsm", "pgd"],
        choices=["noise", "fgsm", "pgd"],
    )
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/phase_d_eval.json"))
    args = parser.parse_args()
    run_phase_d(args)


if __name__ == "__main__":
    main()
