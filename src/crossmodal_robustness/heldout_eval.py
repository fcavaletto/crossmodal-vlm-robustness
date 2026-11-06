"""Phase E held-out evaluation entry point (frozen Checkpoint 1 stack).

Uses ``split=\"test\"`` only, validates exactly 32 images, pins model+processor
revision, and writes uniquely named outputs that cannot overwrite Phase D
development artifacts.

Hard safeguard: inference is refused unless ``--i-authorize-heldout-oneshot``
is passed. ``--validate-only`` (default-safe path) checks the split and prints
the frozen plan without loading the model or touching test images for inference.
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

from .attacks.vlm_loss import make_answer_nll_loss, tensor_to_pil
from .dataset import build_prompt, load_manifest
from .evaluate import load_model, select_device
from .freeze_pins import (
    EXPECTED_TEST_COUNT,
    FROZEN_DO_IMAGE_SPLITTING,
    FROZEN_EPSILONS,
    FROZEN_MODEL_ID,
    FROZEN_OPTIMIZE_CONDITION,
    FROZEN_PGD_STEPS,
    FROZEN_REVISION,
    FROZEN_SEED,
)
from .gradient_gate import freeze_model, load_image_01, mps_mem_mb, rss_mb
from .phase_d_eval import (
    craft_attack,
    evaluate_conditions,
    likelihood_snapshot,
    roundtrip_metrics,
)
from .seeds import attack_noise_seed

DEFAULT_HELDOUT_OUTPUT = Path("results/phase_e_heldout.json")
FORBIDDEN_OUTPUT_STEMS = {
    "phase_d_eval",
    "phase_d_full64",
    "phase_d_balanced16",
    "phase-d-full64",
    "phase-d-balanced16",
}


def load_heldout_records(manifest: Path) -> list[dict[str, Any]]:
    """Load and validate the sealed test split (exactly EXPECTED_TEST_COUNT)."""
    records = load_manifest(manifest, split="test")
    if len(records) != EXPECTED_TEST_COUNT:
        raise RuntimeError(
            f"Held-out split must contain exactly {EXPECTED_TEST_COUNT} images; "
            f"got {len(records)} from {manifest}"
        )
    wrong_split = [r["id"] for r in records if r.get("split") != "test"]
    if wrong_split:
        raise RuntimeError(f"Non-test records leaked into held-out load: {wrong_split}")
    return records


def assert_unique_heldout_output(path: Path) -> None:
    stem = path.stem
    if stem in FORBIDDEN_OUTPUT_STEMS or stem.startswith("phase_d_"):
        raise ValueError(
            f"Refusing output path that could overwrite development artifacts: {path}. "
            "Use a phase_e_heldout* name."
        )
    if "phase_e" not in stem and "heldout" not in stem:
        raise ValueError(
            f"Held-out output stem must contain 'phase_e' or 'heldout'; got {path}"
        )


def build_heldout_plan(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "phase": "E",
        "split": "test",
        "n_images": len(records),
        "image_ids": [r["id"] for r in records],
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "do_image_splitting": FROZEN_DO_IMAGE_SPLITTING,
        "optimize_condition": FROZEN_OPTIMIZE_CONDITION,
        "epsilons": list(args.epsilons),
        "pgd_steps": args.steps,
        "step_size_rule": "epsilon/4",
        "attacks": ["clean", *args.attacks],
        "tasks": ["color", "shape"] if args.task == "both" else [args.task],
        "noise_seed_scheme": "attack_noise_seed/md5_stable_v1",
        "output": str(args.output),
        "jsonl": str(args.output.with_suffix(".jsonl")),
        "authorize_heldout_oneshot": bool(args.i_authorize_heldout_oneshot),
        "validate_only": bool(args.validate_only),
    }


def run_heldout(args: argparse.Namespace) -> dict[str, Any]:
    import transformers

    assert_unique_heldout_output(args.output)
    records = load_heldout_records(args.manifest)
    plan = build_heldout_plan(records, args)

    if args.validate_only:
        plan["status"] = "validate_only_no_inference"
        print(json.dumps(plan, indent=2))
        print(
            "VALIDATE-ONLY: split OK; no model load; no held-out inference.",
            flush=True,
        )
        return plan

    if not args.i_authorize_heldout_oneshot:
        raise SystemExit(
            "Refusing held-out inference without --i-authorize-heldout-oneshot. "
            "Use --validate-only to check the sealed split and frozen plan."
        )

    if args.revision != FROZEN_REVISION:
        raise SystemExit(
            f"Held-out requires frozen revision {FROZEN_REVISION}; got {args.revision}"
        )
    if args.model != FROZEN_MODEL_ID:
        raise SystemExit(
            f"Held-out requires frozen model {FROZEN_MODEL_ID}; got {args.model}"
        )
    if tuple(float(x) for x in args.epsilons) != FROZEN_EPSILONS:
        raise SystemExit(
            f"Held-out requires frozen ε grid {FROZEN_EPSILONS}; got {args.epsilons}"
        )
    if args.steps != FROZEN_PGD_STEPS or args.seed != FROZEN_SEED:
        raise SystemExit(
            f"Held-out requires steps={FROZEN_PGD_STEPS} seed={FROZEN_SEED}; "
            f"got steps={args.steps} seed={args.seed}"
        )
    if FROZEN_DO_IMAGE_SPLITTING:
        raise SystemExit("Frozen stack requires do_image_splitting=False")

    device = select_device(args.device)
    tasks = plan["tasks"]
    attacks = [a for a in args.attacks if a != "clean"]
    epsilons = [float(x) for x in args.epsilons]

    print(
        f"Phase E held-out: {len(records)} test images tasks={tasks} "
        f"eps={epsilons} attacks={attacks} device={device} rev={args.revision[:12]}",
        flush=True,
    )

    processor, model = load_model(args.model, device, revision=args.revision)
    freeze_model(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output.with_suffix(".jsonl")
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
        raise SystemExit(
            f"Refusing to overwrite existing held-out JSONL {jsonl_path}. "
            "Pass --resume to continue, or choose a new --output stem."
        )

    meta = {
        **plan,
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "model_revision_verified": getattr(
            model.config, "_frozen_revision_verified", args.revision
        ),
        "processor_revision": getattr(processor, "_frozen_revision", args.revision),
        "status": "running",
    }
    meta_path = args.output.with_name(args.output.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    t_all = time.perf_counter()
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            path = Path(record["absolute_image"])
            with Image.open(path) as handle_img:
                clean_pil = handle_img.convert("RGB")
            clean_tensor = load_image_01(path).to(device)

            for task in tasks:
                attack_prompt = build_prompt(record, task, FROZEN_OPTIMIZE_CONDITION)
                loss_fn = make_answer_nll_loss(
                    processor,
                    model,
                    image_pil=clean_pil,
                    prompt=attack_prompt,
                    answer=record[task],
                    device=device,
                    do_image_splitting=FROZEN_DO_IMAGE_SPLITTING,
                )

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
                        "split": "test",
                        "task": task,
                        "color": record["color"],
                        "shape": record["shape"],
                        "attack": "clean",
                        "epsilon": 0.0,
                        "optimize_condition": FROZEN_OPTIMIZE_CONDITION,
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
                        seed = attack_noise_seed(
                            base_seed=args.seed,
                            image_index=index,
                            task=task,
                            attack_name=attack_name,
                            epsilon=epsilon,
                        )
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
                            "split": "test",
                            "task": task,
                            "color": record["color"],
                            "shape": record["shape"],
                            "attack": attack_name,
                            "epsilon": epsilon,
                            "optimize_condition": FROZEN_OPTIMIZE_CONDITION,
                            "clean_nll": clean_nll,
                            "adv_nll": adv_nll,
                            "nll_increased": adv_nll > clean_nll + 1e-6,
                            "likelihood": likes,
                            "bounds": bounds,
                            "attack_seconds": round(time.perf_counter() - t0, 3),
                            "evaluations": evals,
                            "noise_seed": seed if attack_name == "noise" else None,
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
    meta["status"] = "complete"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Done wall={meta['wall_seconds']}s wrote {jsonl_path}", flush=True)
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/synthetic_v1/manifest.jsonl"))
    parser.add_argument("--model", default=FROZEN_MODEL_ID)
    parser.add_argument("--revision", default=FROZEN_REVISION)
    parser.add_argument("--device", default="mps", choices=["auto", "mps", "cpu"])
    parser.add_argument("--task", default="both", choices=["color", "shape", "both"])
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=list(FROZEN_EPSILONS),
    )
    parser.add_argument("--steps", type=int, default=FROZEN_PGD_STEPS)
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["noise", "fgsm", "pgd"],
        choices=["noise", "fgsm", "pgd"],
    )
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_HELDOUT_OUTPUT,
        help="Must be a phase_e / heldout stem; never a phase_d_* path",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check sealed test split + frozen plan; no model load / inference",
    )
    parser.add_argument(
        "--i-authorize-heldout-oneshot",
        action="store_true",
        help="Required to run actual held-out inference (one-shot authorization)",
    )
    args = parser.parse_args(argv)
    run_heldout(args)


if __name__ == "__main__":
    main()
