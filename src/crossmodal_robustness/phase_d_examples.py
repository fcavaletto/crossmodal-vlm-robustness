"""Render a few clean vs PGD difference maps for Phase D reporting."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .attacks import pgd_linf
from .attacks.vlm_loss import make_answer_nll_loss, tensor_to_pil
from .dataset import build_prompt, load_manifest
from .evaluate import DEFAULT_MODEL, load_model, select_device
from .gradient_gate import freeze_model, load_image_01


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--image-ids", nargs="+", default=["item_000", "item_030"])
    parser.add_argument("--task", default="color", choices=["color", "shape"])
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures/phase-d/examples"))
    args = parser.parse_args()

    device = select_device(args.device)
    records = {r["id"]: r for r in load_manifest("data/synthetic_v1/manifest.jsonl", split="development")}
    processor, model = load_model(DEFAULT_MODEL, device)
    freeze_model(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_id in args.image_ids:
        record = records[image_id]
        path = Path(record["absolute_image"])
        with Image.open(path) as handle:
            clean_pil = handle.convert("RGB")
        clean = load_image_01(path).to(device)
        prompt = build_prompt(record, args.task, "neutral")
        loss_fn = make_answer_nll_loss(
            processor, model, image_pil=clean_pil, prompt=prompt,
            answer=record[args.task], device=device, do_image_splitting=False,
        )
        adv = pgd_linf(
            clean, loss_fn, epsilon=args.epsilon, step_size=args.epsilon / 4, steps=args.steps
        )
        clean_np = clean.detach().cpu().permute(1, 2, 0).numpy()
        adv_np = adv.detach().cpu().permute(1, 2, 0).numpy()
        diff = (adv_np - clean_np)
        ampl = np.clip(0.5 + 5.0 * diff, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(clean_np)
        axes[0].set_title("clean")
        axes[1].imshow(np.clip(adv_np, 0, 1))
        axes[1].set_title(f"PGD ε={args.epsilon*255:.0f}/255")
        axes[2].imshow(ampl)
        axes[2].set_title("5× signed diff")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{image_id} {args.task}={record[args.task]} (opt@neutral)", fontsize=10)
        fig.tight_layout()
        out = args.output_dir / f"{image_id}_{args.task}_eps{int(round(args.epsilon*255))}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        # also save raw adv png
        tensor_to_pil(adv.cpu()).save(args.output_dir / f"{image_id}_{args.task}_adv.png")
        print("wrote", out)


if __name__ == "__main__":
    main()
