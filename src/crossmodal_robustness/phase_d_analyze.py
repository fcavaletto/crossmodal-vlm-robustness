"""Aggregate Phase D JSONL into metrics, DiD bootstrap CIs, and plots."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = ("neutral", "supportive", "misleading")


def load_rows(jsonl: Path) -> list[dict[str, Any]]:
    rows = []
    with jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def eval_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {e["condition"]: e for e in row["evaluations"]}


def accuracy_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for row in rows:
        key = f"{row['attack']}|{row['epsilon']:.8f}|{row['task']}"
        bucket = table.setdefault(key, {c: {"n": 0, "correct": 0, "invalid": 0} for c in CONDITIONS})
        for condition, evaluation in eval_map(row).items():
            bucket[condition]["n"] += 1
            bucket[condition]["correct"] += int(evaluation["correct"])
            bucket[condition]["invalid"] += int(evaluation["invalid"])
    # attach rates
    out = {}
    for key, by_c in table.items():
        out[key] = {
            c: {
                **stats,
                "accuracy": stats["correct"] / stats["n"] if stats["n"] else None,
                "invalid_rate": stats["invalid"] / stats["n"] if stats["n"] else None,
            }
            for c, stats in by_c.items()
        }
    return out


def conditional_asr(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """ASR among images correct under clean for the same task/condition."""
    cleans = [r for r in rows if r["attack"] == "clean"]
    clean_ok = {}
    for row in cleans:
        for condition, evaluation in eval_map(row).items():
            clean_ok[(row["image_id"], row["task"], condition)] = bool(evaluation["correct"])

    out: dict[str, Any] = {}
    for row in rows:
        if row["attack"] == "clean":
            continue
        for condition, evaluation in eval_map(row).items():
            key = f"{row['attack']}|{row['epsilon']:.8f}|{row['task']}|{condition}"
            base = clean_ok.get((row["image_id"], row["task"], condition), False)
            bucket = out.setdefault(key, {"baseline_correct_n": 0, "flipped": 0})
            if base:
                bucket["baseline_correct_n"] += 1
                bucket["flipped"] += int(not evaluation["correct"])
    for key, bucket in out.items():
        n = bucket["baseline_correct_n"]
        bucket["asr"] = bucket["flipped"] / n if n else None
    return out


def bound_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attacked = [r for r in rows if r["attack"] != "clean"]
    if not attacked:
        return {}
    before_ok = sum(int(r["bounds"]["within_epsilon_before"] and r["bounds"]["pixels_in_unit_interval_before"]) for r in attacked)
    after_ok = sum(int(r["bounds"]["within_epsilon_after"] and r["bounds"]["pixels_in_unit_interval_after"]) for r in attacked)
    return {
        "n": len(attacked),
        "before_roundtrip_ok_rate": before_ok / len(attacked),
        "after_roundtrip_ok_rate": after_ok / len(attacked),
        "max_linf_before": max(r["bounds"]["linf_before_roundtrip"] for r in attacked),
        "max_linf_after": max(r["bounds"]["linf_after_roundtrip"] for r in attacked),
        "mean_roundtrip_quant_error": float(np.mean([r["bounds"]["roundtrip_max_abs_vs_float_adv"] for r in attacked])),
    }


def likelihood_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cleans = {(r["image_id"], r["task"]): r for r in rows if r["attack"] == "clean"}
    summary = defaultdict(list)
    for row in rows:
        if row["attack"] == "clean":
            continue
        clean = cleans[(row["image_id"], row["task"])]
        summary[f"{row['attack']}|{row['epsilon']:.8f}|{row['task']}"].append({
            "delta_correct_nll": row["likelihood"]["correct_nll"] - clean["likelihood"]["correct_nll"],
            "delta_margin": row["likelihood"]["margin_best_wrong_minus_correct"]
            - clean["likelihood"]["margin_best_wrong_minus_correct"],
            "nll_increased": row["nll_increased"],
        })
    out = {}
    for key, items in summary.items():
        out[key] = {
            "n": len(items),
            "mean_delta_correct_nll": float(np.mean([i["delta_correct_nll"] for i in items])),
            "mean_delta_margin": float(np.mean([i["delta_margin"] for i in items])),
            "nll_increased_rate": float(np.mean([i["nll_increased"] for i in items])),
        }
    return out


def image_level_errors(rows: list[dict[str, Any]], attack: str, epsilon: float, task: str) -> dict[str, dict[str, int]]:
    """Per-image error indicators for clean and attacked under each condition."""
    cleans = {
        r["image_id"]: eval_map(r)
        for r in rows
        if r["attack"] == "clean" and r["task"] == task
    }
    attacked = {
        r["image_id"]: eval_map(r)
        for r in rows
        if r["attack"] == attack and r["task"] == task and abs(r["epsilon"] - epsilon) < 1e-12
    }
    images = sorted(set(cleans) & set(attacked))
    out = {}
    for image_id in images:
        out[image_id] = {}
        for condition in CONDITIONS:
            e_clean = int(not cleans[image_id][condition]["correct"])
            e_adv = int(not attacked[image_id][condition]["correct"])
            out[image_id][condition] = {
                "error_clean": e_clean,
                "error_adv": e_adv,
                "delta_error": e_adv - e_clean,
            }
    return out


def did_for_image(per_image: dict[str, dict[str, int]], primary: str = "misleading", control: str = "supportive") -> float:
    """Mean image-level DiD: (Δerr_primary) - (Δerr_control)."""
    values = []
    for stats in per_image.values():
        values.append(stats[primary]["delta_error"] - stats[control]["delta_error"])
    return float(np.mean(values)) if values else float("nan")


def bootstrap_did(
    per_image: dict[str, dict[str, int]],
    *,
    primary: str,
    control: str,
    n_boot: int = 2000,
    seed: int = 20261105,
) -> dict[str, Any]:
    image_ids = sorted(per_image)
    if not image_ids:
        return {"n_images": 0, "estimate": None}
    rng = np.random.default_rng(seed)
    point = did_for_image(per_image, primary, control)
    samples = []
    n = len(image_ids)
    for _ in range(n_boot):
        draw = rng.choice(image_ids, size=n, replace=True)
        resampled = {i: per_image[i] for i in draw}
        # With replacement, duplicate keys collapse — use list of deltas instead.
        deltas = [
            per_image[i][primary]["delta_error"] - per_image[i][control]["delta_error"]
            for i in draw
        ]
        samples.append(float(np.mean(deltas)))
        _ = resampled  # kept for clarity
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "n_images": n,
        "primary": primary,
        "control": control,
        "estimate": point,
        "ci95": [float(lo), float(hi)],
        "n_boot": n_boot,
    }


def additional_error_rates(per_image: dict[str, dict[str, int]]) -> dict[str, Any]:
    out = {}
    for condition in CONDITIONS:
        deltas = [stats[condition]["delta_error"] for stats in per_image.values()]
        clean_err = [stats[condition]["error_clean"] for stats in per_image.values()]
        adv_err = [stats[condition]["error_adv"] for stats in per_image.values()]
        out[condition] = {
            "clean_error_rate": float(np.mean(clean_err)),
            "adv_error_rate": float(np.mean(adv_err)),
            "mean_additional_error": float(np.mean(deltas)),
            "floor_note": (
                "High clean error leaves little room for additional attack-caused errors; "
                "do not interpret low robust accuracy alone as conflict amplification."
                if float(np.mean(clean_err)) >= 0.5
                else None
            ),
        }
    return out


def build_metrics(rows: list[dict[str, Any]], n_boot: int = 2000) -> dict[str, Any]:
    epsilons = sorted({r["epsilon"] for r in rows if r["attack"] != "clean"})
    attacks = sorted({r["attack"] for r in rows if r["attack"] != "clean"})
    tasks = sorted({r["task"] for r in rows})

    did = {}
    add_err = {}
    for attack in attacks:
        for epsilon in epsilons:
            for task in tasks:
                per_image = image_level_errors(rows, attack, epsilon, task)
                key = f"{attack}|{epsilon:.8f}|{task}"
                add_err[key] = additional_error_rates(per_image)
                did[f"{key}|mis_vs_sup"] = bootstrap_did(
                    per_image, primary="misleading", control="supportive", n_boot=n_boot
                )
                did[f"{key}|neu_vs_sup"] = bootstrap_did(
                    per_image, primary="neutral", control="supportive", n_boot=n_boot
                )

    floor = _floor_diagnostics_from_rows(rows)
    return {
        "n_rows": len(rows),
        "n_images": len({r["image_id"] for r in rows}),
        "accuracy": accuracy_table(rows),
        "conditional_asr": conditional_asr(rows),
        "bounds": bound_summary(rows),
        "likelihood_deltas": likelihood_deltas(rows),
        "additional_errors": add_err,
        "did_bootstrap": did,
        "floor_effect_color_misleading": floor,
    }


def _floor_diagnostics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute misleading-color clean accuracy from *this* result set only."""
    cleans = [
        r for r in rows
        if r["attack"] == "clean" and r["task"] == "color"
    ]
    if not cleans:
        return {
            "clean_accuracy_misleading_color": None,
            "n_images": 0,
            "note": "No clean color rows available in this result set.",
        }
    n = 0
    correct = 0
    for row in cleans:
        for evaluation in row["evaluations"]:
            if evaluation["condition"] != "misleading":
                continue
            n += 1
            correct += int(evaluation["correct"])
    accuracy = correct / n if n else None
    note = None
    if accuracy is not None and accuracy <= 0.25:
        note = (
            "Low clean misleading-color accuracy in this split leaves little room "
            "for additional attack-caused errors; do not interpret low robust "
            "accuracy under misleading as conflict amplification. Use "
            "additional-error / DiD on paired images instead."
        )
    return {
        "clean_accuracy_misleading_color": accuracy,
        "correct": correct,
        "n_images": n,
        "note": note,
    }


def make_plots(metrics: dict[str, Any], rows: list[dict[str, Any]], figure_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    epsilons = sorted({r["epsilon"] for r in rows if r["attack"] != "clean"})
    attacks = ["noise", "fgsm", "pgd"]

    # 1) Robust accuracy vs epsilon for each task/condition (PGD focus + controls)
    for task in sorted({r["task"] for r in rows}):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
        for ax, condition in zip(axes, CONDITIONS):
            for attack, style in zip(attacks, ["o--", "s--", "D-"]):
                xs, ys = [], []
                # clean horizontal reference at eps=0 from accuracy table
                clean_key = f"clean|{0.0:.8f}|{task}"
                # clean rows use epsilon 0.0
                clean_acc = None
                for key, by_c in metrics["accuracy"].items():
                    if key.startswith(f"clean|") and key.endswith(f"|{task}"):
                        clean_acc = by_c[condition]["accuracy"]
                for epsilon in epsilons:
                    key = f"{attack}|{epsilon:.8f}|{task}"
                    if key in metrics["accuracy"]:
                        xs.append(epsilon * 255)
                        ys.append(metrics["accuracy"][key][condition]["accuracy"])
                ax.plot(xs, ys, style, label=attack)
                if clean_acc is not None:
                    ax.axhline(clean_acc, color="gray", lw=1, alpha=0.7)
            ax.set_title(f"{task} / {condition}")
            ax.set_xlabel("ε (×1/255)")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("Accuracy")
        axes[-1].legend(frameon=False)
        fig.suptitle("Robust accuracy vs perturbation budget (no-split, opt@neutral)", fontsize=11)
        fig.tight_layout()
        path = figure_dir / f"accuracy_vs_eps_{task}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path.resolve()))

    # 2) Additional error (adv - clean) vs epsilon for PGD
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, task in zip(axes, ["color", "shape"]):
        for condition, marker in zip(CONDITIONS, ["o", "s", "D"]):
            xs, ys = [], []
            for epsilon in epsilons:
                key = f"pgd|{epsilon:.8f}|{task}"
                if key in metrics["additional_errors"]:
                    xs.append(epsilon * 255)
                    ys.append(metrics["additional_errors"][key][condition]["mean_additional_error"])
            ax.plot(xs, ys, marker + "-", label=condition)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"PGD additional error — {task}")
        ax.set_xlabel("ε (×1/255)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Mean (error_adv − error_clean)")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "pgd_additional_error.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(str(path.resolve()))

    # 3) DiD estimates with CI for PGD mis-vs-sup and neu-vs-sup
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, task in zip(axes, ["color", "shape"]):
        for contrast, color in [("mis_vs_sup", "#b33"), ("neu_vs_sup", "#36a")]:
            xs, ys, ylo, yhi = [], [], [], []
            for epsilon in epsilons:
                key = f"pgd|{epsilon:.8f}|{task}|{contrast}"
                stat = metrics["did_bootstrap"].get(key)
                if not stat or stat["estimate"] is None:
                    continue
                xs.append(epsilon * 255)
                ys.append(stat["estimate"])
                ylo.append(stat["ci95"][0])
                yhi.append(stat["ci95"][1])
            if xs:
                ax.errorbar(
                    xs, ys,
                    yerr=[np.array(ys) - np.array(ylo), np.array(yhi) - np.array(ys)],
                    fmt="o-", capsize=3, label=contrast, color=color,
                )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"PGD DiD — {task}")
        ax.set_xlabel("ε (×1/255)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("DiD estimate (image-level)")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "pgd_did_bootstrap.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(str(path.resolve()))

    # 4) Example difference map if available later — skip if no saved images
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/phase_d_metrics.json"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures/phase-d"))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.jsonl)
    metrics = build_metrics(rows, n_boot=args.n_boot)
    figure_paths = []
    if not args.skip_plots:
        figure_paths = make_plots(metrics, rows, args.figure_dir)
    metrics["figures"] = figure_paths
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "n_rows": metrics["n_rows"],
        "n_images": metrics["n_images"],
        "bounds": metrics["bounds"],
        "figures": figure_paths,
        "did_pgd_8_color_mis_vs_sup": metrics["did_bootstrap"].get("pgd|0.03137255|color|mis_vs_sup")
        or metrics["did_bootstrap"].get(f"pgd|{8/255:.8f}|color|mis_vs_sup"),
    }, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
