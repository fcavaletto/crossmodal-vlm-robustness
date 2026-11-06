"""Offline verification of published Phase D/E statistics from compact tables.

Rebuilds clean accuracy, conditional ASR, additional error, and DiD (+ bootstrap)
from committed ``reports/artifacts/*_predictions_compact.jsonl`` files and compares
them to ``reports/phase-*-metrics.json``. No model inference.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .phase_d_analyze import bootstrap_did

CONDITIONS = ("neutral", "supportive", "misleading")
EPSILONS = (2 / 255, 4 / 255, 8 / 255)
ATTACKS = ("noise", "fgsm", "pgd")


def load_compact(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _eps_key(epsilon: float) -> str:
    return f"{epsilon:.8f}"


def index_compact(rows: list[dict[str, Any]]) -> dict[tuple, dict[str, dict[str, Any]]]:
    """Map (image_id, task, attack, eps_key) -> condition -> row."""
    out: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["image_id"], row["task"], row["attack"], _eps_key(float(row["epsilon"])))
        out[key][row["condition"]] = row
    return out


def clean_accuracy(index: dict, task: str) -> dict[str, dict[str, Any]]:
    keys = [k for k in index if k[1] == task and k[2] == "clean"]
    stats = {c: {"n": 0, "correct": 0, "invalid": 0} for c in CONDITIONS}
    for key in keys:
        for condition, row in index[key].items():
            stats[condition]["n"] += 1
            stats[condition]["correct"] += int(row["correct"])
            stats[condition]["invalid"] += int(row["invalid"])
    out = {}
    for condition, bucket in stats.items():
        n = bucket["n"]
        out[condition] = {
            **bucket,
            "accuracy": bucket["correct"] / n if n else None,
            "invalid_rate": bucket["invalid"] / n if n else None,
        }
    return out


def image_level_errors(
    index: dict, *, attack: str, epsilon: float, task: str
) -> dict[str, dict[str, dict[str, int]]]:
    clean_keys = {
        k[0]: index[k]
        for k in index
        if k[1] == task and k[2] == "clean"
    }
    adv_keys = {
        k[0]: index[k]
        for k in index
        if k[1] == task and k[2] == attack and k[3] == _eps_key(epsilon)
    }
    out: dict[str, dict[str, dict[str, int]]] = {}
    for image_id in sorted(set(clean_keys) & set(adv_keys)):
        out[image_id] = {}
        for condition in CONDITIONS:
            e_clean = int(not clean_keys[image_id][condition]["correct"])
            e_adv = int(not adv_keys[image_id][condition]["correct"])
            out[image_id][condition] = {
                "error_clean": e_clean,
                "error_adv": e_adv,
                "delta_error": e_adv - e_clean,
            }
    return out


def conditional_asr(
    index: dict, *, attack: str, epsilon: float, task: str, condition: str
) -> dict[str, Any]:
    per = image_level_errors(index, attack=attack, epsilon=epsilon, task=task)
    baseline = 0
    flipped = 0
    for stats in per.values():
        if stats[condition]["error_clean"] == 0:
            baseline += 1
            flipped += int(stats[condition]["error_adv"] == 1)
    return {
        "baseline_correct_n": baseline,
        "flipped": flipped,
        "asr": flipped / baseline if baseline else None,
    }


def reconstruct_core(index: dict, *, n_boot: int = 2000, seed: int = 20261105) -> dict[str, Any]:
    accuracy = {}
    for task in ("color", "shape"):
        accuracy[f"clean|{0.0:.8f}|{task}"] = clean_accuracy(index, task)

    did = {}
    add_err = {}
    asr = {}
    for attack in ATTACKS:
        for epsilon in EPSILONS:
            for task in ("color", "shape"):
                per = image_level_errors(index, attack=attack, epsilon=epsilon, task=task)
                key = f"{attack}|{_eps_key(epsilon)}|{task}"
                # additional error
                add_err[key] = {}
                for condition in CONDITIONS:
                    deltas = [s[condition]["delta_error"] for s in per.values()]
                    add_err[key][condition] = {
                        "mean_additional_error": float(np.mean(deltas)) if deltas else None,
                    }
                did[f"{key}|mis_vs_sup"] = bootstrap_did(
                    per, primary="misleading", control="supportive", n_boot=n_boot, seed=seed
                )
                did[f"{key}|neu_vs_sup"] = bootstrap_did(
                    per, primary="neutral", control="supportive", n_boot=n_boot, seed=seed
                )
                for condition in CONDITIONS:
                    asr[f"{key}|{condition}"] = conditional_asr(
                        index, attack=attack, epsilon=epsilon, task=task, condition=condition
                    )
                # attacked accuracy table
                adv_acc = {c: {"n": 0, "correct": 0, "invalid": 0} for c in CONDITIONS}
                for key_row, by_c in index.items():
                    if key_row[1] != task or key_row[2] != attack or key_row[3] != _eps_key(epsilon):
                        continue
                    for condition, row in by_c.items():
                        adv_acc[condition]["n"] += 1
                        adv_acc[condition]["correct"] += int(row["correct"])
                        adv_acc[condition]["invalid"] += int(row["invalid"])
                accuracy[key] = {
                    c: {
                        **stats,
                        "accuracy": stats["correct"] / stats["n"] if stats["n"] else None,
                        "invalid_rate": stats["invalid"] / stats["n"] if stats["n"] else None,
                    }
                    for c, stats in adv_acc.items()
                }

    n_images = len({k[0] for k in index})
    return {
        "n_images": n_images,
        "n_prediction_rows": sum(len(v) for v in index.values()),
        "accuracy": accuracy,
        "conditional_asr": asr,
        "additional_errors": add_err,
        "did_bootstrap": did,
    }


def _approx_equal(a: float | None, b: float | None, tol: float = 1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def compare_to_published(
    reconstructed: dict[str, Any],
    published: dict[str, Any],
    *,
    split_label: str,
    ci_tol: float = 1e-9,
) -> list[str]:
    """Return a list of mismatch strings (empty => pass)."""
    mismatches: list[str] = []

    pub_acc = published["accuracy"]
    for key, by_c in reconstructed["accuracy"].items():
        if key not in pub_acc:
            mismatches.append(f"{split_label}: missing published accuracy key {key}")
            continue
        for condition in CONDITIONS:
            for field in ("n", "correct", "invalid", "accuracy", "invalid_rate"):
                got = by_c[condition][field]
                want = pub_acc[key][condition][field]
                if field in ("n", "correct", "invalid"):
                    if got != want:
                        mismatches.append(
                            f"{split_label} accuracy {key}/{condition}.{field}: {got} != {want}"
                        )
                elif not _approx_equal(got, want):
                    mismatches.append(
                        f"{split_label} accuracy {key}/{condition}.{field}: {got} != {want}"
                    )

    for key, got_stat in reconstructed["did_bootstrap"].items():
        want_stat = published["did_bootstrap"].get(key)
        if want_stat is None:
            mismatches.append(f"{split_label}: missing published DiD key {key}")
            continue
        if got_stat["n_images"] != want_stat["n_images"]:
            mismatches.append(
                f"{split_label} DiD {key} n_images: {got_stat['n_images']} != {want_stat['n_images']}"
            )
        if not _approx_equal(got_stat["estimate"], want_stat["estimate"]):
            mismatches.append(
                f"{split_label} DiD {key} estimate: {got_stat['estimate']} != {want_stat['estimate']}"
            )
        for i, (g, w) in enumerate(zip(got_stat["ci95"], want_stat["ci95"])):
            if not _approx_equal(g, w, tol=ci_tol):
                mismatches.append(
                    f"{split_label} DiD {key} ci95[{i}]: {g} != {w}"
                )

    for key, got in reconstructed["conditional_asr"].items():
        want = published["conditional_asr"].get(key)
        if want is None:
            mismatches.append(f"{split_label}: missing published ASR key {key}")
            continue
        for field in ("baseline_correct_n", "flipped"):
            if got[field] != want[field]:
                mismatches.append(
                    f"{split_label} ASR {key}.{field}: {got[field]} != {want[field]}"
                )
        if not _approx_equal(got["asr"], want["asr"]):
            mismatches.append(f"{split_label} ASR {key}.asr: {got['asr']} != {want['asr']}")

    for key, by_c in reconstructed["additional_errors"].items():
        want = published["additional_errors"].get(key)
        if want is None:
            mismatches.append(f"{split_label}: missing published additional_error key {key}")
            continue
        for condition in CONDITIONS:
            if not _approx_equal(
                by_c[condition]["mean_additional_error"],
                want[condition]["mean_additional_error"],
            ):
                mismatches.append(
                    f"{split_label} add_err {key}/{condition}: "
                    f"{by_c[condition]['mean_additional_error']} != "
                    f"{want[condition]['mean_additional_error']}"
                )

    return mismatches


def verify_default_artifacts(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or Path(__file__).resolve().parents[2]
    specs = [
        {
            "label": "heldout",
            "compact": root / "reports/artifacts/phase_e_heldout_predictions_compact.jsonl",
            "published": root / "reports/phase-e-heldout-metrics.json",
            "published_getter": lambda m: m["heldout"],
        },
        {
            "label": "development",
            "compact": root / "reports/artifacts/phase_d_full64_predictions_compact.jsonl",
            "published": root / "reports/phase-d-full64-metrics.json",
            "published_getter": lambda m: m,
        },
    ]
    report: dict[str, Any] = {"ok": True, "splits": {}}
    all_mismatches: list[str] = []
    for spec in specs:
        if not spec["compact"].is_file():
            raise FileNotFoundError(
                f"Missing committed compact table {spec['compact']} "
                "(required for fresh-clone verification)."
            )
        if not spec["published"].is_file():
            raise FileNotFoundError(f"Missing published metrics {spec['published']}")
        index = index_compact(load_compact(spec["compact"]))
        reconstructed = reconstruct_core(index)
        published_blob = json.loads(spec["published"].read_text(encoding="utf-8"))
        published = spec["published_getter"](published_blob)
        mismatches = compare_to_published(reconstructed, published, split_label=spec["label"])
        report["splits"][spec["label"]] = {
            "n_images": reconstructed["n_images"],
            "n_prediction_rows": reconstructed["n_prediction_rows"],
            "n_mismatches": len(mismatches),
            "mismatches": mismatches[:20],
        }
        all_mismatches.extend(mismatches)
    report["ok"] = len(all_mismatches) == 0
    report["n_mismatches"] = len(all_mismatches)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()
    report = verify_default_artifacts(args.repo_root)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
