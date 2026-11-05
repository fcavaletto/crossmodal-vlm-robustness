"""Analyze held-out Phase E results; optionally compare to development metrics.

Primary statistics always come from the held-out JSONL alone. Development
metrics (if provided) are nested under a separate key for side-by-side tables —
never mixed into held-out denominators, and never hardcoded from Phase D.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .phase_d_analyze import build_metrics, load_rows, make_plots


def compare_splits(
    *,
    heldout_metrics: dict[str, Any],
    development_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Package held-out and development metrics as separate blocks."""
    out: dict[str, Any] = {
        "heldout": heldout_metrics,
        "development": development_metrics,
        "comparison_note": (
            "Held-out metrics are computed only from the test-split JSONL. "
            "Development metrics (if present) are shown separately for comparison "
            "and must not be treated as held-out evidence."
        ),
    }
    if development_metrics is None:
        return out

    # Compact side-by-side for the primary / secondary DiD keys shared by both.
    side_by_side: dict[str, Any] = {}
    held_did = heldout_metrics.get("did_bootstrap", {})
    dev_did = development_metrics.get("did_bootstrap", {})
    for key in sorted(set(held_did) | set(dev_did)):
        side_by_side[key] = {
            "heldout": held_did.get(key),
            "development": dev_did.get(key),
        }
    out["did_side_by_side"] = side_by_side

    held_floor = heldout_metrics.get("floor_effect_color_misleading")
    dev_floor = development_metrics.get("floor_effect_color_misleading")
    out["floor_side_by_side"] = {
        "heldout": held_floor,
        "development": dev_floor,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True, help="Held-out Phase E JSONL")
    parser.add_argument(
        "--development-jsonl",
        type=Path,
        default=None,
        help="Optional Phase D development JSONL for side-by-side comparison only",
    )
    parser.add_argument(
        "--development-metrics",
        type=Path,
        default=None,
        help="Optional precomputed Phase D metrics JSON (alternative to --development-jsonl)",
    )
    parser.add_argument("--output", type=Path, default=Path("results/phase_e_heldout_metrics.json"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures/phase-e"))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    if "phase_d" in args.output.stem:
        raise SystemExit(
            f"Refusing held-out metrics output that looks like Phase D: {args.output}"
        )

    held_rows = load_rows(args.jsonl)
    held_metrics = build_metrics(held_rows, n_boot=args.n_boot)
    figure_paths: list[str] = []
    if not args.skip_plots:
        figure_paths = make_plots(held_metrics, held_rows, args.figure_dir)
    held_metrics["figures"] = figure_paths
    held_metrics["split"] = "test"
    held_metrics["source_jsonl"] = str(args.jsonl)

    development_metrics = None
    if args.development_metrics is not None:
        development_metrics = json.loads(args.development_metrics.read_text(encoding="utf-8"))
        development_metrics.setdefault("split", "development")
    elif args.development_jsonl is not None:
        development_metrics = build_metrics(load_rows(args.development_jsonl), n_boot=args.n_boot)
        development_metrics["split"] = "development"
        development_metrics["source_jsonl"] = str(args.development_jsonl)

    payload = compare_splits(
        heldout_metrics=held_metrics,
        development_metrics=development_metrics,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "heldout_n_rows": held_metrics["n_rows"],
        "heldout_n_images": held_metrics["n_images"],
        "heldout_floor": held_metrics.get("floor_effect_color_misleading"),
        "has_development_comparison": development_metrics is not None,
        "figures": figure_paths,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
