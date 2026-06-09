from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from mu_func import mu_func
from tire_model_core import (
    BinSplit,
    DEFAULT_DATA_PATH,
    EvaluationSummary,
    DEFAULT_TEST_BINS,
    evaluate_dataset,
    prepare_dataset,
)

MARKER_COLORS = [
    (0.55, 0.80, 0.95),
    (1.00, 0.75, 0.65),
    (0.95, 0.85, 0.65),
    (0.75, 0.65, 0.75),
    (0.80, 0.90, 0.75),
]
LINE_COLORS = [
    (0.00, 0.45, 0.74),
    (0.85, 0.33, 0.10),
    (0.93, 0.69, 0.13),
    (0.49, 0.18, 0.56),
    (0.47, 0.67, 0.19),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a human-readable tire-model report with optional plots."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV file with the tire test data.",
    )
    parser.add_argument(
        "--training-bins",
        type=int,
        nargs="*",
        default=(1, 2, 3),
        help="Bins used for training.",
    )
    parser.add_argument(
        "--validation-bins",
        type=int,
        nargs="*",
        default=(4,),
        help="Bins reserved for validation.",
    )
    parser.add_argument(
        "--test-bins",
        type=int,
        nargs="*",
        default=DEFAULT_TEST_BINS,
        help="Bins reserved for final testing only.",
    )
    parser.add_argument(
        "--plot-bins",
        type=int,
        nargs="*",
        default=None,
        help="Subset of bins to plot. By default all bins are shown.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional path for saving the plot instead of only showing it.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip opening the interactive plot window.",
    )
    return parser.parse_args()


def print_summary(summary: EvaluationSummary) -> None:
    print(f"training_loss: {summary.training_loss:.6f}")
    print(f"validation_loss: {summary.validation_loss:.6f}")
    print(f"test_loss: {summary.test_loss:.6f}")
    print(f"total_loss: {summary.total_loss:.6f}")
    print("per_bin_loss:")
    for result in summary.bin_results:
        print(
            f"  bin {result.index}: loss={result.loss:.6f}, "
            f"Fz={result.representative_fz / 1000.0:.3f} kN"
        )


def plot_summary(
    summary: EvaluationSummary,
    plot_bins: tuple[int, ...] | None,
) -> None:
    selected_bins = (
        set(plot_bins)
        if plot_bins
        else {result.index for result in summary.bin_results}
    )

    _, ax = plt.subplots()
    for result in summary.bin_results:
        if result.index not in selected_bins:
            continue

        color_index = (result.index - 1) % len(LINE_COLORS)
        label = (
            f"Bin {result.index} "
            f"({result.representative_fz / 1000.0:.1f} kN)"
        )
        ax.plot(
            result.slip_grid,
            result.target_force / 1000.0,
            "o",
            color=MARKER_COLORS[color_index],
            markerfacecolor=MARKER_COLORS[color_index],
            markersize=4,
        )
        ax.plot(
            result.slip_grid,
            result.predicted_force / 1000.0,
            color=LINE_COLORS[color_index],
            linewidth=1,
            label=label,
        )

    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-3.1, 3.1)
    ax.grid(True)
    ax.set_xlabel(r"Lateral slip $\sigma_y$ (-)")
    ax.set_ylabel(r"Lateral force $F_y$ (kN)")
    ax.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.legend(loc="lower right")
    plt.tight_layout()


def main() -> None:
    args = parse_args()

    split = BinSplit(
        training_bins=tuple(args.training_bins),
        validation_bins=tuple(args.validation_bins),
        test_bins=tuple(args.test_bins),
    )
    dataset = prepare_dataset(data_path=args.data_path)
    summary = evaluate_dataset(mu_func, dataset, split=split)

    print_summary(summary)
    plot_summary(
        summary,
        tuple(args.plot_bins) if args.plot_bins is not None else None,
    )

    if args.save_path is not None:
        plt.savefig(args.save_path, bbox_inches="tight")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
