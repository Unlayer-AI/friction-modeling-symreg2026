from __future__ import annotations

import argparse
import json
from pathlib import Path

from mu_func import mu_func
from tire_model_core import (
    BinSplit,
    DEFAULT_DATA_PATH,
    DEFAULT_TRAINING_BINS,
    DEFAULT_VALIDATION_BINS,
    evaluate_dataset,
    prepare_dataset,
)


def round_loss(value: float) -> float:
    return round(float(value), 4)


def build_agent_loss_payload(
    summary,
) -> dict[str, object]:
    visible_bins = tuple(
        sorted(
            set(summary.split.training_bins)
            | set(summary.split.validation_bins)
        )
    )
    return {
        "training_bins": list(summary.split.training_bins),
        "validation_bins": list(summary.split.validation_bins),
        "training_loss": round_loss(summary.training_loss),
        "validation_loss": round_loss(summary.validation_loss),
        "visible_loss": round_loss(summary.loss_for_bins(visible_bins)),
        "loss_by_bin": {
            bin_index: round_loss(loss)
            for bin_index, loss in summary.loss_by_bin(visible_bins).items()
        },
        "representative_fz_by_bin": summary.representative_fz_by_bin(
            visible_bins
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute agent-visible tire-model loss without revealing the "
            "holdout test split."
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
        default=DEFAULT_TRAINING_BINS,
        help="Bins used for training.",
    )
    parser.add_argument(
        "--validation-bins",
        type=int,
        nargs="*",
        default=DEFAULT_VALIDATION_BINS,
        help="Bins reserved for validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        split = BinSplit(
            training_bins=tuple(args.training_bins),
            validation_bins=tuple(args.validation_bins),
            test_bins=(),
        )
        dataset = prepare_dataset(data_path=args.data_path)
        summary = evaluate_dataset(mu_func, dataset, split=split)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        raise SystemExit(1) from exc

    print(json.dumps(build_agent_loss_payload(summary), indent=2))


if __name__ == "__main__":
    main()
