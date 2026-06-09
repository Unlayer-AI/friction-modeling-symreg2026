from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

DEFAULT_DATA_PATH = Path(__file__).resolve().with_name("lateral_tire_test.csv")
DEFAULT_TRAINING_BINS = (1, 2, 3)
DEFAULT_VALIDATION_BINS = (4,)
DEFAULT_TEST_BINS = (5,)
DEFAULT_THETA = np.array(
    [0.0668, 0.0001, 360.4850, 0.0230, 0.6456, 3.07e-05],
    dtype=float,
)
DEFAULT_V_SPEED = 16.0
DEFAULT_N_BINS = 5
DEFAULT_N_POINTS = 200
DEFAULT_N_X = 100
EPSILON = 1e-12

LONGITUDINAL_DATA = 0
LATERAL_DATA = 1


def _normalize_bins(bins: Iterable[int], label: str) -> tuple[int, ...]:
    normalized = tuple(sorted({int(bin_index) for bin_index in bins}))
    if any(bin_index < 1 for bin_index in normalized):
        raise ValueError(f"{label} bins must be positive integers.")
    return normalized


@dataclass(frozen=True)
class BinSplit:
    training_bins: tuple[int, ...] = DEFAULT_TRAINING_BINS
    validation_bins: tuple[int, ...] = DEFAULT_VALIDATION_BINS
    test_bins: tuple[int, ...] = DEFAULT_TEST_BINS

    def __post_init__(self) -> None:
        training_bins = _normalize_bins(
            self.training_bins,
            "Training",
        )
        validation_bins = _normalize_bins(self.validation_bins, "Validation")
        test_bins = _normalize_bins(self.test_bins, "Test")

        if not training_bins:
            raise ValueError("At least one training bin must be selected.")

        overlaps = (
            (set(training_bins) & set(validation_bins))
            | (set(training_bins) & set(test_bins))
            | (set(validation_bins) & set(test_bins))
        )
        if overlaps:
            overlap_text = ", ".join(str(bin_index) for bin_index in sorted(overlaps))
            raise ValueError(f"Bin splits must be disjoint. Overlap: {overlap_text}")

        object.__setattr__(self, "training_bins", training_bins)
        object.__setattr__(self, "validation_bins", validation_bins)
        object.__setattr__(self, "test_bins", test_bins)

    @property
    def all_bins(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                set(self.training_bins)
                | set(self.validation_bins)
                | set(self.test_bins)
            )
        )


@dataclass(frozen=True)
class PreparedBin:
    index: int
    representative_fz: float
    slip_grid: np.ndarray
    target_mu: np.ndarray
    raw_slip: np.ndarray
    raw_force: np.ndarray


@dataclass(frozen=True)
class PreparedDataset:
    bins: tuple[PreparedBin, ...]
    data_path: Path
    data_type: int

    @property
    def n_bins(self) -> int:
        return len(self.bins)

    def bin_indices(self) -> tuple[int, ...]:
        return tuple(prepared_bin.index for prepared_bin in self.bins)


@dataclass(frozen=True)
class BinEvaluation:
    index: int
    representative_fz: float
    slip_grid: np.ndarray
    target_force: np.ndarray
    predicted_force: np.ndarray
    target_mu: np.ndarray
    predicted_mu: np.ndarray
    loss: float


@dataclass(frozen=True)
class EvaluationSummary:
    split: BinSplit
    bin_results: tuple[BinEvaluation, ...]

    def results_for_bins(
        self,
        bins: Iterable[int],
    ) -> tuple[BinEvaluation, ...]:
        selected_bins = set(bins)
        return tuple(
            result for result in self.bin_results if result.index in selected_bins
        )

    def loss_for_bins(self, bins: Iterable[int]) -> float:
        """Returns the mean loss over the bins"""
        return float(
            np.mean([result.loss for result in self.results_for_bins(bins)])
        )

    def loss_by_bin(
        self,
        bins: Iterable[int] | None = None,
    ) -> dict[int, float]:
        results = self.bin_results if bins is None else self.results_for_bins(bins)
        return {result.index: result.loss for result in results}

    def representative_fz_by_bin(
        self,
        bins: Iterable[int] | None = None,
    ) -> dict[int, float]:
        results = self.bin_results if bins is None else self.results_for_bins(bins)
        return {result.index: result.representative_fz for result in results}

    @property
    def training_loss(self) -> float:
        return self.loss_for_bins(self.split.training_bins)

    @property
    def validation_loss(self) -> float:
        return self.loss_for_bins(self.split.validation_bins)

    @property
    def test_loss(self) -> float:
        return self.loss_for_bins(self.split.test_bins)

    @property
    def total_loss(self) -> float:
        return float(sum(result.loss for result in self.bin_results))

    def to_dict(self) -> dict[str, object]:
        return {
            "training_bins": list(self.split.training_bins),
            "validation_bins": list(self.split.validation_bins),
            "test_bins": list(self.split.test_bins),
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "test_loss": self.test_loss,
            "total_loss": self.total_loss,
            "loss_by_bin": self.loss_by_bin(),
            "representative_fz_by_bin": self.representative_fz_by_bin(),
        }


def load_tire_test_data(
    data_path: Path | str = DEFAULT_DATA_PATH,
    data_type: int = LATERAL_DATA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(data_path)

    slip_ratio = frame["SR"].to_numpy(dtype=float)
    sigma_x = slip_ratio / (1.0 + slip_ratio) + 0.05
    sigma_y = np.tan(
        frame["SA"].to_numpy(dtype=float) * np.pi / 180.0 / (1.0 + slip_ratio)
    )

    fx = frame["FX"].to_numpy(dtype=float)
    fy = -frame["FY"].to_numpy(dtype=float)
    fz = -frame["FZ"].to_numpy(dtype=float) * 4.448

    if data_type == LONGITUDINAL_DATA:
        return sigma_x, fx, fz
    if data_type == LATERAL_DATA:
        return sigma_y, fy, fz

    raise ValueError("data_type must be 0 (longitudinal) or 1 (lateral).")


def _clean_bin_samples(
    slip: np.ndarray,
    force: np.ndarray,
    slip_threshold: float,
    mad_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(slip)
    slip = slip[order]
    force = force[order]

    finite_mask = np.isfinite(slip) & np.isfinite(force)
    slip = slip[finite_mask]
    force = force[finite_mask]

    slip_mask = np.abs(slip) > slip_threshold
    slip = slip[slip_mask]
    force = force[slip_mask]

    if slip.size < 2:
        raise ValueError("A bin has too few valid samples after slip filtering.")

    median_force = np.median(force)
    mad_force = np.median(np.abs(force - median_force)) + 1e-12
    mad_mask = np.abs(force - median_force) < mad_threshold * mad_force
    slip = slip[mad_mask]
    force = force[mad_mask]

    if slip.size < 2:
        raise ValueError("A bin has too few valid samples after MAD filtering.")

    unique_slip, inverse = np.unique(slip, return_inverse=True)
    averaged_force = np.zeros(unique_slip.size, dtype=float)
    for unique_index in range(unique_slip.size):
        averaged_force[unique_index] = np.mean(force[inverse == unique_index])

    if unique_slip.size < 2:
        raise ValueError(
            "A bin needs at least two unique slip values for interpolation."
        )

    return unique_slip, averaged_force


def prepare_dataset(
    data_path: Path | str = DEFAULT_DATA_PATH,
    data_type: int = LATERAL_DATA,
    n_bins: int = DEFAULT_N_BINS,
    n_points: int = DEFAULT_N_POINTS,
    slip_threshold: float = 1e-4,
    mad_threshold: float = 2.0,
) -> PreparedDataset:
    slip, force, fz = load_tire_test_data(
        data_path=data_path,
        data_type=data_type,
    )

    fz_edges = np.linspace(fz.min(), fz.max(), n_bins + 1)
    fz_bin = np.digitize(fz, fz_edges, right=False)
    fz_bin = np.clip(fz_bin, 1, n_bins)

    prepared_bins: list[PreparedBin] = []
    for bin_index in range(1, n_bins + 1):
        mask = fz_bin == bin_index
        if not np.any(mask):
            raise ValueError(f"Bin {bin_index} has no samples.")

        raw_slip, raw_force = _clean_bin_samples(
            slip[mask],
            force[mask],
            slip_threshold=slip_threshold,
            mad_threshold=mad_threshold,
        )
        representative_fz = float(np.mean(fz[mask]))

        slip_grid = np.linspace(raw_slip.min(), raw_slip.max(), n_points)
        interpolator = PchipInterpolator(
            raw_slip,
            raw_force / representative_fz,
        )
        target_mu = interpolator(slip_grid)
        
        prepared_bins.append(
            PreparedBin(
                index=bin_index,
                representative_fz=representative_fz,
                slip_grid=slip_grid,
                target_mu=target_mu,
                raw_slip=raw_slip,
                raw_force=raw_force,
            )
        )

    return PreparedDataset(
        bins=tuple(prepared_bins),
        data_path=Path(data_path).resolve(),
        data_type=data_type,
    )


def _validate_split(dataset: PreparedDataset, split: BinSplit) -> None:
    available_bins = set(dataset.bin_indices())
    requested_bins = set(split.all_bins)
    missing_bins = requested_bins - available_bins
    if missing_bins:
        missing_text = ", ".join(str(bin_index) for bin_index in sorted(missing_bins))
        raise ValueError(
            "Requested bins are not available in the dataset: " f"{missing_text}"
        )


def _evaluate_bin(
    mu_function: Callable[[float], float],
    prepared_bin: PreparedBin,
    theta: np.ndarray,
    v_speed: float,
    n_x: int,
) -> BinEvaluation:
    representative_fz = prepared_bin.representative_fz
    contact_length = theta[0] + theta[1] * np.sqrt(representative_fz)
    if contact_length <= 0.0:
        raise ValueError("The fitted contact length must stay positive.")

    peak_friction = theta[4] - theta[5] * representative_fz
    if peak_friction <= 0.0:
        raise ValueError("The fitted peak friction must stay positive.")

    stiffness = (theta[2] - theta[3] * representative_fz) / peak_friction
    if stiffness <= 0.0:
        raise ValueError("The fitted stiffness must stay positive.")

    xi = np.linspace(0.0, contact_length, n_x)
    delta_xi = xi[1] - xi[0]
    velocity = -prepared_bin.slip_grid * v_speed

    try:
        mu_values = np.array(
            [float(mu_function(float(v_i))) for v_i in velocity],
            dtype=float,
        )
    except Exception as exc:  # pragma: no cover
        raise ValueError(
            f"mu_func failed while evaluating bin {prepared_bin.index}: {exc}"
        ) from exc

    if mu_values.shape != velocity.shape or not np.all(np.isfinite(mu_values)):
        raise ValueError(
            f"mu_func returned non-finite values for bin {prepared_bin.index}."
        )
    if np.any(mu_values <= 0.0):
        raise ValueError(
            "mu_func returned non-positive friction for bin " f"{prepared_bin.index}."
        )

    speed_term = np.sqrt(velocity**2 + EPSILON)
    scale = (-mu_values / stiffness) * (velocity / speed_term)
    decay = (speed_term / v_speed) * (stiffness / mu_values)
    z = scale[:, None] * (1.0 - np.exp(-decay[:, None] * xi[None, :]))

    pressure = representative_fz / contact_length
    predicted_force = (
        peak_friction * np.sum(z, axis=1) * delta_xi * stiffness * pressure
    )
    predicted_mu = predicted_force / representative_fz
    target_force = prepared_bin.target_mu * representative_fz
    loss = float(np.mean((predicted_mu - prepared_bin.target_mu) ** 2))

    return BinEvaluation(
        index=prepared_bin.index,
        representative_fz=representative_fz,
        slip_grid=prepared_bin.slip_grid,
        target_force=target_force,
        predicted_force=predicted_force,
        target_mu=prepared_bin.target_mu,
        predicted_mu=predicted_mu,
        loss=loss,
    )


def evaluate_dataset(
    mu_function: Callable[[float], float],
    dataset: PreparedDataset,
    split: BinSplit = BinSplit(),
    theta: np.ndarray | list[float] = DEFAULT_THETA,
    v_speed: float = DEFAULT_V_SPEED,
    n_x: int = DEFAULT_N_X,
) -> EvaluationSummary:
    if v_speed <= 0.0:
        raise ValueError("v_speed must be positive.")
    if n_x < 2:
        raise ValueError("n_x must be at least 2.")

    _validate_split(dataset, split)

    theta_array = np.asarray(theta, dtype=float)
    if theta_array.shape != (6,):
        raise ValueError("theta must contain exactly six tire parameters.")

    bin_results = tuple(
        _evaluate_bin(
            mu_function=mu_function,
            prepared_bin=prepared_bin,
            theta=theta_array,
            v_speed=v_speed,
            n_x=n_x,
        )
        for prepared_bin in dataset.bins
    )
    return EvaluationSummary(split=split, bin_results=bin_results)


def evaluate_mu_function(
    mu_function: Callable[[float], float],
    *,
    data_path: Path | str = DEFAULT_DATA_PATH,
    split: BinSplit = BinSplit(),
    theta: np.ndarray | list[float] = DEFAULT_THETA,
    v_speed: float = DEFAULT_V_SPEED,
    n_x: int = DEFAULT_N_X,
    data_type: int = LATERAL_DATA,
    n_bins: int = DEFAULT_N_BINS,
    n_points: int = DEFAULT_N_POINTS,
    slip_threshold: float = 1e-4,
    mad_threshold: float = 2.0,
) -> EvaluationSummary:
    dataset = prepare_dataset(
        data_path=data_path,
        data_type=data_type,
        n_bins=n_bins,
        n_points=n_points,
        slip_threshold=slip_threshold,
        mad_threshold=mad_threshold,
    )
    return evaluate_dataset(
        mu_function,
        dataset,
        split=split,
        theta=theta,
        v_speed=v_speed,
        n_x=n_x,
    )
