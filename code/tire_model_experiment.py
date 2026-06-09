from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tire_model_core import (
    BinSplit,
    DEFAULT_DATA_PATH,
    DEFAULT_TRAINING_BINS,
    DEFAULT_TEST_BINS,
    DEFAULT_VALIDATION_BINS,
    evaluate_mu_function,
)
from tire_model_loss import build_agent_loss_payload

ROOT_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT_DIR / "logs"
MU_FUNC_PATH = Path(__file__).resolve().with_name("mu_func.py")
UNKNOWN_AGENT = "unknown"
UNKNOWN_MODEL = "unknown"
ATTEMPTS_CSV_HEADER = (
    "attempt,logged_at_utc,elapsed_seconds,agent_name,agent_model,"
    "training_loss,validation_loss,note,snapshot_path,mu_func_sha256\n"
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_timestamp(timestamp: datetime) -> str:
    return timestamp.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def sanitize_label(label: str | None) -> str:
    if not label:
        return ""
    cleaned = [
        character.lower()
        for character in label
        if character.isalnum() or character in {"-", "_"}
    ]
    return "".join(cleaned).strip("-_")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def mu_func_source(snapshot_body: str | None = None) -> str:
    if snapshot_body is not None:
        return snapshot_body
    return MU_FUNC_PATH.read_text(encoding="utf-8")


def source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def load_mu_function():
    from mu_func import mu_func

    return mu_func


def normalize_identity(value: str | None, fallback: str) -> str:
    if value is None:
        return fallback
    cleaned = " ".join(value.split())
    return cleaned or fallback


def agent_metadata_from_args(args: argparse.Namespace) -> dict[str, str]:
    return {
        "agent_name": normalize_identity(
            getattr(args, "agent_name", None),
            UNKNOWN_AGENT,
        ),
        "agent_model": normalize_identity(
            getattr(args, "agent_model", None),
            UNKNOWN_MODEL,
        ),
    }


def agent_metadata_from_manifest(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        "agent_name": normalize_identity(
            manifest.get("agent_name"),
            UNKNOWN_AGENT,
        ),
        "agent_model": normalize_identity(
            manifest.get("agent_model"),
            UNKNOWN_MODEL,
        ),
    }


def build_agent_prompt(manifest: dict[str, Any], run_dir: Path) -> str:
    training_bins = ", ".join(
        str(bin_index) for bin_index in manifest["training_bins"]
    )
    validation_bins = ", ".join(
        str(bin_index) for bin_index in manifest["validation_bins"]
    )
    test_bins = ", ".join(
        str(bin_index) for bin_index in manifest["test_bins"]
    )
    run_dir_text = run_dir.relative_to(ROOT_DIR).as_posix()
    agent_name = manifest["agent_name"]
    agent_model = manifest["agent_model"]

    return (
        "# Tire Model Experiment\n\n"
        f"Run directory: `{run_dir_text}`\n"
        f"Agent name: `{agent_name}`\n"
        f"Agent model: `{agent_model}`\n"
        f"Max iterations: `{manifest['max_iterations']}`\n"
        f"Training bins: `{training_bins}`\n"
        f"Validation bins: `{validation_bins}`\n"
        f"Holdout test bins: `{test_bins}`\n\n"
        "Follow `AGENTS.md` in this repository.\n\n"
        "Required workflow:\n"
        "1. Only edit `code/mu_func.py`.\n"
        "2. Do not inspect or optimize against the holdout test bins.\n"
        "3. After every candidate change, run:\n"
        "   `python code/tire_model_experiment.py record-attempt "
        f"--run-dir {run_dir_text} --note \"<short reason>\"`\n"
        "4. Stop after the configured maximum number of attempts or "
        "when you are stuck.\n"
        "5. Leave `code/mu_func.py` at the chosen final candidate and run:\n"
        "   `python code/tire_model_experiment.py finish "
        f"--run-dir {run_dir_text}`\n"
        "6. Do not make further edits after `finish`; that command "
        "reveals the holdout test metrics.\n"
    )


def build_run_id(label: str | None) -> str:
    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    sanitized = sanitize_label(label)
    if not sanitized:
        return timestamp
    return f"{timestamp}-{sanitized}"


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "run.json"


def csv_path(run_dir: Path) -> Path:
    return run_dir / "attempts.csv"


def jsonl_path(run_dir: Path) -> Path:
    return run_dir / "attempts.jsonl"


def attempts_dir(run_dir: Path) -> Path:
    return run_dir / "attempts"


def final_summary_path(run_dir: Path) -> Path:
    return run_dir / "final_summary.json"


def load_manifest(run_dir: Path) -> dict[str, Any]:
    return read_json(manifest_path(run_dir))


def visible_split_from_manifest(manifest: dict[str, Any]) -> BinSplit:
    return BinSplit(
        training_bins=tuple(manifest["training_bins"]),
        validation_bins=tuple(manifest["validation_bins"]),
        test_bins=(),
    )


def full_split_from_manifest(manifest: dict[str, Any]) -> BinSplit:
    return BinSplit(
        training_bins=tuple(manifest["training_bins"]),
        validation_bins=tuple(manifest["validation_bins"]),
        test_bins=tuple(manifest["test_bins"]),
    )


def evaluate_current_mu(
    manifest: dict[str, Any],
    include_test: bool,
):
    return evaluate_mu_callable(
        manifest,
        load_mu_function(),
        include_test=include_test,
    )


def evaluate_mu_callable(
    manifest: dict[str, Any],
    mu_function,
    *,
    include_test: bool,
):
    split = (
        full_split_from_manifest(manifest)
        if include_test
        else visible_split_from_manifest(manifest)
    )
    return evaluate_mu_function(
        mu_function,
        data_path=manifest["data_path"],
        split=split,
    )


def append_attempt_files(run_dir: Path, record: dict[str, Any]) -> None:
    with jsonl_path(run_dir).open("a") as handle:
        handle.write(json.dumps(record) + "\n")

    with csv_path(run_dir).open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                record["attempt"],
                record["logged_at_utc"],
                f"{record['elapsed_seconds']:.6f}",
                record["agent_name"],
                record["agent_model"],
                f"{record['training_loss']:.12f}",
                f"{record['validation_loss']:.12f}",
                record["note"],
                record["snapshot_path"],
                record["mu_func_sha256"],
            ]
        )


def normalize_note(note: str) -> str:
    cleaned = " ".join(note.split())
    if not cleaned:
        raise ValueError("A non-empty note is required for each attempt.")
    return cleaned


def relative_path(path: Path) -> str:
    return path.relative_to(ROOT_DIR).as_posix()


def write_snapshot(
    snapshot_path: Path,
    header_lines: list[str],
    snapshot_body: str | None = None,
) -> None:
    header = "\n".join(header_lines)
    body = snapshot_body
    if body is None:
        body = MU_FUNC_PATH.read_text(encoding="utf-8")
    snapshot_path.write_text(
        f"{header}\n\n{body}",
        encoding="utf-8",
    )


def update_best_attempt(
    manifest: dict[str, Any],
    record: dict[str, Any],
) -> None:
    best_attempt = manifest.get("best_attempt")
    if best_attempt is None:
        manifest["best_attempt"] = record
        return

    best_training = float(best_attempt["training_loss"])
    candidate_training = float(record["training_loss"])
    if candidate_training < best_training:
        manifest["best_attempt"] = record
        return

    if candidate_training == best_training and float(
        record["validation_loss"]
    ) < float(best_attempt["validation_loss"]):
        manifest["best_attempt"] = record


def start_run(
    *,
    label: str | None,
    max_iterations: int,
    agent_name: str,
    agent_model: str,
    data_path: Path | str,
    training_bins: tuple[int, ...],
    validation_bins: tuple[int, ...],
    test_bins: tuple[int, ...],
) -> dict[str, Any]:
    split = BinSplit(
        training_bins=training_bins,
        validation_bins=validation_bins,
        test_bins=test_bins,
    )
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    run_id = build_run_id(label)
    run_dir = LOGS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    attempts_dir(run_dir).mkdir()

    started_at = utc_now()
    agent_metadata = {
        "agent_name": normalize_identity(agent_name, UNKNOWN_AGENT),
        "agent_model": normalize_identity(agent_model, UNKNOWN_MODEL),
    }
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "label": label,
        "status": "running",
        "started_at_utc": format_timestamp(started_at),
        "ended_at_utc": None,
        "elapsed_seconds": None,
        "data_path": str(Path(data_path).resolve()),
        "max_iterations": max_iterations,
        "agent_name": agent_metadata["agent_name"],
        "agent_model": agent_metadata["agent_model"],
        "attempt_count": 0,
        "training_bins": list(split.training_bins),
        "validation_bins": list(split.validation_bins),
        "test_bins": list(split.test_bins),
        "best_attempt": None,
        "last_attempt": None,
    }

    write_json(manifest_path(run_dir), manifest)
    csv_path(run_dir).write_text(ATTEMPTS_CSV_HEADER)
    jsonl_path(run_dir).write_text("")

    write_snapshot(
        run_dir / "initial_mu_func.py",
        [
            f"# Run ID: {run_id}",
            f"# Logged at (UTC): {format_timestamp(started_at)}",
            f"# Agent name: {agent_metadata['agent_name']}",
            f"# Agent model: {agent_metadata['agent_model']}",
            "# Snapshot type: initial",
        ],
    )
    prompt_path = run_dir / "agent_prompt.md"
    prompt_path.write_text(build_agent_prompt(manifest, run_dir))

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "prompt_path": str(prompt_path),
        "max_iterations": max_iterations,
        "agent_name": agent_metadata["agent_name"],
        "agent_model": agent_metadata["agent_model"],
        "training_bins": list(split.training_bins),
        "validation_bins": list(split.validation_bins),
        "test_bins": list(split.test_bins),
    }


def record_attempt_for_mu(
    *,
    run_dir: Path | str,
    mu_function,
    note: str,
    snapshot_body: str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    manifest = load_manifest(run_dir)
    if manifest["status"] != "running":
        raise ValueError("This run is already finished. Start a new run.")

    attempt_index = int(manifest["attempt_count"]) + 1
    if attempt_index > int(manifest["max_iterations"]):
        raise ValueError("The configured max_iterations has been reached.")

    summary = evaluate_mu_callable(
        manifest,
        mu_function,
        include_test=False,
    )
    visible_payload = build_agent_loss_payload(summary)
    agent_metadata = agent_metadata_from_manifest(manifest)
    logged_at = utc_now()
    elapsed_seconds = (
        logged_at - parse_timestamp(manifest["started_at_utc"])
    ).total_seconds()
    note = normalize_note(note)

    snapshot_path = (
        attempts_dir(run_dir) / f"attempt_{attempt_index:03d}_mu_func.py"
    )
    source = mu_func_source(snapshot_body)
    write_snapshot(
        snapshot_path,
        [
            f"# Run ID: {manifest['run_id']}",
            f"# Attempt: {attempt_index}",
            f"# Logged at (UTC): {format_timestamp(logged_at)}",
            f"# Agent name: {agent_metadata['agent_name']}",
            f"# Agent model: {agent_metadata['agent_model']}",
            f"# Elapsed seconds: {elapsed_seconds:.6f}",
            f"# Training loss: {summary.training_loss:.12f}",
            f"# Validation loss: {summary.validation_loss:.12f}",
            f"# Note: {note}",
        ],
        snapshot_body=source,
    )

    record = {
        "attempt": attempt_index,
        "logged_at_utc": format_timestamp(logged_at),
        "elapsed_seconds": elapsed_seconds,
        "agent_name": agent_metadata["agent_name"],
        "agent_model": agent_metadata["agent_model"],
        "training_loss": summary.training_loss,
        "validation_loss": summary.validation_loss,
        "visible_loss": visible_payload["visible_loss"],
        "loss_by_bin": visible_payload["loss_by_bin"],
        "note": note,
        "snapshot_path": relative_path(snapshot_path),
        "mu_func_sha256": source_sha256(source),
    }
    append_attempt_files(run_dir, record)

    manifest["attempt_count"] = attempt_index
    manifest["last_attempt"] = record
    update_best_attempt(manifest, record)
    write_json(manifest_path(run_dir), manifest)
    return record


def finish_run_for_mu(
    *,
    run_dir: Path | str,
    mu_function,
    snapshot_body: str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    manifest = load_manifest(run_dir)
    if manifest["status"] != "running":
        raise ValueError("This run has already been finished.")

    last_attempt = manifest.get("last_attempt")
    if last_attempt is None:
        raise ValueError("Record at least one attempt before finishing the run.")

    source = mu_func_source(snapshot_body)
    if source_sha256(source) != last_attempt.get("mu_func_sha256"):
        raise ValueError(
            "The current mu_func.py does not match the latest recorded attempt. "
            "Record it before finishing the run."
        )

    summary = evaluate_mu_callable(manifest, mu_function, include_test=True)
    agent_metadata = agent_metadata_from_manifest(manifest)
    finished_at = utc_now()
    elapsed_seconds = (
        finished_at - parse_timestamp(manifest["started_at_utc"])
    ).total_seconds()

    snapshot_path = run_dir / "final_mu_func.py"
    write_snapshot(
        snapshot_path,
        [
            f"# Run ID: {manifest['run_id']}",
            f"# Finished at (UTC): {format_timestamp(finished_at)}",
            f"# Agent name: {agent_metadata['agent_name']}",
            f"# Agent model: {agent_metadata['agent_model']}",
            f"# Total elapsed seconds: {elapsed_seconds:.6f}",
            f"# Attempt count: {manifest['attempt_count']}",
            f"# Training loss: {summary.training_loss:.12f}",
            f"# Validation loss: {summary.validation_loss:.12f}",
            f"# Test loss: {summary.test_loss:.12f}",
        ],
        snapshot_body=source,
    )

    final_summary = {
        "run_id": manifest["run_id"],
        "status": "completed",
        "started_at_utc": manifest["started_at_utc"],
        "ended_at_utc": format_timestamp(finished_at),
        "elapsed_seconds": elapsed_seconds,
        "attempt_count": manifest["attempt_count"],
        "max_iterations": manifest["max_iterations"],
        "agent_name": agent_metadata["agent_name"],
        "agent_model": agent_metadata["agent_model"],
        "training_bins": manifest["training_bins"],
        "validation_bins": manifest["validation_bins"],
        "test_bins": manifest["test_bins"],
        "training_loss": summary.training_loss,
        "validation_loss": summary.validation_loss,
        "test_loss": summary.test_loss,
        "total_loss": summary.total_loss,
        "loss_by_bin": summary.loss_by_bin(summary.split.all_bins),
        "representative_fz_by_bin": summary.representative_fz_by_bin(
            summary.split.all_bins
        ),
        "best_attempt": manifest.get("best_attempt"),
        "final_snapshot_path": relative_path(snapshot_path),
    }

    manifest["status"] = "completed"
    manifest["ended_at_utc"] = final_summary["ended_at_utc"]
    manifest["elapsed_seconds"] = elapsed_seconds
    manifest["final_summary"] = final_summary
    write_json(manifest_path(run_dir), manifest)
    write_json(final_summary_path(run_dir), final_summary)
    return final_summary


def handle_start(args: argparse.Namespace) -> None:
    payload = start_run(
        label=args.label,
        max_iterations=args.max_iterations,
        agent_name=agent_metadata_from_args(args)["agent_name"],
        agent_model=agent_metadata_from_args(args)["agent_model"],
        data_path=args.data_path,
        training_bins=tuple(args.training_bins),
        validation_bins=tuple(args.validation_bins),
        test_bins=tuple(args.test_bins),
    )
    print(json.dumps(payload, indent=2))


def handle_record_attempt(args: argparse.Namespace) -> None:
    record = record_attempt_for_mu(
        run_dir=args.run_dir,
        mu_function=load_mu_function(),
        note=args.note,
    )
    print(json.dumps(record, indent=2))


def handle_finish(args: argparse.Namespace) -> None:
    final_summary = finish_run_for_mu(
        run_dir=args.run_dir,
        mu_function=load_mu_function(),
    )
    print(json.dumps(final_summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and log tire-model optimization experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser(
        "start",
        help="Create a new experiment run directory and prompt.",
    )
    start_parser.add_argument(
        "--label",
        default=None,
        help="Optional label added to the run directory name.",
    )
    start_parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of recorded optimization attempts.",
    )
    start_parser.add_argument(
        "--agent-name",
        default=UNKNOWN_AGENT,
        help="Name of the coding agent running the experiment.",
    )
    start_parser.add_argument(
        "--agent-model",
        default=UNKNOWN_MODEL,
        help="LLM or model identifier used by the coding agent.",
    )
    start_parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV file with the tire test data.",
    )
    start_parser.add_argument(
        "--training-bins",
        type=int,
        nargs="*",
        default=DEFAULT_TRAINING_BINS,
        help="Bins used for training.",
    )
    start_parser.add_argument(
        "--validation-bins",
        type=int,
        nargs="*",
        default=DEFAULT_VALIDATION_BINS,
        help="Bins reserved for validation.",
    )
    start_parser.add_argument(
        "--test-bins",
        type=int,
        nargs="*",
        default=DEFAULT_TEST_BINS,
        help="Holdout bins revealed only when the run is finished.",
    )
    start_parser.set_defaults(handler=handle_start)

    attempt_parser = subparsers.add_parser(
        "record-attempt",
        help="Log the current mu_func.py as the next experiment attempt.",
    )
    attempt_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory created by the start command.",
    )
    attempt_parser.add_argument(
        "--note",
        required=True,
        help="Short explanation of the current mu_func change.",
    )
    attempt_parser.set_defaults(handler=handle_record_attempt)

    finish_parser = subparsers.add_parser(
        "finish",
        help="Finalize a run, compute the holdout loss, and freeze the log.",
    )
    finish_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory created by the start command.",
    )
    finish_parser.set_defaults(handler=handle_finish)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
