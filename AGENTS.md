# AGENTS.md

## Project Overview

This repository optimizes `code/mu_func.py`, a friction model used by the tire-model evaluator.

Your job is to improve the function `mu_func(v)` so that the training loss decreases while the validation loss remains strong.
You must find a new function form, not only tinker with the parameters/coefficients.

## Files That Matter

- `code/mu_func.py`: the only model file to optimize.
- `code/tire_model_loss.py`: agent-facing evaluator. It exposes only training and validation metrics.
- `code/tire_model_report.py`: human-facing plotting script with final test metrics.
- `code/tire_model_experiment.py`: run manager for initializing experiments, logging attempts, and finalizing runs.
- `logs/`: run manifests, per-attempt snapshots, and final summaries.


## Required Commands

To start a new run before making optimization attempts:

```bash
python code/tire_model_experiment.py start --max-iterations 20 --agent-name "<agent name>" --agent-model "<model name>"
```

To observe the training and validation loss:

```bash
python code/tire_model_loss.py --training-bins 1 2 3 --validation-bins 4
```

To record an attempt:

```bash
python code/tire_model_experiment.py record-attempt --run-dir <logs/run-dir> --note "short reason for the function form change"
```

To finalize the run, exactly once after the last model change:

```bash
python code/tire_model_experiment.py finish --run-dir <logs/run-dir>
```

## Optimization Workflow

1. Start from the current `code/mu_func.py`, record its value as first attempt.
2. Change the function form, ensuring that `mu_func` must be: >0, bounded, and differentiable.
3. After each candidate change to `code/mu_func.py`, run `record-attempt` so the run log captures the timestamp, visible losses, and a Python snapshot of the function.
4. Use the logs in `logs/<run-dir>/` or an internal scratchpad to keep track the progress of the optimization and make informed decisions about what to change at the next iteration.
5. Stop when the configured `max_iterations` is reached or when further progress is not justified.
6. Leave `code/mu_func.py` at the selected final candidate and run `finish` once. Do not continue editing after the holdout metrics are revealed.

## Constraints

- DO NOT CHANGE ANYTHIN ELSE THAN `code/mu_func.py`.
- Do not attempt to install any libraries, use what is available.
- Do not run data analysis. Start guessing the function form immediately.
- Do not try to optimize the parameters (eg with SciPy), just guess them.
- Create and evaluate one function form at the time, no sweeps.
- Do not look at other logs in `logs/` than your own.
- Do not change the data bins policy: always use the command as shown above and in particular never try to access the test bins.
