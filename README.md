# Tire Model Optimization via LLM Agents

LLM agent code of the paper **"Genetic and agentic symbolic regression of distributed rate-and-state friction models"** by Luigi Romano (Linköping University), Alessandro Lucantonio (Aarhus University), and Marco Virgolin (Unlayer AI), published at the [Workshop on Symbolic Regression and Equation Discovery](https://heal.heuristiclab.com/research/symbolic-regression-workshop). The GP code of the paper is available at [https://github.com/cpml-au/SR-Tyre](https://github.com/cpml-au/SR-Tyre)


> 👀 Note: this paper inspired the creation of an agent SKILL for model discovery, see [automodel](https://github.com/Unlayer-AI/automodel)

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

## Structure

- `code/tire_model_core.py` contains the shared data preparation and split-aware evaluation logic.
- `code/tire_model_loss.py` is the agent-facing loss entrypoint. It exposes only training and validation metrics.
- `code/tire_model_report.py` is the human-facing plotting script with final test metrics.
- `code/tire_model_experiment.py` initializes experiment runs, records each attempt, and finalizes the holdout test evaluation.

## Model Loss

```bash
uv run python code/tire_model_loss.py
```

This evaluates the current `mu_func.py` with no plots and prints JSON containing:

- the training loss,
- the validation loss,
- the visible per-bin losses for the training and validation bins.

The default split is bins `1,2,3` for training and bin `4` for validation (bin `5` is the holdout test bin). The holdout test metric is excluded from this command, but the holdout data and evaluation code are public. The split is therefore honor-based: agents are expected not to inspect or evaluate the test bin before finishing a run. You can override the visible bins:

```bash
uv run python code/tire_model_loss.py \
  --training-bins 1 2 3 \
  --validation-bins 4
```

## Experiment Runs

To run the experiment with your agent, simply paste the content of `initial_prompt.txt` to kick-start the run.

```text
read AGENTS.md and run the optimization as instructed for 20 iterations
```

Your agent should create a timestamped run directory in `logs/` containing:

- `run.json` with the run manifest,
- `agent_prompt.md` with the exact prompt and commands for the agent,
- `attempts.csv` and `attempts.jsonl` with one record per attempt,
- `attempts/attempt_XXX_mu_func.py` snapshots with comment headers,
- `final_summary.json` after the run is finished.

The manifest, per-attempt records, and final summary also store the agent name and model used for the run.

After each change the agent makes to `code/mu_func.py`, it will record the attempt with:

```bash
uv run python code/tire_model_experiment.py record-attempt \
  --run-dir logs/<run-dir> \
  --note "short reason"
```

When the optimization loop is done, the agent finalizes the run and reveals the holdout test metrics (it is instructed to do so only then, and exactly once):

```bash
uv run python code/tire_model_experiment.py finish --run-dir logs/<run-dir>
```

`finish` requires at least one recorded attempt and verifies that the current
`code/mu_func.py` exactly matches the latest recorded attempt.

## Human Report

```bash
uv run python code/tire_model_report.py
```

For headless checks without opening a window:

```bash
uv run python code/tire_model_report.py --no-show
```

## License

This project is licensed under the MIT License. See `LICENSE`.

## Citation
If you use this code in your research, please cite the paper:

```bibtex
@inproceedings{romano2026genetic,
  title={Genetic and agentic symbolic regression of distributed rate-and-state friction models},
  author={Romano, Luigi and Lucantonio, Alessandro and Virgolin, Marco},
  booktitle={Workshop on Symbolic Regression and Equation Discovery (IEEE WCCI/CEC 2026)},
  year={2026}
}
```
