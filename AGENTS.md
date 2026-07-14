# Repository Guidelines

## Project Structure & Module Organization

`snmr/` is the importable Python package. Core modules cover rotations, skeleton and robot models, data loading, losses, metrics, training, and WBT integration (`snmr/integration/`). `scripts/` contains runnable research and export entry points such as `train_phase1.py`, `benchmark.py`, and `export_wbt_npz.py`. Tests live in `tests/`, with small sample data under `tests/fixtures/`. Design notes and experiment records are in `docs/` plus top-level design files. Treat `runs/`, `exports/`, and local `data/` artifacts as generated experiment outputs unless a change explicitly updates a recorded result.

## Build, Test, and Development Commands

Install the package for development with:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[dev,robot]"  # includes MuJoCo-backed robot validation
```

Run the full test suite:

```bash
python -m pytest -q
```

Run focused checks while iterating:

```bash
python -m pytest tests/test_rotation.py -q
python scripts/overfit_batch.py --steps 800
python scripts/benchmark.py --help
```

## Coding Style & Naming Conventions

Use Python 3.10+ syntax and keep dependencies aligned with `pyproject.toml` (`torch`, `numpy`, `scipy`, optional `mujoco`). Follow the existing style: 4-space indentation, type hints where they clarify tensor or array contracts, lowercase module names, `snake_case` functions and variables, and `PascalCase` classes. Preserve package-wide conventions from the README: quaternions are `wxyz` internally, and I/O boundaries should convert deliberately.

## Testing Guidelines

Pytest is configured in `pyproject.toml` with `tests/` as the test root. Name new tests `tests/test_<feature>.py` and test functions `test_<behavior>`. Prefer small deterministic fixtures in `tests/fixtures/`; avoid adding large generated data. For numerical code, assert tolerances explicitly and include regression coverage for shape, device, gradient, and physical-contract changes such as FK, joint limits, and root-pose transforms.

## Commit & Pull Request Guidelines

Git history uses concise subject lines with either a scope prefix (`perf: ...`, `trainers: ...`) or milestone prefix (`E12 engine: ...`, `N9 complete: ...`). Keep commits focused and describe the observable change. Pull requests should include a short problem statement, implementation summary, commands run, and links to relevant docs, issues, or experiment outputs. Include plots or screenshots only when changing generated figures or reports.

## Agent-Specific Instructions

Do not overwrite generated experiment directories casually. Before editing recorded docs or run summaries, check whether they represent a reproducible result and note any command needed to regenerate them.
