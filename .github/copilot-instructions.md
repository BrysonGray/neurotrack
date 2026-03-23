# Neurotrack AI Agent Instructions (Core)

This file intentionally keeps only repository-wide guidance to minimize context size.
Detailed, domain-specific operating rules are delegated to custom agents in `.github/agents/*.agent.md`.

## Project Snapshot

- Neurotrack is a reinforcement learning system for automated neuron tracing from microscopy images.
- The primary workflow uses a Soft Actor-Critic (SAC) agent and `neurotrack.data.datasets.NeuronPatchDataset`.
- Core CLI workflows:
    - `python -m neurotrack.cli.setup_sac_train`
    - `python -m neurotrack.cli.run_sac_train`
    - `python -m neurotrack.cli.run_inference`
    - `python -m neurotrack.cli.interactive_tracing`

## Current Preferences and Status

- Prefer `deeper_cnn`; `cnn` is deprecated.
- Branch-classifier modules are currently not in the main training path:
    - `neurotrack.models.resnet`
    - `neurotrack.models.resblock`
    - `neurotrack.cli.classifier_train`
    - `neurotrack.cli.collect_branch_data`
- `neurotrack.data.neuron_data` is not currently used in the main training path.

## Global Engineering Rules (All Agents)

1. Prioritize simplicity and readability over over-engineering.
2. Preserve behavior unless the user explicitly asks for behavior changes.
3. Keep clear layer boundaries across `core`, `data`, `environments`, `models`, `training`, `inference`, `evaluation`, `visualization`, `pipelines`, and `cli`.
4. Avoid circular dependencies and hidden coupling; move shared utilities to `neurotrack/core`.
5. Use explicit imports; do not use wildcard imports.
6. Keep `__init__.py` files minimal and avoid heavy import side effects.
7. Avoid global mutable state; pass configuration/state explicitly.
8. Document public classes and functions with clear docstrings.
9. Keep tests under top-level `tests/` (not inside `neurotrack/`).
10. Keep notebooks in `notebooks/`; do not place notebooks in `docs/source/`.

## Repository Structure Expectations

- The canonical runtime package is top-level `neurotrack/` (not nested under `src/`).
- Root should include packaging/project files (`README.md`, `LICENSE`, `requirements.txt`, and `pyproject.toml` or `setup.py`) plus `docs/`, `tests/`, and `configs/`.

## Anti-Patterns to Flag

- Circular dependencies
- Hidden coupling across unrelated modules
- Overloaded `__init__.py` files
- Ravioli code (duplicated/ambiguous module responsibilities)
- Variable name reuse across different types
- Missing packaging file (`pyproject.toml` or `setup.py`)

## Delegated Specialized Instructions

Use these custom agents for detailed, task-specific instructions:

- `architect` → `.github/agents/architect.agent.md`
- `refactor` → `.github/agents/refactor.agent.md`
- `gui` → `.github/agents/gui.agent.md`
- `data-pipeline` → `.github/agents/data-pipeline.agent.md`
- `inference-eval` → `.github/agents/inference-eval.agent.md`
- `qa-tests` → `.github/agents/qa-tests.agent.md`

When a request is domain-specific, follow the matching custom agent file rather than expanding this core file.