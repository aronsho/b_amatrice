# Project Structure

Load this file when working on training, validation, test, launchers, or result-folder logic.

## Workflows

Keep these workflows conceptually separate:

1. `a_training_1.py` -> `b_validate.py`
2. `c_training.py` -> `d_test.py`

Plotting and summaries should reflect those workflows explicitly.

## Result Folders

- Canonical layout for new runs:
  - `results/a_training/<run_name>`
  - `results/validation_legacy/<run_name>`
  - `results/validation/<run_name>`
  - `results/training/<run_name>`
  - `results/test/<run_name>`
- Older flat folders such as `results/training_20260504` still exist and should remain readable.

When editing launchers or plotting code, keep these mappings consistent.

## Launchers

- Safe local launchers are preferred.
- Default launcher behavior should remain conservative:
  - serial execution by default
  - skip existing results by default
  - lightweight local test mode by default
- Training launchers:
  - `e_train_run.py` for `a_training_1.py`
  - `e_train_run2.py` for `c_training.py`
- Validation and test launchers:
  - `e_run.py` for `b_validate.py`
  - `e_run2.py` for `d_test.py`
- Shared launcher logic lives in `e_runner_common.py`.
  - Prefer changing shared logic there instead of duplicating code.
