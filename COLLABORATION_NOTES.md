# Collaboration Notes

This file captures project-specific expectations and context learned during the current collaboration.

## Working Expectations

- Always test notebooks after changing them.
- When writing or changing Python logic, add focused unit tests whenever feasible.
- Keep code concise without losing readability or functionality.
- Prefer maintainable shared helpers over duplicated launcher logic.

## Workflow Structure

There are two distinct workflows and the plotting should reflect them separately:

1. `a_training` -> `validation`
2. `training` -> `test`

More concretely:

- Pre-training stage:
  - script: `a_training_1.py`
  - local launcher: `e_train_run.py`
  - result folder: `results/a_training_20260504`
- Validation stage:
  - script: `b_validate.py`
  - local launcher: `e_run.py`
  - result folder: `results/validation_20260504`
- Training stage:
  - script: `c_training.py`
  - local launcher: `e_train_run2.py`
  - result folder: `results/training_20260504`
- Test stage:
  - script: `d_test.py`
  - local launcher: `e_run2.py`
  - result folder: `results/test_20260504`

## Job Launching

- Local launchers should run safely by default.
- Default local behavior should be conservative:
  - serial execution unless the user explicitly increases parallelism
  - skip existing outputs when possible
  - lightweight test mode by default for the local launchers
- Scripts should accept job indices from either:
  - `sys.argv[1]`
  - `SLURM_ARRAY_TASK_ID`

This is handled via:

- `functions.general_functions.resolve_job_index`
- `functions.general_functions.resolve_realization_settings`

## Realization Settings

For lightweight local testing, use:

- `space_realizations = 10`
- `time_realizations = 5`

`min_count` should be scaled consistently with the realization count.

## Performance Findings

Main performance findings from this conversation:

- The biggest runtime win came from caching repeated global time min/max/span computations inside `mac_spacetime`.
- Runtime is roughly proportional to:
  - `space_realizations * time_realizations`
- Replacing the dense temporal Moran matrix with a direct temporal computation mainly helps:
  - memory scaling
  - larger jobs
- For small jobs, the dense-matrix replacement may not visibly speed up total runtime, but it still prevents cluster-scale memory blowups for larger cases.

## Moran's I

The temporal Moran shortcut in `functions/space_time_separated_map.py` must behave robustly in edge cases.

Important behaviors:

- if there are no valid temporal pairs, return `nan`
- if variance is zero, return `nan`
- do not crash with `ZeroDivisionError`

These behaviors are covered in:

- `tests/test_space_time_separated_map.py`

## Notebook Expectations

The main result notebook is:

- `2026_Amatrice_plotresults.ipynb`

Notebook requirements established in this conversation:

- it should be tested after edits
- it should load result CSVs automatically
- it should treat the two workflows separately:
  - pre-training with validation
  - training with test
- it should not plot `mac_time`
- plotting conventions should follow the original style more closely

## Plotting Conventions

Established plotting preferences:

- MAC plots use `plasma`
- IG plots use `coolwarm`
- colorbars should be slim and aligned neatly with the axes
- physical scales should be shown on the axes:
  - length scale
  - time scale
- the user prefers the cleaner style from the original plotting notebook over a dashboard-like style

## Testing Done In This Conversation

Validated during this collaboration:

- unit tests for temporal Moran consistency and edge cases
- launcher behavior for:
  - validation
  - test
  - pre-training
  - training
- notebook execution checks for `2026_Amatrice_plotresults.ipynb`

## Known Practical Context

- Some result folders may exist but be empty except for placeholder files like `x`.
- The plotting notebook should prefer non-empty folders when possible.
- Legacy training outputs may still exist in older folders such as `results/30042026`, but current pre-training should prefer `results/a_training_20260504` when available.

## Future Work Guidance

Before changing code in this repo:

1. Check which workflow the change belongs to.
2. Preserve the distinction between `a_training + validation` and `training + test`.
3. Keep launchers and result folder conventions consistent.
4. Add or update tests for Python logic.
5. Execute notebooks after editing them.
