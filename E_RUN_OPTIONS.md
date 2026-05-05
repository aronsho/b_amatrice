# Local Launcher Options

This note explains how the local `e_*run*.py` launcher scripts work and which CLI options they accept.

## Which launcher does what

All of these launchers use the same shared logic from `e_runner_common.py`, so they all accept the same options.

| Launcher | Runs | Workflow name | Results folder |
| --- | --- | --- | --- |
| `e_train_run.py` | `a_training_1.py` | `pretraining` | `results/a_training/<run_name>/` |
| `e_run.py` | `b_validate.py` | `validation` | `results/validation/<run_name>/` |
| `e_train_run2.py` | `c_training.py` | `training` | `results/training/<run_name>/` |
| `e_run2.py` | `d_test.py` | `test` | `results/test/<run_name>/` |
| `e_run_legacy.py` | `b_validate_legacy.py` | `legacy_validation` | `results/validation_legacy/<run_name>/` |

## Shared defaults

- Jobs run serially by default: `--max-parallel 1`
- Existing result files are skipped by default: `--skip-existing`
- Lightweight local test mode is enabled by default
- The parameter grid is always the same:
  - `n_time = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]`
  - `n_space = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]`
- Total jobs per launcher: `10 * 12 = 120`

The grid order comes from `itertools.product(n_time_list, n_space_list)`, so job indices advance across `n_space` first:

- job `0` -> `n_time=2`, `n_space=1`
- job `1` -> `n_time=2`, `n_space=2`
- ...
- job `11` -> `n_time=2`, `n_space=2048`
- job `12` -> `n_time=4`, `n_space=1`

## Main options

- `--max-parallel N`
  Run up to `N` local child jobs at once. Default: `1`.

- `--start-index N`
  First job index to consider. Default: `0`.

- `--end-index N`
  Exclusive end index. If omitted, the launcher goes to the end of the grid.

- `--poll-seconds N`
  Sleep interval while waiting for free worker slots. Default: `0.5`.

- `--python PATH`
  Python executable used for child jobs. Default: the interpreter that launched the launcher script.

- `--full-mode`
  Disable lightweight local test mode and use the script defaults.

- `--skip-existing`
  Skip jobs whose output CSV already exists. This is the default.

- `--no-skip-existing`
  Re-run jobs even if the output CSV already exists.

- `--dry-run`
  Print which jobs would be launched, but do not actually start them.

- `--space-realizations N`
  Override `B_AMATRICE_SPACE_REALIZATIONS` for child jobs.

- `--time-realizations N`
  Override `B_AMATRICE_TIME_REALIZATIONS` for child jobs.

- `--min-count N`
  Override `B_AMATRICE_MIN_COUNT` for child jobs.

- `--run-name NAME`
  Put outputs into a specific results subfolder and reuse that name across related runs.

## What `--full-mode` changes

Without `--full-mode`, the launcher sets `B_AMATRICE_TEST_MODE=1` for child jobs.

That changes the realization settings in `functions/general_functions.py`:

- `space_realizations` becomes `10`
- `time_realizations` becomes `5`
- `min_count` is scaled down accordingly unless you override it explicitly

With `--full-mode`, each worker script uses its normal defaults instead:

- `space_realizations = 40`
- `time_realizations = 20`
- `min_count = 20`

You can still override any of those with:

- `--space-realizations`
- `--time-realizations`
- `--min-count`

## Examples

Show what would run for validation without launching anything:

```bash
python3 e_run.py --dry-run
```

Run only the first 10 validation jobs:

```bash
python3 e_run.py --end-index 10
```

Run validation jobs 24 through 35:

```bash
python3 e_run.py --start-index 24 --end-index 36
```

Run training with 4 local workers:

```bash
python3 e_train_run2.py --max-parallel 4
```

Run the full, non-lightweight test workflow:

```bash
python3 e_run2.py --full-mode
```

Force a re-run even if CSVs already exist:

```bash
python3 e_run.py --no-skip-existing
```

Group related runs under the same folder name:

```bash
python3 e_train_run.py --run-name 20260505
python3 e_run.py --run-name 20260505
```

Use custom realization settings without enabling full mode:

```bash
python3 e_run.py --space-realizations 20 --time-realizations 10 --min-count 10
```

## Notes

- The launcher creates a run folder automatically.
- When launched through an `e_*run*.py` script, the default `run_name` is a timestamp like `YYYYMMDD_HHMMSS`.
- If a worker script is run directly without a launcher, it falls back to a date-only run name like `YYYYMMDD`.
- Child jobs receive their job index through `SLURM_ARRAY_TASK_ID`, even for local launches.
- You can always inspect the built-in help with, for example:

```bash
python3 e_run.py --help
```
