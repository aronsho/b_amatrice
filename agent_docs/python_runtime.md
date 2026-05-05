# Python Runtime

Load this file when editing Python logic, runtime parameters, or Moran's I behavior.

## Job Index Handling

- Scripts should support both:
  - `sys.argv[1]`
  - `SLURM_ARRAY_TASK_ID`
- Use `functions.general_functions.resolve_job_index(...)` instead of open-coding this logic.

## Testing Mode

- Lightweight local testing uses:
  - `space_realizations = 10`
  - `time_realizations = 5`
- `min_count` should scale consistently with the realization count.
- Use `functions.general_functions.resolve_realization_settings(...)`.

## Performance Notes

- The major runtime win came from caching repeated time min, max, and span computations in `mac_spacetime`.
- Runtime is roughly proportional to `space_realizations * time_realizations`.
- The temporal Moran optimization primarily helps memory and scaling on larger jobs.

## Moran's I

- `est_morans_i_temporal(...)` must not crash on sparse edge cases.
- If there are no valid temporal pairs, return `nan`.
- If variance is zero, return `nan`.
- Keep regression coverage in `tests/test_space_time_separated_map.py`.

## Python Changes

- When changing Python logic, add focused unit tests whenever feasible.
- Keep code concise, but do not sacrifice readability or correctness.
- Prefer small shared helpers over repeated ad hoc logic.
