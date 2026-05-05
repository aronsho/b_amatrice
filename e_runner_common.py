import argparse
import itertools as it
import os
import subprocess
import sys
import time
from pathlib import Path

from functions.result_paths import (
    RESULTS_DIR_ENV_VAR,
    RUN_NAME_ENV_VAR,
    default_run_name,
    ensure_results_dir,
)


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--max-parallel", type=int, default=1,
                        help="Maximum number of local jobs to run at once.")
    parser.add_argument("--start-index", type=int, default=0,
                        help="First job index to consider.")
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Exclusive end job index. Defaults to the end of the grid.")
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=0.5,
        help="Sleep interval while waiting for worker slots.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for child jobs.")
    parser.add_argument(
        "--full-mode",
        action="store_true",
        help="Disable lightweight test mode and use the script defaults.")
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip jobs whose output file already exists.")
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Run jobs even if the output file already exists.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the jobs that would run without launching them.")
    parser.add_argument(
        "--space-realizations",
        type=int,
        default=None,
        help="Optional override passed through the environment.")
    parser.add_argument(
        "--time-realizations",
        type=int,
        default=None,
        help="Optional override passed through the environment.")
    parser.add_argument(
        "--min-count",
        type=int,
        default=None,
        help="Optional override passed through the environment.")
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Run identifier used for the results folder. Reuse the same value "
            "across related workflows when you want them grouped together."
        ),
    )
    return parser


def iter_combinations(n_time_list: list[int], n_space_list: list[int]):
    return list(it.product(n_time_list, n_space_list))


def output_path(
        results_dir: Path,
        output_prefix: str,
        n_time: int,
        n_space: int) -> Path:
    return results_dir / f"{output_prefix}_n_time{n_time}_n_space{n_space}.csv"


def wait_for_slot(
        running: list[tuple[int, subprocess.Popen]],
        poll_seconds: float,
        failures: list[tuple[int, int]]) -> list[tuple[int, subprocess.Popen]]:
    while running:
        still_running = []
        for job_idx, process in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((job_idx, process))
            elif return_code != 0:
                failures.append((job_idx, return_code))
        if len(still_running) != len(running):
            return still_running
        time.sleep(poll_seconds)
    return running


def child_env(
        args: argparse.Namespace,
        job_idx: int,
        results_dir: Path,
        run_name: str) -> dict[str, str]:
    env = os.environ.copy()
    env["SLURM_ARRAY_TASK_ID"] = str(job_idx)
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    env[RESULTS_DIR_ENV_VAR] = str(results_dir)
    env[RUN_NAME_ENV_VAR] = run_name
    if not args.full_mode:
        env["B_AMATRICE_TEST_MODE"] = "1"
    if args.space_realizations is not None:
        env["B_AMATRICE_SPACE_REALIZATIONS"] = str(args.space_realizations)
    if args.time_realizations is not None:
        env["B_AMATRICE_TIME_REALIZATIONS"] = str(args.time_realizations)
    if args.min_count is not None:
        env["B_AMATRICE_MIN_COUNT"] = str(args.min_count)
    return env


def run_local_grid(
        *,
        description: str,
        n_time_list: list[int],
        n_space_list: list[int],
        workflow: str,
        script_path: str,
        output_prefix: str) -> int:
    args = build_parser(description).parse_args()
    combinations = iter_combinations(n_time_list, n_space_list)
    end_index = len(combinations) if args.end_index is None else args.end_index

    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1.")
    if args.start_index < 0 or end_index > len(combinations):
        raise ValueError("Requested job index range is out of bounds.")
    if args.start_index >= end_index:
        raise ValueError("start-index must be smaller than end-index.")

    run_name = args.run_name or default_run_name()
    results_root = ensure_results_dir(workflow, run_name)
    command = [args.python, script_path]
    mode_label = "full" if args.full_mode else "test"
    running: list[tuple[int, subprocess.Popen]] = []
    failures: list[tuple[int, int]] = []
    launched = 0
    skipped = 0

    print(f"workflow={workflow} run={run_name} results={results_root}")

    for job_idx in range(args.start_index, end_index):
        n_time, n_space = combinations[job_idx]
        destination = output_path(results_root, output_prefix, n_time, n_space)
        if args.skip_existing and destination.exists():
            print(f"skip job {job_idx}: {destination} already exists")
            skipped += 1
            continue

        print(f"launch job {job_idx}: n_time={n_time} n_space={n_space} "
              f"mode={mode_label}")
        launched += 1
        if args.dry_run:
            continue

        while len(running) >= args.max_parallel:
            running = wait_for_slot(running, args.poll_seconds, failures)
        running.append(
            (
                job_idx,
                subprocess.Popen(
                    command,
                    env=child_env(args, job_idx, results_root, run_name),
                ),
            )
        )

    while running:
        running = wait_for_slot(running, args.poll_seconds, failures)

    print(f"launched={launched} skipped={skipped} failures={len(failures)}")
    for job_idx, return_code in failures:
        print(f"job {job_idx} failed with exit code {return_code}")
    return 1 if failures else 0
