from e_runner_common import run_local_grid


N_TIME_LIST = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
N_SPACE_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


if __name__ == "__main__":
    raise SystemExit(run_local_grid(
        description="Safely run validation jobs locally.",
        n_time_list=N_TIME_LIST,
        n_space_list=N_SPACE_LIST,
        workflow="validation",
        script_path="b_validate.py",
        output_prefix="valid",
    ))
