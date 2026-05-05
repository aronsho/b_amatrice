import os
import tempfile
import unittest
from pathlib import Path

from functions import result_paths


class TestResultPaths(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.original_root = result_paths.RESULTS_ROOT
        result_paths.RESULTS_ROOT = Path(self.tmpdir.name) / "results"
        result_paths.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        self.addCleanup(self.restore_root)

    def restore_root(self) -> None:
        result_paths.RESULTS_ROOT = self.original_root

    def test_ensure_results_dir_uses_canonical_nested_layout(self) -> None:
        path = result_paths.ensure_results_dir("training", "trial_alpha")
        self.assertEqual(
            path,
            result_paths.RESULTS_ROOT / "training" / "trial_alpha",
        )
        self.assertTrue(path.is_dir())

    def test_resolve_results_dir_prefers_explicit_env_directory(self) -> None:
        explicit = result_paths.RESULTS_ROOT / "custom-output"
        env = {result_paths.RESULTS_DIR_ENV_VAR: os.fspath(explicit)}
        resolved = result_paths.resolve_results_dir(
            "validation",
            env=env,
            fallback_run_name="ignored",
        )
        self.assertEqual(resolved, explicit)
        self.assertTrue(explicit.is_dir())

    def test_resolve_results_dir_resolves_relative_env_directory_from_project_root(self) -> None:
        env = {result_paths.RESULTS_DIR_ENV_VAR: "custom/nested"}
        resolved = result_paths.resolve_results_dir(
            "validation",
            env=env,
            fallback_run_name="ignored",
        )
        self.assertEqual(
            resolved,
            result_paths.PROJECT_ROOT / "custom" / "nested",
        )
        self.assertTrue(resolved.is_dir())

    def test_list_available_runs_includes_canonical_and_legacy_layouts(self) -> None:
        result_paths.ensure_results_dir("pretraining", "alpha")
        result_paths.ensure_results_dir("pretraining", "20260504")

        self.assertEqual(result_paths.list_available_runs("pretraining"), [
            "20260504",
            "alpha",
        ])

    def test_resolve_run_dir_lists_available_runs_on_error(self) -> None:
        result_paths.ensure_results_dir("training", "20260505")

        with self.assertRaisesRegex(
            FileNotFoundError,
            "Available runs: 20260505",
        ):
            result_paths.resolve_run_dir("training", "20260504")


if __name__ == "__main__":
    unittest.main()
