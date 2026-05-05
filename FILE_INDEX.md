# File Index

This document lists the main repository files and their purpose.

It is intended as a quick navigation aid for both humans and agents.

## Top Level Docs

- [AGENTS.md](/Users/aron/polybox/Projects/b_amatrice/AGENTS.md)
  Repo-scoped Codex instructions for this project.

- [COLLABORATION_NOTES.md](/Users/aron/polybox/Projects/b_amatrice/COLLABORATION_NOTES.md)
  Project-specific working notes gathered during the current collaboration.

- [E_RUN_OPTIONS.md](/Users/aron/polybox/Projects/b_amatrice/E_RUN_OPTIONS.md)
  Reference for the local `e_*run*.py` launcher scripts, including which workflow each one launches and the shared CLI options such as `--full-mode`, index ranges, parallelism, skipping, and run naming.

- [FILE_INDEX.md](/Users/aron/polybox/Projects/b_amatrice/FILE_INDEX.md)
  This file. High-level map of the repository.

- [LICENSE](/Users/aron/polybox/Projects/b_amatrice/LICENSE)
  License file for the repository.

## Data Preparation

- [a_preparecatalog.py](/Users/aron/polybox/Projects/b_amatrice/a_preparecatalog.py)
  Reads the raw Amatrice catalog, filters it, transforms coordinates, splits it into training and testing sets, and writes the processed CSV files under `data/training` and `data/testing`.

## Main Analysis Scripts

- [a_training_1.py](/Users/aron/polybox/Projects/b_amatrice/a_training_1.py)
  Pre-training stage for the `a_training -> validation` workflow. Runs `mac_spacetime` on the training portion of the training catalog and writes spatial-MAC summary CSVs to `results/a_training_20260504`.

- [b_validate.py](/Users/aron/polybox/Projects/b_amatrice/b_validate.py)
  Validation stage paired with `a_training_1.py`. Uses the training catalog split into train/validation subsets, computes forecasting scores, and writes validation summary CSVs to `results/validation_20260504`.

- [c_training.py](/Users/aron/polybox/Projects/b_amatrice/c_training.py)
  Training stage for the `training -> test` workflow. Runs `mac_spacetime` on the full training catalog and writes spatial-MAC summary CSVs to `results/training_20260504`.

- [d_test.py](/Users/aron/polybox/Projects/b_amatrice/d_test.py)
  Testing stage paired with `c_training.py`. Uses the training catalog plus held-out test catalog, computes forecasting scores, and writes summary CSVs to `results/test_20260504`.

## Alternative / Legacy Analysis Scripts

- [b_validate_pos.py](/Users/aron/polybox/Projects/b_amatrice/b_validate_pos.py)
  Alternative validation script using magnitude differences / positive-event style logic. Writes to a separate validation output folder.

- [b_validate_legacy.py](/Users/aron/polybox/Projects/b_amatrice/b_validate_legacy.py)
  Legacy validation pipeline that works directly from the raw catalog rather than the current processed split-data workflow.

## Local Launcher Scripts

- [e_runner_common.py](/Users/aron/polybox/Projects/b_amatrice/e_runner_common.py)
  Shared helper used by all local launcher scripts. Handles argument parsing, safe defaults, skipping existing outputs, environment-variable setup, and local process orchestration.

- [e_train_run.py](/Users/aron/polybox/Projects/b_amatrice/e_train_run.py)
  Safe local launcher for `a_training_1.py`.

- [e_run.py](/Users/aron/polybox/Projects/b_amatrice/e_run.py)
  Safe local launcher for `b_validate.py`.

- [e_train_run2.py](/Users/aron/polybox/Projects/b_amatrice/e_train_run2.py)
  Safe local launcher for `c_training.py`.

- [e_run2.py](/Users/aron/polybox/Projects/b_amatrice/e_run2.py)
  Safe local launcher for `d_test.py`.

- [e_run_legacy.py](/Users/aron/polybox/Projects/b_amatrice/e_run_legacy.py)
  Safe local launcher for the legacy validation script `b_validate_legacy.py`.

## Plotting / Exploration Notebooks

- [2026_Amatrice_plotresults.ipynb](/Users/aron/polybox/Projects/b_amatrice/2026_Amatrice_plotresults.ipynb)
  Main maintained results notebook. Loads CSV outputs automatically and plots the two workflows separately:
  - `a_training -> validation`
  - `training -> test`

- [2026_Amatrice_train.ipynb](/Users/aron/polybox/Projects/b_amatrice/2026_Amatrice_train.ipynb)
  Exploratory notebook for training-side analysis and earlier plotting experiments.

- [2026_Amatrice_test.ipynb](/Users/aron/polybox/Projects/b_amatrice/2026_Amatrice_test.ipynb)
  Exploratory notebook for test-side data inspection and analysis.

- [2026_Amatrice_plots.ipynb](/Users/aron/polybox/Projects/b_amatrice/2026_Amatrice_plots.ipynb)
  Older general-purpose plotting notebook for visual inspection of the catalogs and intermediate results.

- [202601_Amatrice_plotresults.ipynb](/Users/aron/polybox/Projects/b_amatrice/202601_Amatrice_plotresults.ipynb)
  Older or alternate result-plotting notebook kept for reference. Not the main maintained plotting notebook.

## Python Package: `functions/`

- [functions/__init__.py](/Users/aron/polybox/Projects/b_amatrice/functions/__init__.py)
  Package marker for the `functions` module namespace.

- [functions/general_functions.py](/Users/aron/polybox/Projects/b_amatrice/functions/general_functions.py)
  General utilities used across the project:
  - likelihood functions
  - Welford running mean/variance helpers
  - simulation helpers
  - local test-mode settings
  - job-index resolution

- [functions/main_functions.py](/Users/aron/polybox/Projects/b_amatrice/functions/main_functions.py)
  High-level evaluation helpers used in validation/test scripts, especially the log-likelihood comparison functions.

- [functions/eval_functions.py](/Users/aron/polybox/Projects/b_amatrice/functions/eval_functions.py)
  Statistical helpers for Moran’s I interpretation, such as expected mean/variance, p-values, and z-values.

- [functions/transformation_functions.py](/Users/aron/polybox/Projects/b_amatrice/functions/transformation_functions.py)
  Geographic coordinate transformation utilities:
  - spherical to Cartesian conversion
  - translation / rotation
  - alignment of the fault-section coordinate system

- [functions/space_functions.py](/Users/aron/polybox/Projects/b_amatrice/functions/space_functions.py)
  Spatial partition helpers:
  - Voronoi construction
  - nearest-node lookup
  - tile assignment
  - Voronoi cell volumes
  - neighborhood matrix construction

- [functions/one_dimensional.py](/Users/aron/polybox/Projects/b_amatrice/functions/one_dimensional.py)
  Older / more general one-dimensional autocorrelation and partitioning routines, including constant-value and random splitting helpers.

- [functions/space_map.py](/Users/aron/polybox/Projects/b_amatrice/functions/space_map.py)
  Spatial-only Moran’s I and map estimation routines based on Voronoi partitioning.

- [functions/space_time_separated_map.py](/Users/aron/polybox/Projects/b_amatrice/functions/space_time_separated_map.py)
  Core space-time analysis module. Contains:
  - `mac_spacetime`
  - the optimized temporal Moran’s I helper
  - logic for time slicing, spatial partitioning, aggregation, and MAC estimation

## Tests

- [tests/test_space_time_separated_map.py](/Users/aron/polybox/Projects/b_amatrice/tests/test_space_time_separated_map.py)
  Unit tests for the temporal Moran’s I shortcut and edge-case behavior in `space_time_separated_map.py`.

## Data Files

- [data/readme.md](/Users/aron/polybox/Projects/b_amatrice/data/readme.md)
  Minimal setup note about installing required Python packages.

- [data/catalogs/Amatrice_CAT5.v20210325](/Users/aron/polybox/Projects/b_amatrice/data/catalogs/Amatrice_CAT5.v20210325)
  Main raw Amatrice earthquake catalog used by the project.

- [data/catalogs/Amatrice_CAT4.v202012.1](/Users/aron/polybox/Projects/b_amatrice/data/catalogs/Amatrice_CAT4.v202012.1)
  Older raw catalog version kept in the repository.

- [data/catalogs/Amatrice2016_FMs_SKHASH_pPol_v2p0.dat](/Users/aron/polybox/Projects/b_amatrice/data/catalogs/Amatrice2016_FMs_SKHASH_pPol_v2p0.dat)
  Additional raw catalog / focal-mechanism-style data file kept in the repository.

- [data/training/Amatrice_CAT5_train.csv](/Users/aron/polybox/Projects/b_amatrice/data/training/Amatrice_CAT5_train.csv)
  Processed training split generated by `a_preparecatalog.py`.

- [data/testing/Amatrice_CAT5_test.csv](/Users/aron/polybox/Projects/b_amatrice/data/testing/Amatrice_CAT5_test.csv)
  Processed test split generated by `a_preparecatalog.py`.

## Repository Hygiene Files

- [data/catalogs/.gitignore](/Users/aron/polybox/Projects/b_amatrice/data/catalogs/.gitignore)
  Git ignore rules for the raw catalog directory.

- [data/training/.gitignore](/Users/aron/polybox/Projects/b_amatrice/data/training/.gitignore)
  Git ignore rules for processed training outputs in the data folder.

- [data/testing/.gitignore](/Users/aron/polybox/Projects/b_amatrice/data/testing/.gitignore)
  Git ignore rules for processed test outputs in the data folder.

## Not Indexed Here

The following are intentionally not documented file-by-file in this index:

- `env/`
  Local virtual environment.
- `results/`
  Generated run outputs; these are data products, not maintained source files.
- `__pycache__/`
  Python bytecode cache files.
