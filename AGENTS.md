# AGENTS.md

This file applies to the entire repository.

Keep this file short. Read extra instruction files only when they are relevant to the task.

## General Behavior

- Prefer simple, root-cause fixes over layered workarounds.
- Do not overcomplicate plotting, notebook, or layout code.
- When something looks wrong, explain the mechanism briefly, then fix the underlying cause
- Keep the two project workflows conceptually separate.
- Prefer shared helpers over repeated local logic.
- Verify the thing you changed:
  - Python behavior: run or update focused tests when feasible.
  - Notebook changes: execute the notebook after editing it.
- If physical units are available for plots, show them directly and clearly.
- Keep instruction-token usage low: open only the relevant companion file below.

## Instruction Map

- General workflow style:
  - `agent_docs/general_behavior.md`
- Project workflows, result folders, and launchers:
  - `agent_docs/project_structure.md`
- Plotting and notebook expectations:
  - `agent_docs/plotting_notebooks.md`
- Python runtime, testing mode, and Moran's I notes:
  - `agent_docs/python_runtime.md`
