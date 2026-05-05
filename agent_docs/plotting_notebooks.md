# Plotting And Notebooks

Load this file when editing notebooks, plotting helpers, or result visualizations.

## Notebook Scope

- Always test notebooks after changing them.
- The main plotting notebook is `2026_Amatrice_plotresults.ipynb`.

## Plotting Requirements

- Show `a_training + validation` together.
- Show `training + test` together.
- Keep legacy validation distinct when it is plotted.
- Do not plot `mac_time`.
- MAC colormap: `plasma`.
- IG colormap: `coolwarm`.
- Colorbars should be slim and neatly aligned.
- Include physical length and time scales on axes when available.

## Plotting Style

- Prefer simple heatmaps on a shared matrix grid.
- Use physical km and day values as the visible tick labels when available.
- Side-by-side panels should keep the same visual box size.
- Avoid aspect rules, secondary axes, or layout tricks that make one panel visibly smaller than the others.
- Prefer one clear shared heatmap helper over figure-specific hacks.
- If a dataset is empty, show that clearly instead of suggesting a real pattern.
