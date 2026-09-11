# Cell–fragment interaction and fusion simulation

This repository contains the Python code used to simulate cell recruitment,
material-fragment generation and uptake, cell fusion, and
stress-relaxation-driven cell growth associated with the manuscript.

## Requirements

- Python >= 3.9
- NumPy
- SciPy
- Shapely
- Matplotlib
- imageio

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Running the simulation

From the repository root:

```bash
python simulation.py
```

Simulation snapshots and the animated GIF are written to:

```text
outputs/
```

The random seeds are fixed (`RANDOM_SEED = 2`) to facilitate reproducibility.

## Code organization

- `simulation.py` — simulation model and visualization
- `requirements.txt` — Python package dependencies
- `outputs/` — generated simulation snapshots and animation

## Reproducibility

The parameter values in `simulation.py` correspond to the parameterization
used for the manuscript analysis. The code has been reorganized for
readability and reproducibility without intentionally changing the
mathematical form of the original model.

## Citation

If you use this code, please cite the associated manuscript.
