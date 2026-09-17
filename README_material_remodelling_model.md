# Cell-based model of material remodelling, macrophage phagocytosis and fusion

This repository contains the Python implementation of a cell-based
computational model developed to investigate the coupling between
material remodelling, macrophage phagocytosis, intracellular material
retention and macrophage fusion.

The model simulates:

-   stochastic material fragmentation at the hydrogel interface
-   macrophage recruitment and growth
-   material fragment uptake and intracellular material retention
-   viscoelastic relaxation-driven cell spreading
-   contact-dependent macrophage fusion

The model was developed for mechanistic investigation of how distinct
hydrogel network properties regulate macrophage fusion behaviour.

## Model overview

    Hydrogel interface
            |
            ↓
    Stochastic material fragmentation
            |
            ↓
    Macrophage phagocytosis
            |
            ↓
    Intracellular material retention tracking
            |
            ↓
    Maxwell viscoelastic cell spreading
            |
            ↓
    Macrophage fusion
            |
            ↓
    Post-simulation intracellular retention analysis

Macrophages are represented as deformable cellular agents with dynamic
geometry, nuclear information and material uptake history. During cell
fusion, cellular geometry, nuclear number and intracellular material
retention history are conserved.

## Requirements

-   Python \>= 3.9
-   NumPy
-   SciPy
-   Shapely
-   Matplotlib
-   imageio

Install dependencies:

``` bash
pip install -r requirements.txt
```

## Running the simulation

Default:

``` bash
python simulation_material.py
```

Select material condition:

``` bash
python simulation_material.py --material SA
python simulation_material.py --material UC
python simulation_material.py --material CC
```

Additional options:

``` bash
python simulation_material.py --help
```

## Output

Results are saved to:

    results/
    ├── SA/
    ├── UC/
    └── CC/

Outputs include:

-   time-dependent cell-material interaction snapshots
-   simulation animation
-   intracellular material retention analysis

## Model parameters

The model includes three hydrogel conditions:

  Material   Network characteristic
  ---------- -----------------------------------------------------------------
  SA         Highly remodelable network with enhanced fragment accessibility
  UC         Intermediate physical network remodelling
  CC         Stable covalent network with limited remodelling

Parameters regulate:

-   fragmentation probability
-   fragment size distribution
-   viscoelastic relaxation behaviour
-   intracellular material processing kinetics

Parameter values represent qualitative differences between material
systems and are not fitted to experimental datasets.

## Reproducibility

All stochastic processes are controlled by a fixed random seed:

    seed = 2

The model preserves:

-   material uptake history during cell fusion
-   nuclear information during fusion events
-   intracellular material retention dynamics after phagocytosis

## Code organization

    simulation_material.py
        Main simulation workflow

        ├── Material fragmentation
        ├── Cell recruitment
        ├── Fragment uptake
        ├── Cell spreading
        ├── Cell fusion
        └── Intracellular retention module

    requirements.txt
        Python dependencies

## Scientific description

This model provides a mechanistic framework linking hydrogel network
remodellability with macrophage behaviour. Material fragmentation
controls phagocytic accessibility, while intracellular material
retention influences cell morphology and fusion probability.

## Citation

If you use this code, please cite the associated manuscript.
