# Motion artifacts in flexible MoS2 FET cortisol sensors: simulation code and data

Supplementary code and data for *Separating Motion Artifacts from Biomarker Signals in Flexible MoS2 FET Cortisol
Sensors: A Dynamic Multiphysics and Finite-Element Simulation of Strain-Binding Cross-Talk and Mitigation Strategies*
(A. Adeyemi, V. Ogunyemi).

## Contents
| path | what |
|---|---|
| `motion_artifact_sim.py` | six-stage sensor chain (motion, strain, electrical, binding, false concentration, filtering / twin sensor); all parameters in `PARAMS` with sources |
| `make_figures.py` | Figs. 1 to 8 (`--source elmer` uses the finite-element strain) |
| `elmer/make_models.py` | writes every Elmer mesh (ElmerGrid `.grd`) and solver file (ElmerSolver `.sif`) |
| `elmer/run_elmer.py` | mesh convergence, all designs, sensor chain, `elmer/results.json`, `elmer/comparison.csv`, figures |
| `elmer/island_sweep.py`, `elmer/plot_island_sweep.py` | rigid-island PDMS-thickness sweep (Table 7, Fig. 9b) |
| `elmer/run_motions.py` | time-stepped (Scanning) wrist, elbow, walking, PET and island motions |
| `elmer/export_images.py` | strain-field images (ParaView `pvpython`) |
| `elmer/models/<design>/` | `.grd` and `case.sif` of each model |
| `elmer/data/`, `elmer/results.json`, `elmer/comparison.csv` | all generated data |
| `fem/` | independent scikit-fem cross-check of the laminate mechanics |
| `elmer/revision_runs.py` | PET and rigid-island mesh convergence (Tables E1, E2); rigid island under skin-imposed curvature (Table E3) |
| `revision_analysis.py` | twin-sensor worked example (Appendix A), worked examples (Table B1), causal vs zero-phase filtering and mismatch cases (Tables C1, C2), sensitivity and robustness (Tables D1, D2), Fig. 7(c) |

## Requirements
- Python 3.9 or later with the packages in `requirements.txt`
- Elmer FEM 9.0 (ElmerSolver 26.2) and ParaView 5.11 for the finite-element runs and images

The scripts call ElmerGrid and ElmerSolver through `orb -m elmer` (an Ubuntu 24.04 virtual machine named `elmer`).
On a Linux machine with Elmer installed, change the `elmer()` function in `elmer/run_elmer.py` to run the
command directly.

## Reproduce
```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python elmer/run_elmer.py        # static models, chain, results.json, figures 3-8
.venv/bin/python elmer/island_sweep.py     # PDMS sweep
.venv/bin/python elmer/run_motions.py      # motion runs
pvpython elmer/export_images.py            # strain-field images
.venv/bin/python elmer/revision_runs.py    # convergence and skin-loading runs
.venv/bin/python revision_analysis.py      # appendix analyses and Fig. 7(c)
```
A full rebuild on a freshly installed system reproduces every number in `elmer/results.json` exactly.
