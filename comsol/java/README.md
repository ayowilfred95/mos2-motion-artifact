# Running the COMSOL models from compiled Java classes

The three classes build, mesh, solve and save the models. They read their parameters from the
`params_*.txt` / `vars_chain.txt` files in this folder, which `export_params.py` generates from
`comsol/params.py`. Rerun that script only if `PARAMS` in `motion_artifact_sim.py` change.

| class | model file | writes (in `comsol/data/java/`) |
|---|---|---|
| `MechModel` | `comsol/mos2_patch_mechanics.mph`, strain images in `comsol/figures/` | `raw_mech.json`, `epk.txt` |
| `ChainModel` (needs `epk.txt` from MechModel) | `comsol/mos2_sensor_chain.mph` | `raw_chain.json`, `sweeps.txt`, `time_*.txt`, `motion.txt` |
| `DebyeModel` (optional, needs the AC/DC module) | `comsol/debye_1d.mph` | `raw_debye.json` |

## Commands (run from the project folder `mos2_motion_artifact`)

COMSOL's `bin` folder must be on the PATH, or type the full path, for example
`/Applications/COMSOL60/Multiphysics/bin/comsol` (macOS),
`"C:\Program Files\COMSOL\COMSOL60\Multiphysics\bin\win64\comsolcompile.exe"` (Windows) or
`/usr/local/comsol60/multiphysics/bin/comsol` (Linux).

```
comsolcompile comsol/java/MechModel.java
comsolcompile comsol/java/ChainModel.java
comsolcompile comsol/java/DebyeModel.java

comsol batch -inputfile comsol/java/MechModel.class
comsol batch -inputfile comsol/java/ChainModel.class
comsol batch -inputfile comsol/java/DebyeModel.class
```
On macOS, `comsolcompile` is `comsol compile`. On Windows, use `comsolbatch.exe -inputfile ...`.

`MechModel` prints the mesh-convergence table when it finishes that step.

Alternatively, open a `.class` file in COMSOL Desktop (File > Open). It runs the same code and shows the
model when finished. The project folder is found from the `MOS2_ROOT` environment variable, the current
folder, or `~/Downloads/mos2_motion_artifact`, `~/mos2_motion_artifact`, `~/Desktop/...` or `~/Documents/...`.

## Post-processing (Python, no COMSOL needed)
```
python comsol/import_java.py
```
This converts the Java exports and writes `comsol/results.json`, `comsol/comparison.csv`, figs 3-8 in
`comsol/figures/` and `comsol/COMSOL_METHODS.md`.
