# Fabbri-Linder CuBIE / Myokit-CUDA comparison

Run this script with the Myokit-CUDA environment and point it at the CuBIE
environment's Python executable:

```text
GPU_ODE_MYOKIT_CUDA/venv/Scripts/python.exe runner_scripts/cubie_myokit_fabbri/compare_fabbri.py --cellml C:/local_working_projects/cubie/tests/fixtures/cellml/Fabbri_Linder.cellml --cubie-python C:/local_working_projects/cubie/.venv/Scripts/python.exe
```

The default scaling run compares 512, 2,048, 8,192, and 32,768 independent
trajectories for 1,000 float32 forward-Euler steps of `1e-5` seconds. Use
`--trajectory-counts 512 2048 8192 32768 131072` for the extended scaling
set, or `--trajectories 64` for a single smoke point. Each timing point uses
100 repeats by default, matching the main timing benchmark protocol. The
Myokit exporter output and CuBIE both use the original CellML model without
singularity rewriting. Compilation and one warmup per count are excluded,
and both measured paths synchronize the CUDA stream.

The source fixture is not modified. For Myokit's stricter CellML importer,
the runner creates a temporary normalized copy that marks the three
Ca-buffering variables used by cAMP as public outputs and merges the
oppositely ordered duplicate ATPi/cAMP connections. The runner asserts the
exact known source structure before applying these metadata-only repairs and
records them in the report.

Results are written below `data/Fabbri_Myokit_CUDA/<os>_<gpu>/`, with
per-count detail under `N_<count>/` and aggregate `scaling.csv` plus a
top-level Markdown timing table. Per-count detail includes:

- every timing sample;
- per-state accuracy statistics;
- a machine-readable JSON report; and
- a Markdown comparison report.

Full final-state matrices remain in memory only; they are not duplicated to
disk for every scaling count.
