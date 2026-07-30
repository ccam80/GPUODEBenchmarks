# Direct Cubie ↔ DiffEqGPU benchmark suite

Run from the repository root:

```text
python run_cubie_julia_overlap.py --profile full
python run_cubie_julia_overlap.py --phase performance --nmax 16777216
python run_cubie_julia_overlap.py --phase numerical --framework cubie
python run_cubie_julia_overlap.py --phase performance --from-n 2048
```

Results are written to `data/cubie_julia_overlap/<dataset-key>/`, one
directory per GPU and OS. A run replaces the rows it regenerates and leaves
the rest:

- `--phase` selects the leg: `performance`, `numerical`, `work_precision`, `all`.
- `--framework` selects the package: `cubie`, `julia`, `both`.
- `--from-n` continues the performance leg at that N, keeping the rows below it.
- `--profile smoke` runs a reduced protocol across all five algorithms and every
  metric family; `--profile full` runs the published protocol.
- `--nmax` caps the performance leg.

`manifest.json` records the commands of the last run. Protocol settings — dt,
tolerance, repeat counts, and the dt/tolerance grids — are constants in
`common.py`, mirrored in `julia_worker.jl`.

The executable overlap table is `algorithms.csv`. The complete eight-method
DiffEqGPU ODE inventory, including the three exclusions, is
`diffeqgpu_ode_inventory.csv`; `GPUEM` and `GPUSIEA` are SDE-only and outside
this suite. The Julia worker requires the root Julia project to be
instantiated already and never runs package setup.

A worker records individual point failures, continues through the remaining
sweep, then exits nonzero so the launcher marks the run incomplete. Every
phase records finite/failed trajectory counts; invalid or missing-validity
points are excluded from timing summaries and speedups. Re-run the analyzer
alone with:

```text
python runner_scripts/cubie_julia_overlap/analyze.py --output data/cubie_julia_overlap/<dataset-key>
```
