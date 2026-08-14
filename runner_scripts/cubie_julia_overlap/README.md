# Direct Cubie ↔ DiffEqGPU benchmark suite

Run from the repository root:

```text
python run_cubie_julia_overlap.py -a performance -n 16777216
python run_cubie_julia_overlap.py -a numerical -p cubie
python run_cubie_julia_overlap.py -a performance --from-n 2048
python run_cubie_julia_overlap.py -a performance -n 32768,134217728 -p julia
python run_cubie_julia_overlap.py --algorithm kvaerno5 -p cubie
```

CSVs are written to `data/cubie_julia_overlap/<dataset-key>/`, one directory
per GPU and OS; figures to `plots/<dataset-key>/overlap_*.png` and the report
to `plots/<dataset-key>/cubie_julia_overlap.md`, the same group directory the
other analyzers write to. A run replaces the rows it regenerates and leaves
the rest:

- `-a, --analysis` selects the analysis: `performance`, `numerical`, `work-precision`, `all`.
- `-p, --package` selects the package: `cubie`, `julia`, `all`.
- `-n, --nmax` is a sweep ceiling (8, 32, ... <= n) or a comma list of exact trajectory counts.
- `--from-n` restarts the performance analysis at that N; lower-N rows stay.
- `--algorithm` runs one row of `algorithms.csv`; other rows stay.

`manifest.json` records the commands of the last run. Protocol settings live in
`common.py` (numerical grids and adaptive pins shared with the NE suite's
`ne_common.py`), mirrored in `julia_worker.jl`. The CSVs record the analysis
in a `phase` column, where `work-precision` is written `work_precision`.

The `numerical` phase's cubie side is imported from
`data/numerical_equivalence/cubie/<dataset-key>/` — run
`run_numerical_equivalence` first. The DiffEqGPU side solves untimed, one
pass per point. erk-family rows run no fixed numerical sweep. The `pi`
comparison tier is skipped where it equals cubie's shipped defaults.

The executable overlap table is `algorithms.csv`. The complete eight-method
DiffEqGPU ODE inventory, including the three exclusions, is
`diffeqgpu_ode_inventory.csv`; `GPUEM` and `GPUSIEA` are SDE-only and outside
this suite. The Julia worker requires the root Julia project to be
instantiated already and never runs package setup.

A worker records individual point failures, continues through the remaining
sweep, then exits nonzero so the launcher marks the run incomplete. Every
analysis records finite/failed trajectory counts; invalid or missing-validity
points are excluded from timing summaries and speedups. Re-run the analyzer
alone with:

```text
python runner_scripts/cubie_julia_overlap/analyze.py
```

Pass `--key <os>_<gpu>` to redraw another machine's results.
