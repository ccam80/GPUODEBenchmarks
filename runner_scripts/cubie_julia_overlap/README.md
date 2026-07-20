# Direct Cubie ↔ DiffEqGPU benchmark suite

Run from the repository root:

```text
python run_cubie_julia_overlap.py --profile smoke
python run_cubie_julia_overlap.py --profile full --nmax 16777216
```

The executable overlap table is `algorithms.csv`. The complete eight-method
DiffEqGPU ODE inventory, including the three exclusions, is
`diffeqgpu_ode_inventory.csv`; `GPUEM` and `GPUSIEA` are SDE-only and outside
this suite. Every run is isolated beneath
`data/cubie_julia_overlap/<dataset-key>/<run-id>/`; its `manifest.json`
contains the exact commands and protocol overrides. The Julia worker assumes
the root Julia project is already instantiated and never runs package setup in
the timed workflow. Use `--help` for phase/framework/repeat overrides.

Smoke executes all five algorithms but reduces the grids and ensemble sizes.
Full executes the complete performance, numerical-equivalence, and
work-precision protocols. A worker records individual point failures and
continues through the remaining sweep, then exits nonzero so the launcher
marks the run incomplete. Every phase records finite/failed trajectory counts;
invalid or missing-validity points are excluded from timing summaries and
speedups. Re-run the analyzer alone with:

```text
python runner_scripts/cubie_julia_overlap/analyze.py --output <run-directory>
```
