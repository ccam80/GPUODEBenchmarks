# GPUODEBenchmarks

Ensemble ODE solvers on the GPU, timed and checked against one another on
the same problems, grids and stepping: `cubie` (numba-cuda) and `cubie_mlir`
(the MLIR backend of the same install), `jax` (Diffrax), `pytorch`
(torchdiffeq, vmap fork), `myokit_cuda`, `cpp` (MPGOS) and `julia_gpu`
(DiffEqGPU kernels). `julia_cpu` (DifferentialEquations.jl on
`EnsembleThreads`) runs the same trials and supplies the float64 reference
every error is measured against. The figures of *Automated Translation and
Accelerated Solving of Differential Equations on Multiple GPU Platforms* are
under `paper_artifacts/`.

## How a run works

A set under `sets/` expands to run specs. Each spec is one solve: problem,
duration, precision, the swept parameter and its grid, the algorithm,
controller, step or tolerance, and Newton tolerances. `bench.py` writes the
specs as one JSONL trial file per package and hands each file to that
package's runner, which builds once per leg, times every trial after one
untimed warm-up, and records one row per (spec, transfers) in the parquet
store under `data/`. The analyses compute every error offline from the
finals files. Contracts: `store.py` (spec columns, hashes, schema), `trials.py`, `runner.py`, `sets.py`.

## Setup

`SETUP.md` covers the environments: one Python venv per package (the cubie
venv doubles as the suite interpreter), the pinned Julia project, CUDA 13,
and the MPGOS toolchain. `python setup_all_environments.py` builds them all.

## Running

```
python bench.py plan --set perf                 # trials/<key>/<package>.jsonl and counts
python bench.py run  --set perf,golden_grid     # every package the sets name
python bench.py run  --set perf -p cubie,julia_gpu -s lorenz -g tsit5 -n 8,32,128
python bench.py run  --set golden_grid --resume # only the trials with a missing row
python bench.py run  --set perf --floor         # rerun; the lower time per row stays
```

`-p`, `-s`, `-g`, `-n`, `--mode`, `--controller`, `--tol` and `--dt` narrow the
expanded specs; `-n` names counts of the grids' trajectory lists. `--resume`
runs the transfers rows that are missing, `--no-overwrite` those missing
or NaN; a trial keeps asking finals once a row of its carries them. A run
pins the GPU clocks to the row for this card in
`runner_scripts/gpu_clocks.conf` when the shell is elevated (`--no-lock-clocks`
skips it; `runner_scripts/calibrate/calibrate_clocks.py` prints the row for a
new card), samples them at 1 Hz, and writes `logs/<key>_<stamp>/` with one
log per package, `run_manifest.txt`, `summary.tsv` and `clocks.csv`. The
`run_*.sh` and `.bat` wrappers forward to `bench.py`. Julia runs through
`julia +1.13`; set `JULIA` to use another launcher.

| set | packages | what it times |
|---|---|---|
| `perf` | every GPU package | trajectory sweep, 8 to 2^24, one step and one tolerance per algorithm |
| `states` | every GPU package | lorenz96 at 4 to 128 states, cold builds timed, n = 131072 |
| `golden_grid` | every package | finals at every step and tolerance on the 131072-point grid (julia_cpu: the 1024-point prefix) |
| `golden` | `julia_cpu` | the float64 reference at each problem's golden algorithm and tolerance |

Every run is keyed by `<os>_<gpu>` (`runner_scripts/bench_key.py`); a run
refuses to start when `nvidia-smi` cannot name the GPU. A solve past the
trial's watchdog is recorded NaN with a reason and the rest of its leg is
abandoned; a runner that never returns hard-exits and the driver re-invokes
it with the trials still missing.

## Data and analyses

Rows live in `data/key=<key>/package=<pkg>/results/<problem>__<algorithm>.parquet`,
finals in `finals/<trial_id>.parquet` beside them, and the whole tree reads
as one DuckDB table:

```
python runner_scripts/store.py query "SELECT package, n, min_ms FROM results WHERE problem = 'lorenz'"
python analyses/timing.py --x n      --set perf         # min_ms against n
python analyses/timing.py --x states --set states       # min_ms and build_s against states
python analyses/timing.py --x error  --set golden_grid  # min_ms against error
python analyses/agreement.py --set golden_grid          # errors and package-pair differences
```

`data/` is an untracked mirror of the store; the analyses pull it before reading.

Both analyses take `--set` (repeatable) or `--where "<sql>"`, read every key,
and write figures and CSVs under `plots/<key>/<problem>/`.

## Using the store

The store is one `data/` tree on a store box over Tailscale; a machine writes its own key and clocks files. A run pulls the tree before planning and pushes its key after the runners; an analysis pulls before reading; both refuse to run without the store unless `--no-sync`. `sync/README.md` covers setting up the box, connecting a machine and `sync/sync.py`. An error is the RMS
over every state of the difference from the julia_cpu float64 finals, paired
by exact grid value, over the trajectories neither side flags as errored.

## Catalogues

`runner_scripts/problems.csv` is one row per problem: states, duration, the
swept parameter and its range, the golden algorithm and tolerance, and the
packages that implement it. `runner_scripts/algorithms.csv` is one row per
(algorithm, package): whether it runs fixed-step and adaptive, and whether it
takes a Newton tolerance; a package without a row cannot run the algorithm.
`runner_scripts/julia_algorithms.csv` carries the Julia constructors.
`runner_scripts/protocol.toml` holds the repeat schedule and the watchdog.
Adding a problem is one `problems.csv` row plus its right-hand side in each
package's system module and an MPGOS header; adding an algorithm is its
`algorithms.csv` rows plus each package's solver-table entry.

## Tests

```
GPU_ODE_CUBIE/venv/bin/python -m unittest discover -s runner_scripts/tests
julia +1.13 --project=. runner_scripts/tests/test_julia_systems.jl
```

The Python suite runs under the cubie venv (`venv\Scripts\python.exe` on
Windows); the Julia and C++ round trips inside it skip when `julia` or `nvcc`
is absent. The remaining Julia tests (`test_bench_ode_gpu.jl`,
`test_errored.jl`, `test_grid.jl`, `test_results_shim.jl`) run by file the
same way and are also driven from the Python suite.
