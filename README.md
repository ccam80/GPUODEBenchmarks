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
package's runner, which keeps one build while consecutive lines share a
system, algorithm and controller, times every trial after one untimed
warm-up, and records one row per (spec, transfers) in the parquet
store under `data/`. A cubie package's default adaptive controller is
Julia's for the algorithm (`runner_scripts/julia_controllers.csv`, written
by `julia_controllers.jl` from OrdinaryDiffEq's defaults and mapped to
cubie's gains by `cubie_adapter.julia_controller`) and cubie's own where
Julia has none that maps; the spec carries the resolved controller and
gains. The analyses compute every error offline from the
finals files. Contracts: `store.py` (spec columns, hashes, schema), `trials.py`, `runner.py`, `sets.py`, `completeness.py` (what a stored trial must carry to be reused).

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
expanded specs; `-n` names counts of the grids' trajectory lists and exits
for a count no grid of the named sets lists. A trial is one line per point;
a point several set files declare runs under one contract whichever sets
are named: it builds cold, keeps finals, optimizes and times each
transfers mode when any declaration asks; a package every declaration
lists under `[set.untimed]` runs each line once with no warm-up. A cubie
optimize runs once per build and stepping (once per build across dt for
an explicit fixed-step algorithm) for the packages a set's
`[set.optimize]` names, timing the batch and duration cubie's own
`optimize` sizes.
Lines run
easiest first (states, then n, then step and tolerance loose to tight); a
timeout or out-of-memory run abandons every harder run of its family (same
problem, precision, algorithm, controller and gains; larger n or states,
smaller step or tolerance) on the same transfers. Without a flag every
selected trial runs, its rows are overwritten and its cubie kernel is
optimized again (once per run). `--resume` runs the trials the store lacks
rows of, keeping a recorded NaN or error row and a timed-out optimize;
`--no-overwrite` runs every trial without a finite time. Under either, a
trial lacking its cold build time, a readable finals file or its kernel's
optimize record (timed out, under `--no-overwrite`) runs every transfers
again. The
`run_*.sh` and `.bat` wrappers forward to `bench.py`. Julia runs through
`julia +1.13`; set `JULIA` to use another launcher.

Clocks: a run locks the GPU to the card's row in
`runner_scripts/gpu_clocks.conf` or `--lock-clocks SM[,MEM]` from an
elevated shell, and refuses to start when it cannot;
`runner_scripts/calibrate/calibrate_clocks.py` writes the row. A 25 Hz
sample log lands in `data/clocks/<key>_<stamp>.csv`; each row carries
`run`, `driver`, `clock_lock_mhz`, its timing window (`timed_start_utc`,
`timed_end_utc`) and that window's `clock_sm_mhz`, `clock_sm_min_mhz` and
`clock_throttled` (NaN when the log does not cover it). A row throttled or
more than `--clock-tolerance` under the lock fails the run.
`store.py annotate <run> <clocks.csv>` refills the columns from a log. After
the push the box deletes this key's clock logs a day old that no row names
(`sync/box_prune.py`); the mirror and `logs/<run>/` follow.

| set | packages | what it times |
|---|---|---|
| `perf` | every GPU package | trajectory sweep, 8 to 2^24, one step and one tolerance per algorithm |
| `states` | every GPU package | lorenz96 at 4 to 128 states, cold builds timed, n = 131072 |
| `golden_grid` | every package | finals at every step and tolerance on the 131072-point grid, the first 8192 trajectories kept (julia_cpu: the 1024-point prefix) |
| `golden` | `julia_cpu` | the float64 reference at each problem's golden algorithm and tolerance, fabbri_linder excepted |
| `fabbri_linder` | `cubie_mlir` | every adaptive cubie algorithm at 1e-2 to 1e-8 on the 131072-point ACh x Iso grid, traced on the head lattice, single run past 30 s, 200 s watchdog |
| `fabbri_euler` | `cubie_mlir`, `myokit_cuda` | Euler at 100 us to 250 ns on the same grid, traced on the head lattice, single run past 30 s |
| `fabbri_golden` | `julia_cpu` | the fabbri_linder float64 reference: finals and traces of the 1024-point head lattice |
| `fabbri_perf` | `cubie_mlir`, `myokit_cuda` | fabbri_linder trajectory sweep, 8 to 2^20: Kvaerno3, Rosenbrock23 and Tsit5 at 1e-5, Euler at 5 us |

`fabbri_linder` (35 states, `runner_scripts/models/fabbri_linder.cellml`,
cAMP cascade on) sweeps an index: the first 1024 form a 32 x 32
acetylcholine (0..100 nM) by isoprenaline (0..1000 nM) lattice, the rest
fill the plane by bit reversal (`runner_scripts/fabbri.py`, `fabbri.jl`).
cubie loads the CellML, myokit_cuda carries the inputs as zero-derivative
states, julia_cpu integrates `runner_scripts/generated/fabbri_linder_rhs.jl`
(written by `fabbri_export.py` under the cubie venv). A set with
`traces = true` also solves the first 1024 points of every run untimed and
keeps every state at the `[traces]` sample times of `protocol.toml` in
`traces/<trial_id>.parquet`; the error is the RMS over every sample and
state against the golden's traces, over the trajectories that neither side
flags with a failure code or a non-finite sample. A set with `single_run = <seconds>`
takes a first run past that long as the timing, with no warm-up or repeats.

Every run is keyed by `<os>_<gpu>` (`runner_scripts/bench_key.py`); a run
refuses to start when `nvidia-smi` cannot name the GPU. A solve past the
trial's watchdog is recorded NaN with a reason and every harder run of its
family is abandoned; a runner that never returns hard-exits and the driver re-invokes
it with the trials still missing. A cubie package first precompiles the
kernels of its trial file into the package cache, with the optimize
candidates of the kernels whose lines optimize (`bench_cubie.py --trials
<file> --precompile`, four workers of eight kernels each, a worker past 6 GB handing the rest of its chunk to a new one), then runs a fresh runner every 8 kernels at the next family
boundary (`<package>.part<N>.jsonl`). A cold cubie line optimizes on a warm
build first; its timed cold build compiles the optimized kernel once.

A cubie compile the optimize watchdog takes abandons its problem, algorithm
and controller: every line of the group without a row gets a NaN row with
`compile = compile_timeout`, the precompile skips the group's other kernels,
and the trial file is rewritten with the lines marked `"compile": "timeout"`,
which run without an optimize. Every plan, under every flag, reads those rows
and marks the group again; `bench.py plan` counts the timed-out compiles.
To retry, drop the rows (filter as a JSON file or `-` for stdin):
`echo '{"compile": "compile_timeout", "problem": "<p>", "algorithm": "<a>"}' | python runner_scripts/store.py clear -`.
A cubie row's `compile` column is `optimized`, `unoptimized` or
`compile_timeout`; other packages leave it empty.

## Data and analyses

Rows live in `data/key=<key>/package=<pkg>/results/<problem>__<algorithm>.parquet`,
finals in `finals/<trial_id>.parquet` beside them (the first 8192 trajectories of the grid, `store.FINALS_ROWS`; the solve and its timing cover all n), traces in `traces/<trial_id>.parquet` (the first 1024 trajectories at every sample time, one row per trajectory and sample, each carrying its trajectory's failure code), and the whole tree reads
as one DuckDB table:

```
python runner_scripts/store.py query "SELECT package, n, min_ms FROM results WHERE problem = 'lorenz'"
python analyses/plots.py                                        # every kind from every row
python analyses/plots.py --kind error_vs_runtime --where "problem = 'lorenz' AND algorithm = 'tsit5'"
```

`data/` is an untracked mirror of the store; the analysis pulls it before reading.

`analyses/plots.py` reads every row of the store, or the rows
`--where "<sql>"` (a predicate over the columns of the results view, key and
n included) matches, and writes every kind, or the kinds `--kind` names
(repeatable), as the base figures of every (key, problem, algorithm) in
`plots/<key>/<kind>/<problem>_<algorithm>.png`, with the points of a problem
in `plots/<key>/<kind>/<problem>.csv`. What a key's store lacks of a set is
`bench.py plan --set <name> --resume`.

| kind | x | y | rows |
|---|---|---|---|
| `runtime_vs_n` | n | min_ms | the GPU packages |
| `error_vs_runtime` | min_ms | error against the golden | the GPU packages, along the dt and tolerance sweeps |
| `error_vs_dt` | dt | error | every package, the fixed steppings |
| `error_vs_tol` | tolerance | error | every package, the adaptive steppings |
| `states` | states | min_ms and, beside it, the cold build time | the GPU packages of a resized problem |

A package is a colour, a stepping kind a marker (fixed-step; adaptive
steps) and the transfers a line style (solid; dashed and labelled
`+ transfer` when the timing includes the transfers). A series holds the
rows of one key, package, stepping kind and
transfers along the axis and is drawn when it has two or more x values. julia_cpu,
whose timing is not of interest, appears on the error-against-dt and
error-against-tolerance figures only. Rows with `errored_pct` above 10 are
dropped. A figure with one package family (the two cubie backends count as
one) or no series past three points goes under `<kind>/limited_data/`. The
`runtime_vs_n`, `error_vs_runtime` and `states` kinds also get
`<problem>_algorithms.png`, a subplot per algorithm, and
`<algorithm>_problems.png`, a subplot per problem. `plots/all_cards/` holds
the same figures with every key's series together, a marker set per key and
the legend sectioned by key.

A column a file lacks reads as null, and two rows of one run_id are refused. The golden of a system is its julia_cpu float64
row with finals or traces running the problem's golden algorithm; a run with traces is compared over its traces when the golden has them, otherwise over its finals. A CSV leaves out the
columns no row captured.

## Using the store

The store is one `data/` tree on a store box over Tailscale; a machine writes its own key and clocks files. A run pulls the tree before planning and pushes its key after the runners, refusing to start while its key holds files the box lacks or differs from (a pull keeps newer local files); an analysis pulls before reading; both refuse to run without the store unless `--no-sync`. `sync/README.md` covers setting up the box, connecting a machine and `sync/sync.py`. An error is the RMS
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
