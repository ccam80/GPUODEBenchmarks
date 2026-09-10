# Unification plan

Sets expand to run specs; runners execute trial files; the store holds one row per spec; analyses compute every error offline.

## 1. Contracts

### 1.1 Packages

`cubie`, `cubie_mlir`, `jax`, `pytorch`, `myokit_cuda`, `cpp`, `julia_gpu`
(DiffEqGPU kernels), `julia_cpu` (DifferentialEquations.jl on
`EnsembleThreads`). Package names are the store partition names.

| package | package_version |
|---|---|
| cubie, cubie_mlir | `importlib.metadata.version("cubie")` + `+` + backend name |
| jax | `jax.__version__` |
| pytorch | `torch.__version__` + `+` + torchdiffeq fork commit |
| myokit_cuda | `myokit.__version__` |
| cpp | 12-character MPGOS source hash + `+nvcc` + release |
| julia_gpu | DiffEqGPU version from `Manifest.toml` |
| julia_cpu | OrdinaryDiffEq version from `Manifest.toml` |

`suite_rev`: `git rev-parse --short HEAD`.

### 1.2 Run spec

One solve, fully described; the store identity and the trial body.

| group | column | type | values |
|---|---|---|---|
| system | problem | string | a name in `problems.csv` |
| system | system_params | string | canonical JSON object (sorted keys, no whitespace); `{"states":<int>}` for lorenz96, `{}` otherwise |
| system | duration | float64 | integration end; t0 = 0 |
| system | precision | string | `float32` or `float64` |
| ensemble | parameter | string | the swept parameter name |
| ensemble | grid_scale | string | `linear` or `log` |
| ensemble | grid_min, grid_max | float64 | first and last value |
| ensemble | n | int64 | point count |
| ensemble | grid_dtype | string | `float32`; the values are rounded to it |
| stepping | algorithm | string | an algorithm in `algorithms.csv` with a row for the package |
| stepping | controller | string | `fixed`; `default` (the package's shipped controller for the algorithm); or a name the package recognises (cubie: `i`, `pi`, `pid`, `gustafsson`) |
| stepping | dt | float64 | the step when `fixed`; dt0 otherwise; NaN = package default |
| stepping | dt_min, dt_max | float64 | NaN = package default |
| stepping | atol, rtol | float64 | NaN when `fixed` |
| stepping | gains | string | canonical JSON of the controller parameters set explicitly; `{}` when none |
| stepping | newton_atol, newton_rtol | float64 | NaN unless the (package, algorithm) row of `algorithms.csv` has `newton = true` |
| conditions | transfers | string | `both` or `none` |
| conditions | package, key | string | the executing package and machine |

Grid values (every language; tested against a numpy reference at n = 131072):

```
linear: v[i] = grid_min + i * ((grid_max - grid_min) / (n - 1)) in float64, v[n-1] = grid_max
log:    v[i] = 10 ** (log10(grid_min) + i * ((log10(grid_max) - log10(grid_min)) / (n - 1))), v[n-1] = grid_max
then cast to grid_dtype; float64 runs use the cast values widened back
```

Hashes: `sha1` of the canonical JSON of the fields in table order, floats `%.17g`, NaN `"nan"`, first 16 hex.
- `trial_id`: every spec field except `transfers` and `key`.
- `run_id`: every spec field; the store's replace key.
- `group_id`: every spec field except the ensemble fields, `transfers`, `package` and `key`; the analyses' comparison key.
- Equal hashes mean the same spec; floats compare exactly.

### 1.3 Store

```
data/key=<os>_<gpu>/package=<pkg>/results/<problem>__<algorithm>.parquet
data/key=<os>_<gpu>/package=<pkg>/finals/<trial_id>.parquet
data/key=<os>_<gpu>/package=julia_cpu/controllers/<problem>.csv
data/clocks/
```

Row schema: every spec column of 1.2 plus:

| column | type | notes |
|---|---|---|
| run_id, trial_id, group_id | string | 1.2 |
| states | int32 | resulting state count |
| min_ms | float64 | NaN when not timed |
| samples_ms | list<float64> | every attempt in ms, warm-up first |
| errored_pct | float64 | trajectories flagged by `store.errored_mask`, percent; NaN when unknown |
| build_s | float64 | cold build seconds; NaN unless measured |
| reason | string | empty unless the trial failed |
| finals | string | `finals/<trial_id>.parquet` when kept; empty otherwise |
| package_version, suite_rev | string | 1.1 |
| recorded_utc | timestamp[us, UTC] | |

Rules:
- A record with the same `run_id` replaces the row; under `--floor` the lower finite `min_ms` stays and NaN never wins.
- Absent rows are gaps; no placeholder rows.
- A runner records a `reason` on every failed trial; analyses treat NaN `min_ms` as untimed whatever the `reason`.
- Finals file: `traj int32, s1..sk float32 (float64 for float64 runs), t_final float64, retcode string` (the package's failure code, empty on success), one row per grid value in grid order, all n rows.
- Errored trajectory (`store.errored_mask`): a non-finite state, `t_final` off `duration` by over 1e-4 relative, or a non-empty `retcode`.
- One finals file per `trial_id`, shared by the `both` and `none` rows, replaced on rerun.
- No error column: errors are computed by the analyses from finals.

`runner_scripts/store.py` is the only writer and the Python reader:

```
Store(root="data").record(row, floor=False)              # atomic leg-file swap under a mkdir lock
Store.record_batch(rows, floor=False)
Store.record_finals(spec, finals, t_final, retcode) -> relative path
Store.load_finals(package, key, relative) -> (traj, states, t_final, retcode)
store.errored_mask(states, t_final, retcode, duration) -> bool[n]
store.errored_pct(states, t_final, retcode, duration) -> float
Store.status(run_id) -> "absent" | "nan" | "finite"
Store.rows(sql_where="", **eq_filters) -> list[dict]       # DuckDB over the tree
python store.py record  <rows.json | -> [--floor]
python store.py finals  <spec.json> <finals.csv>
python store.py status  <run_id>
python store.py query   "<sql over results>"
python store.py clear   <filter.json>
python store.py hash    <spec.json>                        # prints trial_id, run_id, group_id
```

- `results.jl` serialises rows with JSON.jl and calls the CLI; `Bench.cu` calls the CLI.
- Suite interpreter: `GPU_ODE_CUBIE/venv` with `pyarrow` and `duckdb`, `launch.suite_python()`.

### 1.4 Trials

One JSONL file per package, the runner's only input; one line per trial: the 1.2 fields except `transfers` and `key`, plus:

| field | values |
|---|---|
| trial_id | 1.2 |
| kind | `solve`, `warm`, `optimize` |
| finals | `true` (finals kept) or `false` |
| transfers | list, subset of `both`, `none`, in timing order; empty on `warm` and `optimize` |
| leg | `<problem>/<system_params>/<algorithm>/<controller>/<precision>/<axis>` |
| axis | `n`, `dt`, `tol`, `states` |
| ordinal | cost order within the leg |
| cold | `true` on the `warm` line of a `build = "cold"` set: fresh cache directory, `build_s` recorded |
| watchdog_s | the soft cap in seconds: the set's `watchdog`, else `[watchdog] seconds` |

- Ordinal order: `n` ascending, `dt` descending, `tol` descending, `states` ascending.
- A leg is one `warm` line (the cheapest spec), the `optimize` lines of `[set.optimize]`, then the `solve` lines.
- `warm` trials are never recorded; `optimize` trials (cubie) apply the point's `optimize.csv` row when one was recorded from the same source (system hash, cubie source and version) and compile, else run `Solver.optimize` and replace the row.
- Specs with one `trial_id` across sets merge: first set's leg and axis, `transfers` union, `finals` true over false, the larger `watchdog_s`.

### 1.5 Runner contract

```
<runner argv> --trials <path> [--floor]
```

1. Reads the JSONL, groups `solve` trials by `leg`, builds once per leg, walks ordinals ascending, times each listed transfer leg with one untimed warm-up and the `[repeats]` schedule of `protocol.toml` (`timed_min_ms` semantics); the soft cap is the trial's `watchdog_s` and the hard exit fires 30 s after it; an adapter's `reset` runs untimed before every attempt after the first.
2. Writes `<trials>.progress` (`{"trial_id": ..., "kind": ..., "started_utc": ...}`) before each line, warm and optimize lines included.
3. Records every finished trial through the store before the next starts.
4. Builds the grid by the 1.2 formula; rejects a `controller` name it does not recognise with `reason = "error: unknown controller <name>"`.
5. One abandon rule. Outcomes per (trial, transfers): `ok`, `timeout` (the trial's `watchdog_s` passed, run returned), `oom`, `error`. After `timeout` or `oom` at ordinal k, every higher ordinal of the leg with the same transfers is recorded NaN with `reason = "abandoned: <timeout|oom> at ordinal k"` and not run. `error` records `reason = "error: <Type>: <message[:200]>"` and the leg continues. A `none` leg failing after a good `both` leg marks the `none` row only. OOM is classified by exception type or message: CUDA `OUT_OF_MEMORY`, numba `CUDA_ERROR_OUT_OF_MEMORY`, XLA `RESOURCE_EXHAUSTED`, torch `OutOfMemoryError`, Julia `CuError(OUT_OF_MEMORY)`.
6. Exit 0 when the loop completed; 3 on a watchdog hard exit (`wp_common.run_watchdogged`, `watchdog.jl`); other on a crash. On 3 the driver reads the progress file, records `reason = "abandoned: hard-exit at ordinal k"` for the leg's ordinal k and every higher one (each requested transfers row still absent), and re-invokes the runner with the trials that still have no row. When the progress file names an `optimize` line, the driver records an `optimize.csv` row labelled `timeout`, drops that line and re-invokes the runner, so the leg's solves run at the solver's own geometry.
7. `errored_pct`: `store.errored_pct` over the trial's finals, `t_final` and `retcode`.
8. `finals = true`: all n rows through `record_finals` with each trajectory's `t_final` and `retcode`.
9. `warm` trials: cubie `Solver.compile(...)`; jax `jit(f).lower(args).compile()` at the trial's n; MPGOS nvcc into the build cache; Myokit `load_model`; julia_gpu one solve at n = 8 in the leg's process, off the GPU lock; pytorch none. A warm line followed by an `optimize` line builds without compiling. A `cold` warm line builds in a fresh cache directory and its wall time is the leg's `build_s`. `optimize` trials: cubie `Solver.optimize` on the line's n-trajectory batch, the winning launch geometry applied to the leg's later solves. Build and compile-only warm lines run without a cap; a warm line that solves hard-exits 30 s after the trial's cap; an optimize line hard-exits after `[watchdog] optimize_seconds`.
10. `package_version` and `suite_rev` on every row.
11. Reads `protocol.toml` for `[repeats]` and `[watchdog]` (`seconds`, `exit_code`, `optimize_seconds`); no environment variables.
12. Applies `dt_min` and `dt_max` only when they are not NaN; a NaN leaves the package's own floor and cap in place.

### 1.6 Sets and the entry point

A set is a TOML file under `sets/` that expands to run specs:

```toml
[set]
packages   = ["cubie", "cubie_mlir", "jax", "pytorch", "myokit_cuda", "cpp", "julia_gpu", "julia_cpu"]
problems   = "all"                 # or a list of problems.csv names
algorithms = "all"                 # or a list; always narrowed by algorithms.csv capability
precision  = "float32"
finals     = false                 # true: every solve keeps its finals
transfers  = ["both", "none"]
build      = "warm"                # or "cold": the warm line carries cold = true
watchdog   = 120                   # seconds one run may take; default [watchdog] seconds

[set.optimize]                     # optional; without it no optimize lines
packages = ["cubie", "cubie_mlir"]
n = 262144                         # or "solve": each solve's own n
per = "leg"                        # one line after the warm line; or "solve": one before every solve

[[grid]]
packages = "all"                   # optional narrowing
parameter = "default"              # problems.csv sweep_parameter, or a name
scale = "default"                  # problems.csv sweep_scale, linear, log
min = "default"                    # problems.csv sweep_min, or a float
max = "default"                    # problems.csv sweep_max, or a float
problems = {lorenz = {max = 0.16389...}}   # per-problem overrides of any grid field
n = [8, 32, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608, 16777216]
system_params = {}                 # or {states = [4, 8, 16, 32, 64, 128]}: one grid per value

[[stepping]]
packages = "all"                   # optional narrowing
algorithms = "all"                 # optional narrowing
controller = "fixed"
dt = {duration_times_2_pow = [-10]}       # one spec per exponent
newton = {atol = 1.0e-6, rtol = 1.0e-6}   # rows with newton = true; others get NaN

[[stepping]]
controller = "default"             # or a package name, or "matched", or "pi"
tol = [1.0e-5]                     # atol = rtol, one spec per value
dt0 = {duration_times_2_pow = -10}
dt_min = "none"                    # package default; a float pins it
dt_max = "none"
newton = "tol"                     # newton_atol = newton_rtol = tol
gains = {}                         # explicit controller parameters, or "dirk_defaults"
```

Expansion, in `runner_scripts/sets.py`:

1. Cartesian product of packages × problems the package implements (`problems.csv frameworks`) × algorithms whose `algorithms.csv` row for the package has the stepping's kind (`fixed` / `adaptive`) true × grids × steppings, each narrowed by its own `packages` and `algorithms`.
2. `default` grid fields resolve from `problems.csv`; `problems.<name>` overrides fields with literal floats.
3. `duration_times_2_pow = k` resolves to `duration * 2**k`; `duration_times = f` to `duration * f`.
4. `controller = "matched"` (cubie packages only) reads `controllers/<problem>.csv` of julia_cpu under the run key and resolves the algorithm's row through `cubie_adapter.matched_controller` into `controller = "pi"` with explicit `gains`; skipped when the row is absent or the result equals cubie's shipped controller. `gains = "dirk_defaults"` resolves through `cubie_adapter.pi_tier_controller(order)`; skipped when equal to shipped.
5. `system_params` with a list value yields one grid per value; the problem's default construction parameters otherwise (`{"states":32}` for lorenz96).
6. Trials that share a `trial_id` merge: `transfers` union, `finals` true over false.
7. The axis of a grid and stepping: `states` when the grid lists `system_params`, else `dt` or `tol` when the stepping lists more than one value, else `n`.
8. `newton_atol`, `newton_rtol` resolve only for a (package, algorithm) row with `newton = true`; every other spec carries NaN.

```
bench.py plan|run --set <name>[,<name>] [-p pkgs] [-s problems] [-g algorithms]
                  [--mode fixed|adaptive] [--controller names] [-n list] [--tol list] [--dt list]
                  [--resume | --no-overwrite] [--floor] [--cooldown S] [clock flags]
```

- `-p -s -g --mode --controller -n --tol --dt` narrow the expanded specs; `-n` replaces the perf `n` list.
- `--resume` drops trials whose every requested transfers row exists; `--no-overwrite` those whose rows are all finite.
- `plan` writes `trials/<key>/<package>.jsonl` and prints counts per package and leg.
- `run` writes the same under `logs/<key>_<stamp>/`, drives runners per 1.5 (6), keeps the clock guard, manifest and summary; no analysis.

Shipped sets (`precision = "float32"`, `build = "warm"`, Newton `1e-6` fixed and `tol` adaptive, `dt0 = duration * 2^-10`, `dt_min` and `dt_max` package default, unless stated). Timed algorithms are those two package families run, cubie and cubie_mlir counting as one and julia_cpu excluded: fixed euler, classical-rk4, tsit5, rosenbrock23_sciml, kvaerno3, vern7, kvaerno5; adaptive tsit5, cash-karp-54, rosenbrock23_sciml, kvaerno3, vern7, kvaerno5.

| set | packages | problems | grid | stepping | optimize | finals, transfers |
|---|---|---|---|---|---|---|
| perf | all but julia_cpu | all | default range; n = perf list | the timed algorithms: fixed dt 2^-10; default controller tol 1e-5; cubie packages add `pi` with `dirk_defaults` at tol 1e-5 | cubie packages, 262144, per leg | false; both, none |
| states | all but julia_cpu | lorenz96 | default range; n = 131072; states 4, 8, 16, 32, 64, 128 | as perf | as perf | false; both, none; build cold |
| golden_grid | all | all | default range; n = 131072; julia_cpu n = 1024 with per-problem `max` = the float64 `v[1023]` of the 131072-point grid, written out with 17 digits | every algorithm the package runs: fixed dt 2^-k, k 1..13 (euler 8..17); default controller tol 1e-2..1e-8; cubie packages add `matched` and `pi` with `dirk_defaults` over the same tolerances | cubie packages, 262144, per solve | true; none |
| golden | julia_cpu | all | default range; n = 131072 | `problems.csv golden_algorithm`, default controller, tol = `golden_tol`, dt0 and Newton package default | none | true; none; precision float64; watchdog 86400 |

- Every list, grid and pin is spelled out in the set files.
- `protocol.toml` keeps `[repeats]` and `[watchdog]` only; a set's `watchdog` overrides `[watchdog] seconds` for its trials.

### 1.7 Catalogues

- `problems.csv`: `problem, display, states, duration, sweep_parameter, sweep_min, sweep_max, sweep_scale, golden_algorithm, golden_tol, frameworks`; read by `sets.py` only.
- `algorithms.csv`: one row per (package, algorithm): `algorithm, package, display, family, order, fixed, adaptive, newton`; `fixed` and `adaptive` say whether the package's solver table runs the algorithm at a fixed step and under an adaptive controller, `newton` whether it takes a Newton tolerance; a package without a row cannot run the algorithm. The loader rejects a table where rows of one algorithm disagree on `display`, `family` or `order`. Read by `sets.py` only.
- `runner_scripts/julia_algorithms.csv`: the Julia constructor columns; read by the Julia adapters.
- Runners map `problem` to their system modules and `algorithm` to their solver tables; everything else comes from the trial.

### 1.8 Analyses

`analyses/errors.py`, the one comparison:
- `compare(a, b)`: two rows with finals; both grids rebuilt by the 1.2 formula; rows paired by exact float32 parameter value; the RMS over every state, in float64, of the difference over the paired rows neither side flags by `store.errored_mask`; NaN when no row pairs.
- `golden_of(row)`: the julia_cpu `float64` finals row with the same `problem`, `system_params` and `duration` under any key; None when absent; raises naming the rows when more than one exists.
- `error(row)` = `compare(row, golden_of(row))`.

Two scripts, each taking `--set <name>` (repeatable) or `--where "<sql over the spec columns>"`, resolved to `trial_id`s with the grid fields ignored, under every key:
- `timing.py --x n|error|states`: `min_ms` against the axis per (key, problem, `group_id`, transfers), one series per package; `--x error` computes `error(row)`; `--x states` adds `build_s`.
- `agreement.py`: per `group_id`, `error(row)` for every package that ran it and `compare` between every pair of packages; CSV and figures per (key, problem).
- Rows with `errored_pct > 10` are dropped where the column is a number; rows with NaN `min_ms` are dropped from timing axes.
- Absent `errored_pct`, `reason`, `samples_ms`, `finals`, `package_version` never raise.
- Output under `plots/<key>/<problem>/`.
