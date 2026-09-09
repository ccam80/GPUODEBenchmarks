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

`suite_rev`: `git rev-parse --short HEAD`, `-dirty` appended when tracked files differ.

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
- A failed trial always has a `reason`; NaN `min_ms` with an empty `reason` is a finals-only row.
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

- Ordinal order: `n` ascending, `dt` descending, `tol` descending, `states` ascending.
- A leg is one `warm` line (the cheapest spec), the `optimize` lines of `[set.optimize]`, then the `solve` lines.
- `warm` trials are never recorded; `optimize` trials (cubie) record to `optimize.csv` and apply to the leg's later solves.
- Specs with one `trial_id` across sets merge: first set's leg and axis, `transfers` union, `finals` true over false.

### 1.5 Runner contract

```
<runner argv> --trials <path> [--floor]
```

1. Reads the JSONL, groups `solve` trials by `leg`, builds once per leg, walks ordinals ascending, times each listed transfer leg with one untimed warm-up and the `[repeats]` schedule of `protocol.toml` (`timed_min_ms` semantics).
2. Writes `<trials>.progress` (`{"trial_id": ..., "started_utc": ...}`) before each trial.
3. Records every finished trial through the store before the next starts.
4. Builds the grid by the 1.2 formula; rejects a `controller` name it does not recognise with `reason = "error: unknown controller <name>"`.
5. One abandon rule. Outcomes per (trial, transfers): `ok`, `timeout` (soft cap, run returned), `oom`, `error`. After `timeout` or `oom` at ordinal k, every higher ordinal of the leg with the same transfers is recorded NaN with `reason = "abandoned: <timeout|oom> at ordinal k"` and not run. `error` records `reason = "error: <Type>: <message[:200]>"` and the leg continues. A `none` leg failing after a good `both` leg marks the `none` row only. OOM is classified by exception type or message: CUDA `OUT_OF_MEMORY`, numba `CUDA_ERROR_OUT_OF_MEMORY`, XLA `RESOURCE_EXHAUSTED`, torch `OutOfMemoryError`, Julia `CuError(OUT_OF_MEMORY)`.
6. Exit 0 when the loop completed; 3 on a watchdog hard exit (`wp_common.run_watchdogged`, `watchdog.jl`); other on a crash. On 3 the driver reads the progress file, records `reason = "abandoned: hard-exit at ordinal k"` for the leg's ordinal k and every higher one (each requested transfers row still absent), and re-invokes the runner with the trials that still have no row.
7. `errored_pct`: `store.errored_pct` over the trial's finals, `t_final` and `retcode`.
8. `finals = true`: all n rows through `record_finals` with each trajectory's `t_final` and `retcode`.
9. `warm` trials: cubie `Solver.compile(...)`; jax `jit(f).lower(args).compile()` at the trial's n; MPGOS nvcc into the build cache; Myokit `load_model`; julia_gpu one solve at n = 8 in the leg's process, off the GPU lock; pytorch none. A `cold` warm line builds in a fresh cache directory and its wall time is the leg's `build_s`. `optimize` trials: cubie `Solver.optimize` on the line's n-trajectory batch, the winning launch geometry applied to the leg's later solves.
10. `package_version` and `suite_rev` on every row.
11. Reads `protocol.toml` for `[repeats]` and `[watchdog]`; no environment variables.
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
| golden | julia_cpu | all | default range; n = 131072 | `problems.csv golden_algorithm`, default controller, tol = `golden_tol`, dt0 and Newton package default | none | true; none; precision float64 |

- Every list, grid and pin is spelled out in the set files.
- `protocol.toml` keeps `[repeats]` and `[watchdog]` only.

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

## 2. Packets

- One worktree, one branch off `main`, one PR per packet.
- "Done" is the acceptance line; "Review" is what the PR is read against.
- A packet replaces what it touches: no shim, no dual path; `main` may not run between packets.
- Code documentation describes the code and never names this document, a packet, a section or a phase.

### P1 store and grid
Depends on: nothing.
- `runner_scripts/store.py` and `results.jl`: the 1.2 columns, `run_id`, `trial_id`, `group_id`, `states`, the 1.3 file names, `hash`, `record_batch`, `load_finals`.
- `runner_scripts/grid.py`: the 1.2 formula, `grid_point` (the float64 `v[index]` for set authors); writes `runner_scripts/tests/grids/<problem>_131072.npy`; `grid.jl` and `GPU_ODE_MPGOS/grid.cuh` reproduce the values.
- `test_store.py`, the Julia shim test and a grid test per language.
Done: tests pass; Julia and C++ grids equal the numpy files bit for bit.
Review: the identity is exactly 1.2; hashes match a hand-computed fixture; no error or reference column.

### P2 data conversion
Depends on: P1.
`runner_scripts/convert_legacy.py` maps the CSV trees under `data/` to 1.2 and deletes them; one data commit for both keys.

| legacy | spec |
|---|---|
| `mode = fixed`, `setting` | `controller = fixed`, `dt = setting`; atol, rtol, dt_min, dt_max NaN |
| `mode = adaptive`, `setting` | `controller = default`, `atol = rtol = setting`, `dt = duration * 2^-10`; `dt_min`, `dt_max` NaN |
| implicit fixed rows | `newton_atol = newton_rtol = 1e-6` where the (package, algorithm) row has `newton = true`, else NaN |
| implicit adaptive rows | `newton_atol = newton_rtol = setting` where the (package, algorithm) row has `newton = true`, else NaN |
| explicit rows | newton NaN |
| printed settings | a dt within 1e-8 relative of `duration * 2^-k` becomes that float; a tolerance within 1e-8 of `10^-k` becomes that float |
| `problem`, `states` | `system_params = {"states": states}` for lorenz96, `{}` otherwise; `duration`, `parameter`, `grid_scale`, `grid_min`, `grid_max` from `problems.csv`; `grid_dtype = float32` |
| `error` column | not carried |
| `data/numerical_equivalence/julia/<key>/<problem>/controller_constants.csv` | `controllers/<problem>.csv` under julia_cpu |
| `data/numerical/golden_*` | julia_cpu rows, `precision = float64`, `algorithm = golden_algorithm`, `atol = rtol = golden_tol` from `problems.csv`, dt and newton fields NaN, `n = 131072`, finals kept with `t_final = duration` and empty `retcode`, the `_retcodes.csv` rows carrying their `retcode` and `t_final = NaN`, key `windows_RTX-4070-SUPER`, `min_ms` NaN, `reason` empty |
| overlap `julia_timings.csv` tiers `fixed`, `julia` | `controller = fixed` / `default`; `errored_pct` from the metrics counts; `package_version` from the metadata |
| rows with finals | `errored_pct` by `store.errored_pct` |
| `julia` | `julia_gpu` |
| rows meeting by `run_id` | the row with samples wins, then the later `recorded_utc`; NaN and empty values fill from the loser |

Dropped: every `cpp` row and `mpgos*.csv`; every `nand_gate` row outside the golden; rows with `transfers = d2h`; jax `kvaerno3` rows; every `data/numerical_equivalence` sweep row and finals file; every `data/numerical/<key>` finals file; overlap `performance` phase rows of tier `julia` (tolerance 1e-8) and the overlap `numerical` phase, failures and derived tables.
Done: DuckDB counts per (key, package) equal the script's counts; every converted row whose spec a shipped set produces hashes to that set's `run_id`; the PR body counts the rows no shipped set produces, by (package, controller).
Review: no row invented; the mapping and drop counts reproduced in the PR body.

### P3 sets, trials, entry point
Depends on: P1.
- `runner_scripts/sets.py`: schema, expansion and merge of 1.6; the four set files under `sets/`.
- `runner_scripts/trials.py`: the 1.4 record, JSONL, legs, ordinals.
- `bench.py`: the 1.6 CLI, run loop, exit-3 handling; `launch.py` runner registry.
- `algorithms.csv` and `problems.csv` per 1.7; `julia_algorithms.csv`.
- Delete `resume.py`, `resume.jl`, `wp_common.parse_bench_args`, the `BENCH_*` environment contract, and every `protocol.toml` table but `[repeats]`, `[watchdog]`.
Done: `test_sets.py` covers every shipped set's expansion counts, the julia_cpu 1024 grid equalling the first 1024 values of the 131072 grid, matched and pi resolution, the optimize table, the newton column, the loader's consistency check, merge, and the narrowing flags; `bench.py plan --set perf` prints counts.
Review: no trial field outside 1.4; no runner reads a catalogue; no error or reference anywhere; no package tuple in code that the catalogue should carry.

### P4 runner core and cubie
Depends on: P3.
- `runner_scripts/runner.py`: the 1.5 loop on an adapter interface (`build_leg`, `solve`, `finals`, `compile`, `optimize`, `version`).
- `cubie_bench.py` as that adapter; the controller built from `controller` and `gains`.
- `test_runner.py` with a fake adapter: every outcome, the abandon rule, finals kept and not kept.
Done: tests; `bench.py run --set perf,golden_grid -p cubie -s lorenz -n 32` on the 4070 and the rows read back.
Review: the abandon rule is one function; warm uses `Solver.compile`; `reason` on every NaN row; no golden read by a runner.

### P5 jax, pytorch, myokit_cuda
Depends on: P4.
Adapters on `runner.py`; jax warm keeps `lower().compile()`; pytorch and myokit reject any `controller` but `fixed`.
Done: tests; tiny-n runs where the platform allows.

### P6 julia_gpu
Depends on: P3.
- `julia_driver.py` splits the trial file per leg, spawns `bench_ode_gpu.jl --trials <leg file>` under the GPU lock, applies exit-3 abandon from each progress file.
- `bench_ode_gpu.jl` consumes trials, records through `results.jl`, warms at n = 8, rejects `controller` other than `fixed` or `default`.
Done: `test_julia_driver.py` on a fake julia; a tiny-n run on the 4070.

### P7 julia_cpu and golden
Depends on: P3.
- `GPU_ODE_Julia/bench_ode_cpu.jl` from `ne_diffeq.jl`: trials in, `float32` and `float64` per spec, timings and finals out, `controllers/<problem>.csv` per problem; each finals row carries the final state, `t_final` and the retcode text (empty on `Success`).
- `runner_scripts/julia_systems.jl`, the one system module:

| item | contract |
|---|---|
| builders | `_<problem>_entry(::Type{T})` (`_lorenz96_entry(::Type{T}, n)`), `_build_entry(raw, ::Type{T}; u0map, golden_vars, consistent_u0 = false)`; literal helpers take `T` first |
| element type | literals `T(x)`, `ifelse` thresholds `zero(T)`, `@parameters` defaults `T(x)`; mass matrix `SMatrix{n,n,T}`, u0 `SVector{n,T}`; `_consistent_u0` Newton in Float64; `generate_jacobian` failure is an error; `mtkcompile(raw; split = false)` |
| registry | `_ENTRIES::Dict{Tuple{String, DataType}, Any}`; `julia_system(problem, ::Type{T} = Float32)` |
| `cpu_problem(system, problem, p)` | `T = eltype(system.u0)`; `ODEFunction{true}(system.rhs!; jac = system.jac!)` plus `mass_matrix = Matrix{T}(system.mass_matrix)` when present; `ODEProblem{true}(f, Vector{T}(system.u0_for(T(p))), (zero(T), T(duration)), T[p])` |
| solve kwargs | `abstol`, `reltol`, the trial's `dt`, `dtmin`, `dtmax` when not NaN, `save_everystep = false`, `save_start = false`, `dense = false`, `maxiters = 10^8`, `verbose` off, no tstops |
| finals row | `sol.u[end][system.golden_index]`, `t_final = sol.t[end]`, `retcode = string(sol.retcode)` unless `Success` |
| `GPU_ODE_JuliaKernels` | `_ENTRIES[(name, Float32)] = Base.structdiff(_ENTRY_BUILDERS[name](Float32), NamedTuple{(:sys,)})` |
| `golden/verify_references.jl` | `julia_system(name, Float64)` through `cpu_problem` at the catalogue's golden algorithm and tolerance: pollu k1 = 0.35 vs `POLLU_REF` 1e-13; pleiades m1 = 1.0 vs `PLEI_REF` 1e-10; nand_gate c9 = 5e-5 vs `NAND_REF` 1e-8; ring_modulator Cs = 2e-12 vs Test Set II-3 at t = 1e-3, 1e-8; lorenz96 F = 8 Vern9 vs RadauIIA5 1e-7; Float32 rhs vs Float64 rhs at 20 random states 1e-5 relative; `@allocated rhs!` 0 after one call |
| `tests/test_julia_systems.jl` | per problem and element type: entry builds, `eltype(u0) == T`, `mass_matrix` `nothing` or `SMatrix{n,n,T}`, `rhs!` 0 B, algebraic rows of `rhs!` at `u0_for(sweep_min)` below 1e-12 Float64 and 1e-6 Float32, `golden_index` is `states` distinct indices in `1:n`; runs `verify_references.jl` |
| delete | `reference_systems.jl`, `golden/generate_golden.jl`, `golden/retcode_sidecar.jl`, `data/numerical/golden_*_retcodes.csv`, every `include` of `reference_systems.jl` |

Done: `bench.py run --set golden -s lorenz` reproduces the converted golden finals to float64 roundoff; `golden_grid -p julia_cpu -s lorenz` lands 1024-row finals; both test files pass under `julia +1.13`.

### P8 cpp
Depends on: P3.
`run_ode_cpp.ps1` and `.sh` take `--trials`; `mpgos_trials.py` lists builds and points; `Bench.cu` records through the store CLI and builds its grid with `grid.cuh`.
Done: a tiny-n run on Windows lands rows; the Linux script mirrors it.

### P9 analyses
Depends on: P2, P3.
`analyses/errors.py`, `analyses/timing.py`, `analyses/agreement.py` per 1.8. Delete `runner_scripts/plot/*.jl`, `compare_numerical_equivalence.py`, `compare_numerical_results.py`, `run_cubie_julia_overlap.py`, `runner_scripts/cubie_julia_overlap/`, `run_numerical_equivalence.*`, `ne_common.py`.
Done: `test_errors.py` pairs a 1024-row finals with a 131072-row golden by value; every script runs on the converted data and writes figures.

### P10 docs and tests
Depends on: P4 to P9.
README to about 100 wrapped lines; `SETUP.md`; this document reduced to section 1; dead tests, `results.py` and `wp_common` leftovers removed.

### P11 smoke and reruns
Depends on: P10.
`bench.py run --set perf,states,golden_grid -n 128` on the 4070 for every package; then per key: `cpp` in full; `golden_grid` for every package including julia_cpu on every problem; jax `kvaerno3` (WSL); every set on `nand_gate`; cubie and cubie_mlir in full.

## 3. Schedule

```
wave 1:  P1
wave 2:  P2 | P3
wave 3:  P4 | P6 | P7 | P8 | P9       (P4, P6, P7, P8 need P3; P9 needs P2 and P3)
wave 4:  P5                           (needs P4)
wave 5:  P10, then P11
```

Conflict hotspots: `launch.py` (P3 owns the registry; runner packets add one line each), `store.py` (P1 only), the set files (P3 only).

## 4. Fixed decisions

- Store: parquet leg files read with DuckDB; no database service; cross-machine sync deferred.
- Analyses: Python and matplotlib only; every error is computed there.
- Suite interpreter: `GPU_ODE_CUBIE/venv`.
- The golden is a julia_cpu float64 row, not a separate artefact.
- One timing tolerance, 1e-5, for every package.
- `dt_min` and `dt_max` are never pinned by a shipped set; every package keeps its own floor and cap.
- Package capability lives in `algorithms.csv`, never in a tuple in code.
- No reason category or column exists for legacy data alone.
