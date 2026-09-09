# Unification plan

One entry point expands named sets into fully explicit run specs, one runner
per package executes them, one store holds one row per executed spec, and
analyses select rows by expanding the same sets. Every packet is built and
reviewed against section 1.

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

`suite_rev` is `git rev-parse --short HEAD`, suffixed `-dirty` when tracked
files differ.

### 1.2 Run spec

A run spec is the complete description of one solve. Its fields are the store
identity and the trial body; nothing in it names a benchmark type.

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
| stepping | algorithm | string | a name in `algorithms.csv` |
| stepping | controller | string | `fixed`; `default` (the package's shipped controller for the algorithm); or a name the package recognises (cubie: `i`, `pi`, `pid`, `gustafsson`) |
| stepping | dt | float64 | the step when `fixed`; dt0 otherwise; NaN = package default |
| stepping | dt_min, dt_max | float64 | NaN when `fixed` or package default |
| stepping | atol, rtol | float64 | NaN when `fixed` |
| stepping | gains | string | canonical JSON of the controller parameters set explicitly; `{}` when none |
| stepping | newton_atol, newton_rtol | float64 | NaN for explicit algorithms and packages without the setting |
| conditions | transfers | string | `both` or `none` |
| conditions | package, key | string | the executing package and machine |

Grid values, identical in every language, tested against a numpy reference
file at n = 131072 per problem:

```
linear: v[i] = grid_min + i * ((grid_max - grid_min) / (n - 1)) in float64, v[n-1] = grid_max
log:    v[i] = 10 ** (log10(grid_min) + i * ((log10(grid_max) - log10(grid_min)) / (n - 1))), v[n-1] = grid_max
then cast to grid_dtype; float64 runs use the cast values widened back
```

Identity hashes, both `sha1` over the canonical JSON of the named fields in
the order listed above, floats as `%.17g`, NaN as `"nan"`, first 16 hex:

- `trial_id`: every spec field except `transfers` and `key`.
- `run_id`: every spec field. The replace key of the store.

Two specs are the same when their hashes are equal; floats are compared
exactly, so set expansion must produce values by one formula.

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
| run_id, trial_id | string | 1.2 |
| states | int32 | resulting state count |
| min_ms | float64 | NaN when not timed |
| samples_ms | list<float64> | every attempt in ms, warm-up first |
| errored_pct | float64 | trajectories with a non-finite final, percent; NaN when unknown |
| error | float64 | 1.5 (7); NaN when no reference |
| reference | string | `run_id` of the reference row scored against; empty when none |
| build_s | float64 | cold build seconds; NaN unless measured |
| reason | string | empty on success; 1.5 (5) |
| finals | string | `finals/<trial_id>.parquet` when kept; empty otherwise |
| package_version, suite_rev | string | 1.1 |
| recorded_utc | timestamp[us, UTC] | |

Rules:
- A record with the same `run_id` replaces the row; under `--floor` the lower
  finite `min_ms` stays and NaN never wins.
- Absent rows are gaps; no placeholder rows.
- Finals file: `traj int32, s1..sk float32 (float64 for float64 runs),
  converged bool`, one row per grid value in grid order, all n rows.
- One finals file per `trial_id`, shared by the `both` and `none` rows, replaced
  on rerun.

`runner_scripts/store.py` is the only writer and the Python reader:

```
Store(root="data").record(row, floor=False)              # atomic leg-file swap under a mkdir lock
Store.record_batch(rows, floor=False)
Store.record_finals(spec, finals, converged) -> relative path
Store.status(run_id) -> "absent" | "nan" | "finite"
Store.rows(sql_where="", **eq_filters) -> list[dict]       # DuckDB over the tree
Store.covering(spec) -> row | None                         # 1.6 reuse rule
python store.py record  <rows.json | -> [--floor]
python store.py finals  <spec.json> <finals.csv>
python store.py status  <run_id>
python store.py query   "<sql over results>"
python store.py clear   <filter.json>
python store.py hash    <spec.json>                        # prints trial_id and run_id
```

`results.jl` serialises rows with JSON.jl and calls the CLI; `Bench.cu` calls
the CLI. The suite interpreter is `GPU_ODE_CUBIE/venv` with `pyarrow` and
`duckdb`, exposed as `launch.suite_python()`.

### 1.4 Trials

One JSONL file per package; a runner takes that path and nothing else. One
line per trial: every spec field of 1.2 except `transfers` and `key`, plus:

| field | values |
|---|---|
| trial_id | 1.2 |
| kind | `solve`, `warm`, `optimize` |
| role | `timed` (no finals), `numerical` (finals kept) |
| transfers | list, subset of `both`, `none`, in timing order |
| reference | store-relative path `key=<k>/package=julia_cpu/finals/<id>.parquet` or `""` |
| leg | `<problem>/<system_params>/<algorithm>/<controller>/<precision>/<axis>` |
| axis | `n`, `dt`, `tol`, `states` |
| ordinal | cost order within the leg |

Ordinal order: `n` ascending, `dt` descending, `tol` descending, `states`
ascending. `warm` trials are never recorded. `optimize` trials exist for cubie
packages and record to `optimize.csv`.

### 1.5 Runner contract

```
<runner argv> --trials <path> [--floor]
```

1. Reads the JSONL, groups `solve` trials by `leg`, builds once per leg, walks
   ordinals ascending, times each listed transfer leg with one untimed warm-up
   and the `[repeats]` schedule of `protocol.toml` (`timed_min_ms` semantics).
2. Writes `<trials>.progress` (`{"trial_id": ..., "started_utc": ...}`) before
   each trial.
3. Records every finished trial through the store before the next starts.
4. Builds the grid by the 1.2 formula; rejects a `controller` name it does not
   recognise with `reason = "error: unknown controller <name>"`.
5. One abandon rule. Outcomes per (trial, transfers): `ok`, `timeout` (soft
   cap, run returned), `oom`, `error`. After `timeout` or `oom` at ordinal k,
   every higher ordinal of the leg with the same transfers is recorded NaN with
   `reason = "abandoned: <timeout|oom> at ordinal k"` and not run. `error`
   records `reason = "error: <Type>: <message[:200]>"` and the leg continues.
   A `none` leg failing after a good `both` leg marks the `none` row only. OOM
   is classified by exception type or message: CUDA `OUT_OF_MEMORY`, numba
   `CUDA_ERROR_OUT_OF_MEMORY`, XLA `RESOURCE_EXHAUSTED`, torch
   `OutOfMemoryError`, Julia `CuError(OUT_OF_MEMORY)`.
6. Exit 0 when the loop completed; 3 on a watchdog hard exit
   (`wp_common.run_watchdogged`, `watchdog.jl`); other on a crash. On 3 the
   driver reads the progress file, records
   `reason = "abandoned: hard-exit at ordinal k"` for the leg's higher ordinals,
   and re-invokes the runner with the trials that still have no row.
7. `error` when `reference` is set: the reference finals are loaded, both grids
   are built by the formula and must be float32-equal over all n (else
   `reason = "error: reference grid mismatch"`); `error` is the root mean square
   over the rows where both are `converged` and finite, of the difference over
   every state, in float64; `errored_pct` counts the trial's own non-finite rows.
8. `role = numerical`: the finals of all n rows are written through
   `record_finals`; `converged` is the package's success flag when it has one,
   else finiteness.
9. `warm` trials: cubie `Solver.compile(...)`; jax
   `jit(f).lower(args).compile()` at the trial's n; MPGOS nvcc into the build
   cache; Myokit `load_model`; julia_gpu one solve at n = 8 in the leg's
   process, off the GPU lock; pytorch none. Legs with `system_params` varying
   (`axis = states`) are never warmed; their cold `build_s` is the measurement.
10. `package_version` and `suite_rev` on every row.

Runners read `protocol.toml` for `[repeats]` and `[watchdog]` only, and no
environment variable for resume or floor.

### 1.6 Sets and the entry point

A set is a TOML file under `sets/` that expands to run specs. Schema:

```toml
[set]
packages   = ["cubie", "cubie_mlir", "jax", "pytorch", "myokit_cuda", "cpp", "julia_gpu", "julia_cpu"]
problems   = "all"                 # or a list of problems.csv names
algorithms = "all"                 # or a list; always narrowed by algorithms.csv capability
precision  = "float32"
role       = "timed"               # or "numerical"
transfers  = ["both", "none"]
build      = "warm"                # or "cold": no warm trials, build_s recorded

[[grid]]
parameter = "default"              # problems.csv sweep_parameter, or a name
scale = "default"                  # problems.csv sweep_scale, linear, log
min = "default"                    # problems.csv sweep_min, or a float
max = "default"                    # problems.csv sweep_max, a float, or {grid_n = 131072, index = 1023}
n = [8, 32, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608, 16777216]
system_params = {}                 # or {states = [4, 8, 16, 32, 64, 128]}: one grid per value

[[stepping]]
algorithms = "all"                 # optional narrowing
controller = "fixed"
dt = {duration_times_2_pow = [-10]}       # one spec per exponent
newton = {atol = 1.0e-6, rtol = 1.0e-6}   # implicit algorithms; explicit get NaN

[[stepping]]
controller = "default"             # or a package name, or "matched", or "pi"
tol = [1.0e-5]                     # atol = rtol, one spec per value
dt0 = {duration_times_2_pow = -10}
dt_min = {duration_times = 1.0e-6}
dt_max = "none"
newton = "tol"                     # newton_atol = newton_rtol = tol
gains = {}                         # explicit controller parameters
```

Expansion, in `runner_scripts/sets.py`:

1. Cartesian product of packages × problems the package implements
   (`problems.csv frameworks`) × algorithms the package can run with the
   stepping's controller kind (`algorithms.csv fixed` / `adaptive`) × grids
   × steppings.
2. `default` grid fields resolve from `problems.csv`; `{grid_n, index}` resolves
   `max` to `v[index]` of the `grid_n`-point default grid by the 1.2 formula.
3. `duration_times_2_pow = k` resolves to `duration * 2**k`;
   `duration_times = f` to `duration * f`.
4. `controller = "matched"` (cubie packages only) reads
   `controllers/<problem>.csv` of julia_cpu under the run key and resolves the
   algorithm's row through `cubie_adapter.matched_controller` into
   `controller = "pi"` with explicit `gains`; the entry is skipped when the row
   is absent or the result equals cubie's shipped controller.
   `controller = "pi"` with `gains = "dirk_defaults"` resolves through
   `cubie_adapter.pi_tier_controller(order)`; skipped when equal to shipped.
5. `system_params` with a list value yields one grid per value with the
   product's `system_params` set; the problem's default construction
   parameters otherwise (`{"states":32}` for lorenz96).
6. Trials that share a `trial_id` merge: `transfers` union, `numerical` over
   `timed`, `reference` kept when set.
7. `reference` is set on every `numerical` trial and every trial whose grid is
   covered by a julia_cpu `float64` finals row of the same problem,
   `system_params` and duration (same key first, else any key): the reference
   row's grid must equal the trial's grid, or contain it (same scale and min,
   larger n, trial `grid_max` float32-equal to the reference `v[n-1]`).

Reuse rule (`Store.covering`): a `numerical` trial is covered, and not
emitted, when a row with `finals` exists whose spec equals the trial's except
for the grid, and whose grid contains the trial's grid as in (7). A `timed`
trial is covered by any row of the same `run_id` per requested transfers.

```
bench.py plan|run --set <name>[,<name>] [-p pkgs] [-s problems] [-g algorithms]
                  [--mode fixed|adaptive] [--controller names] [-n list] [--tol list] [--dt list]
                  [--resume | --no-overwrite] [--floor] [--cooldown S] [clock flags]
```

`-p -s -g --mode --controller -n --tol --dt` narrow the expanded specs; `-n`
also replaces the `n` list of every grid whose file value is the perf list.
`--resume` drops trials whose every requested transfers row exists;
`--no-overwrite` drops those whose rows are all finite. `plan` writes
`trials/<key>/<package>.jsonl` and prints counts per package and leg; `run`
writes the same under `logs/<key>_<stamp>/`, drives the runners per 1.5 (6),
keeps the clock guard, manifest and summary, and never invokes an analysis.

The shipped sets, all with `precision = "float32"`, `build = "warm"` and the
1.5 Newton and pin values unless stated:

| set | packages | problems | algorithms | grid | stepping | role, transfers |
|---|---|---|---|---|---|---|
| perf | all | all | all | default range; n = perf list | fixed dt 2^-10; default controller tol 1e-5 | timed; both, none |
| pairwise | all | all | all | default range; n = 32768 | as perf | numerical; both, none |
| wp | all | all | all | default range; n = 131072 | fixed dt 2^-k, k 4..13 (euler 8..17); default controller tol 1e-2..1e-8 | timed; none |
| ne | cubie, cubie_mlir, julia_cpu | all | `backwards_euler, crank_nicolson, trapezoidal_dirk, implicit_midpoint, sdirk_2_2, l_stable_sdirk_4, kvaerno3, kvaerno5, radau_iia_3, radau_iia_5, radau_iia_9, ros3p, rodas3p, rosenbrock23_sciml` fixed; those plus `tsit5, cash-karp-54, bogacki-shampine-32, dormand-prince-54, fehlberg-45, dormand-prince-853, vern7` adaptive | n = 1024, max = {grid_n = 131072, index = 1023} | fixed dt 2^-k, k 1..13; default controller and matched, tol 1e-2..1e-8 | numerical; none |
| states | packages implementing lorenz96 | lorenz96 | all | default range; n = 131072; states 4, 8, 16, 32, 64, 128 | as perf | timed; both, none; build cold |
| overlap | cubie, cubie_mlir, julia_gpu | all | `tsit5, rosenbrock23_sciml, kvaerno3, kvaerno5, vern7` | perf grid; wp grid; ne grid | fixed dt 2^-10 and default controller tol 1e-8 on the perf grid; wp steppings on the wp grid; ne steppings on the ne grid; cubie adds `pi` with `dirk_defaults` | perf and wp timed, ne numerical; both, none |
| golden | julia_cpu | all | `problems.csv golden_algorithm` per problem | default range; n = 131072 | default controller, tol = `golden_tol`, dt0, dt_min, dt_max package default | numerical; none; precision float64 |

The default ne and wp `dt` and `tol` lists, the perf `n` list and the pins live
in the set files, spelled out. `protocol.toml` keeps `[repeats]`, `[watchdog]`
and `[optimize]` only.

### 1.7 Catalogues

`problems.csv`: `problem, display, states, duration, sweep_parameter,
sweep_min, sweep_max, sweep_scale, golden_algorithm, golden_tol, frameworks`,
read by `sets.py` and by nothing under a runner.

`algorithms.csv`: `algorithm, display, family, order, fixed, adaptive`
(capability per package), read by `sets.py`. The Julia constructor columns move
to `runner_scripts/julia_algorithms.csv`, read by the Julia adapters.

Runners map `problem` to their own system modules and `algorithm` to their own
solver tables; every other quantity comes from the trial.

### 1.8 Analyses

Each analysis under `analyses/` takes `--set <name>` (repeatable), expands it
with `sets.py` under every key present, selects the rows whose `trial_id`
matches, and reads finals where it needs them:

| script | set | output |
|---|---|---|
| `times.py` | perf, states | time vs n per (key, problem, algorithm, controller, transfers); time and `build_s` vs states |
| `wp.py` | wp | error vs time per (key, problem, algorithm) |
| `ne.py` | ne | cubie packages against julia_cpu per setting, both scored against the golden row |
| `overlap.py` | overlap | cubie packages against julia_gpu per setting; report markdown |
| `pairwise.py` | pairwise | finals across packages per problem |

Rows with `errored_pct > 10` are dropped where the column is a number. Absent
`errored_pct`, `reason`, `samples_ms`, `finals`, `package_version` never raise.
Output under `plots/<key>/<problem>/`.

## 2. Packets

One worktree, one branch off `main`, one PR per packet. "Done" is the
acceptance line; "Review" is what the PR is read against.
- A packet replaces what it touches: no compatibility shim, no dual path.
- `main` may not run between packets.

### P1 base cleanup
Merged (#106).

### P2 store library
Merged (#107) on the earlier identity. Reopened as P2b.

### P2b store on the run spec
Depends on: nothing.
`store.py` and `results.jl` carry the 1.2 columns, `run_id`, `trial_id`,
`reference`, `states`, the 1.3 file names, `hash`, `covering`; `mode`,
`setting_kind`, `setting`, `tier`, `grid` gone. `runner_scripts/grid.py`
implements the 1.2 formula and writes the numpy reference file
`runner_scripts/tests/grids/<problem>_131072.npy`; `grid.jl` and a C++
function reproduce it; `test_store.py` and the shim test rewritten.
Done: tests pass; the Julia and C++ grids equal the numpy file bit for bit.
Review: identity is exactly 1.2; hashes match a fixture computed by hand.

### P3 data conversion
Open as #110 on the earlier identity. Reopened as P3b.

### P3b conversion on the run spec
Depends on: P2b.
`convert_legacy.py` maps the legacy CSV trees (main before #110) to 1.2:

| legacy | spec |
|---|---|
| `mode = fixed`, `setting` | `controller = fixed`, `dt = setting`, atol, rtol, dt_min, dt_max NaN |
| `mode = adaptive`, `setting` | `controller = default`, `atol = rtol = setting`, `dt = duration * 2^-10`, `dt_min = duration * 1e-6` (cubie, julia_cpu), NaN (jax, julia_gpu, cpp), `dt_max` NaN |
| implicit fixed rows | `newton_atol = newton_rtol = 1e-6` |
| implicit adaptive rows | `newton_atol = newton_rtol = setting` (cubie, jax, julia_cpu); NaN (julia_gpu) |
| explicit rows | newton NaN |
| `problem`, `states` | `system_params = {"states": states}` for lorenz96, `{}` otherwise; `duration`, `parameter`, `grid_scale`, `grid_min`, `grid_max` from `problems.csv`; `grid_dtype = float32` |
| NE julia rows | `n = 1024`, `grid_max = v[1023]` of the 131072 grid; `precision = float32`; finals kept; `reason = untimed` |
| goldens | julia_cpu rows, `precision = float64`, `algorithm = golden_algorithm`, `atol = rtol = golden_tol`, dt fields NaN, `n = 131072`, finals kept with `converged` false on the retcode sidecar rows, key `windows_RTX-4070-SUPER`, `reason = untimed` |
| overlap tiers `fixed`, `julia` | `controller = fixed` / `default`; `pi` rows `controller = pi`, `gains` from `pi_tier_controller(order)` |
| `julia` | `julia_gpu` |

Drops as in #110: every `cpp` row and `mpgos*.csv`; wp rows with
`transfers != none`; jax `kvaerno3` rows; `julia_*.csv` finals; the overlap
`numerical` phase. `data/numerical/golden_*` deleted after conversion. One data
commit for both keys.
Done: DuckDB counts per (key, package) equal the script's counts; every
converted row hashes to the `run_id` `sets.py` produces for the matching set
member (checked for the perf, wp, ne and golden sets).
Review: no row invented; the drop and mapping tables reproduced in the PR.

### P4 trial model and entry point
Open as #113 on the earlier identity. Reopened as P4b.

### P4b sets, trials and entry point
Depends on: P2b.
`runner_scripts/sets.py` (schema, expansion, merge, reference and reuse rules
of 1.6), the seven set files, `trials.py` (1.4 record, JSONL, legs, ordinals),
`bench.py` (1.6 CLI, run loop, exit-3 handling), `launch.py` runner registry,
`algorithms.csv` and `problems.csv` per 1.7, `julia_algorithms.csv`.
Delete: `--for`, `mode`, `tier`, `grid`, `NE_PACKAGES`, `OVERLAP_PACKAGES`,
the `ne`, `ne_adaptive`, `julia_cpu`, `julia_gpu` columns, `resume.py`,
`resume.jl`, the env-var contract, `protocol.toml` tables other than
`[repeats]`, `[watchdog]`, `[optimize]`.
Done: `test_sets.py` covers every shipped set's expansion counts, the grid_n
max resolution, matched and pi resolution, merge, reference assignment, reuse
against a scratch store, and the narrowing flags; `bench.py plan --set perf`
prints counts.
Review: no trial field outside 1.4; no runner reads a catalogue.

### P5 runner core and cubie
Depends on: P4b.
`runner_scripts/runner.py`: the 1.5 loop on an adapter interface
(`build_leg`, `solve`, `finals`, `compile`, `optimize`, `version`);
`cubie_bench.py` reduced to that adapter, the controller resolved from
`controller` and `gains`; `test_runner.py` with a fake adapter covering every
outcome, the abandon rule, the reference scoring and numerical finals.
Done: tests; `bench.py run --set perf,wp -p cubie -s lorenz -n 32` at tiny n
on the 4070 and the rows read back.
Review: the abandon rule is one function; warm uses `Solver.compile`;
`reason` on every NaN row.

### P6 jax, pytorch, myokit_cuda
Depends on: P5. Adapters on `runner.py`; jax warm keeps `lower().compile()`;
pytorch and myokit reject any `controller` but `fixed`.
Done: tests; tiny-n runs where the platform allows.

### P7 julia_gpu
Depends on: P4b; parallel with P5.
`julia_driver.py` splits the trial file per leg, spawns
`bench_ode_gpu.jl --trials <leg file>` under the GPU lock, applies exit-3
abandon from each progress file. `bench_ode_gpu.jl` consumes trials, records
through `results.jl`, warms at n = 8; `controller` other than `fixed` or
`default` is rejected.
Done: `test_julia_driver.py` on a fake julia; a tiny-n run on the 4070.

### P8 julia_cpu and the golden
Depends on: P4b; parallel with P5 and P7.
`GPU_ODE_Julia/bench_ode_cpu.jl` from `ne_diffeq.jl`: trials in, `float32`
and `float64` per the spec, timings and finals out, `controllers/<problem>.csv`
written per problem, retcode to `converged`. `verify_references.jl` checks the
MTK systems in Float64 against the published values; `reference_systems.jl`
and `generate_golden.jl` go when it passes, else `reference_systems.jl`
becomes the julia_cpu system module.
Done: `bench.py run --set golden -s lorenz` reproduces the converted golden
finals to float64 roundoff; the ne set at n = 1024 lands rows.

### P9 cpp
Depends on: P4b; parallel with P5, P7, P8.
`run_ode_cpp.ps1` and `.sh` take `--trials`; `mpgos_trials.py` lists builds
and points; `Bench.cu` records rows through the store CLI and builds its grid
by the 1.2 formula.
Done: a tiny-n run on Windows lands rows; the Linux script mirrors it.

### P10 analyses
Depends on: P2b, P3b, P4b; parallel with P5 to P9.
`analyses/` per 1.8 on `sets.py`. Delete `runner_scripts/plot/*.jl`,
`compare_numerical_equivalence.py`, `compare_numerical_results.py`,
`run_cubie_julia_overlap.py`, `runner_scripts/cubie_julia_overlap/`,
`run_numerical_equivalence.*`, `ne_common.py`.
Done: every script runs on the converted data and writes figures.

### P11 docs and tests
Depends on: P5 to P10. README to about 100 wrapped lines; `SETUP.md`; this
document reduced to section 1; dead tests and `results.py`, `wp_common`
leftovers removed.

### P12 smoke and reruns
Depends on: P11. `bench.py run --set perf,wp,ne,states,overlap -n 128` on the
4070 for every package; then per key: `cpp` in full; `wp` for julia_gpu, cpp
and myokit_cuda; jax `kvaerno3` (WSL); `pairwise` for julia_gpu; `ne` for
julia_cpu on nand_gate and ring_modulator_index2; `golden` where a problem's
converted golden is absent; cubie and cubie_mlir in full.

## 3. Schedule

```
wave 2:  P2b
wave 3:  P3b | P4b                    (both need P2b)
wave 4:  P5 | P7 | P8 | P9 | P10      (P5, P7, P8, P9 need P4b; P10 needs P3b and P4b)
wave 5:  P6                           (needs P5)
wave 6:  P11, then P12
```

Conflict hotspots: `launch.py` (P4b owns the registry; runner packets add one
line each), `store.py` (P2b only), the set files (P4b only).

## 4. Fixed decisions

- Store: parquet leg files read with DuckDB; no database service; cross-machine
  sync deferred.
- Analyses: Python and matplotlib only.
- Suite interpreter: `GPU_ODE_CUBIE/venv`.
- The golden is a julia_cpu float64 row, not a separate artefact.
