# Unification plan

One entry point generates trials, one runner per package executes them, one
store holds every row, and analyses are offline views over the store. Every
packet is built and reviewed against section 1.

## 1. Contracts

### 1.1 Packages

`cubie`, `cubie_mlir`, `jax`, `pytorch`, `myokit_cuda`, `cpp`, `julia_gpu`
(DiffEqGPU kernels), `julia_cpu` (DifferentialEquations.jl on
`EnsembleThreads`). Package names are the store partition names.

`package_version` per package: cubie `importlib.metadata.version("cubie")` plus
the backend name; jax `jax.__version__`; pytorch `torch.__version__` plus the
torchdiffeq fork commit; myokit_cuda `myokit.__version__`; cpp the 12-character
MPGOS source hash plus the nvcc release; julia_gpu the DiffEqGPU version from
`Manifest.toml`; julia_cpu the OrdinaryDiffEq version. `suite_rev` is
`git rev-parse --short HEAD`, suffixed `-dirty` when the tree is dirty.

### 1.2 Store

Layout (Hive partitions, DuckDB `read_parquet(..., hive_partitioning = true)`):

```
data/key=<os>_<gpu>/package=<pkg>/results/<problem>__<algorithm>__<mode>.parquet
data/key=<os>_<gpu>/package=<pkg>/finals/<problem>__<algorithm>__<mode>__<kind>-<setting>__n<n>__s<states>__<tier>.parquet
data/key=<os>_<gpu>/package=julia_cpu/controllers/<problem>.csv
data/numerical/golden_<problem>_131072.csv            (unchanged, plus retcode sidecars)
data/clocks/                                          (unchanged)
```

`<setting>` is `format(setting, ".10g")`. One results file per leg
`(package, problem, algorithm, mode)`; all axes (N, dt or tol, states) of that
leg live in the one file.

Row schema (Arrow types; every column present in every file):

| column | type | notes |
|---|---|---|
| package, key, problem, algorithm, mode | string | identity |
| setting_kind | string | `dt` or `tol`; identity |
| setting | float64 | identity, matched at rel 1e-8 |
| n | int64 | identity |
| states | int32 | identity |
| tier | string | `default`, `matched`, `pi`; identity |
| transfers | string | `both` or `none`; identity |
| min_ms | float64 | NaN when not timed |
| samples_ms | list<float64> | every attempt in ms, warm-up first; empty when none |
| errored_pct | float64 | NaN when unknown |
| error | float64 | ensemble l2 against the golden; NaN when no golden applies |
| build_s | float64 | cold build seconds; NaN unless measured |
| reason | string | empty on success; see 1.4 |
| finals | string | path relative to the package dir; empty when none |
| package_version, suite_rev | string | |
| recorded_utc | timestamp[us, UTC] | |

No `analysis` column: the views in 1.6 select rows by their settings. Absent
rows are gaps; no placeholder rows are ever written. A record with the same
identity replaces the row; under `--floor` the lower finite `min_ms` stays and
NaN never wins.

Finals file schema: `traj int32, s1..sk float32, converged bool`. Rows equal the
trial's `finals` count and are the first rows of the trial's grid. The golden
comparison uses the same rows of `golden_<problem>_131072.csv`.

`runner_scripts/store.py` is the only writer and the Python reader:

```
Store(root="data").record(row, floor=False)            # one row, atomic leg-file swap under a mkdir lock
Store.record_finals(identity, finals, converged) -> relative path
Store.status(identity) -> "absent" | "nan" | "finite"
Store.rows(sql_where="", **eq_filters) -> list[dict]     # DuckDB over the whole tree
python store.py record  <rows.json | ->                    # JSON array of rows; samples_ms as a list
python store.py finals  <identity.json> <finals.csv>       # prints the relative path
python store.py status  <identity.json>
python store.py query   "<sql over results>"               # CSV on stdout
python store.py clear   <filter.json>
```

`results.jl` serialises rows with JSON.jl and calls the CLI; `Bench.cu` calls
the CLI. Packet 2 tries Parquet2.jl; the Julia writers use it directly only if
it round-trips through DuckDB.

The suite interpreter is `GPU_ODE_CUBIE/venv` with `pyarrow` and `duckdb`
installed, exposed as `launch.suite_python()`. `bench.py`, the analyses and the
shell wrappers run under it.

### 1.3 Trials

The entry point writes one JSONL file per package; a runner takes that path and
nothing else. One line per trial:

```json
{"id": "cubie/lorenz/tsit5/adaptive/tol=1e-05/n=32768/s=3/default",
 "kind": "solve",                       "solve" | "warm" | "optimize"
 "package": "cubie", "problem": "lorenz", "algorithm": "tsit5", "mode": "adaptive",
 "setting_kind": "tol", "setting": 1e-05, "n": 32768, "states": 3, "tier": "default",
 "transfers": ["both", "none"],         legs to time, in this order
 "grid": "sweep",                       "sweep" = n points over the range; "prefix" = first n points of the N_WP sweep
 "finals": 32768,                       rows of final state to save, 0 for none
 "controller": null,                    cubie controller dict for matched / pi tiers
 "leg": "lorenz/tsit5/adaptive/n",      (problem, algorithm, mode, axis); axis in n | setting | states
 "ordinal": 6}                          cost order within the leg
```

Ordinal order is ascending cost: N ascending, dt descending, tol descending,
states ascending. `error` is computed for `grid = prefix` trials only; `sweep`
trials record NaN. `warm` trials are never recorded; `optimize` trials exist for
cubie packages only and record to `optimize.csv`.

### 1.4 Runner contract

```
<runner argv> --trials <path> [--floor]
```

Every runner, in every language:

1. Reads the JSONL, groups `solve` trials by `leg`, builds once per leg, walks
   ordinals ascending, and times each requested transfer leg with one untimed
   warm-up followed by the protocol repeat schedule (`timed_min_ms` semantics).
2. Writes `<trials>.progress` (`{"id": ..., "started_utc": ...}`) before starting
   each trial.
3. Records every finished trial through the store before starting the next.
4. Applies one abandon rule. Outcomes per (trial, transfers) are `ok`,
   `timeout` (soft cap, run returned), `oom`, `error`. After `timeout` or `oom`
   at ordinal k, every higher ordinal of the same leg with the same transfers is
   recorded as NaN with `reason = "abandoned: <timeout|oom> at ordinal k"` and
   not run. An `error` records `reason = "error: <Type>: <message[:200]>"` and
   the leg continues. A `none` leg failing after a good `both` leg marks only the
   `none` row. OOM is classified by exception type or message (CUDA
   `OUT_OF_MEMORY`, numba `CUDA_ERROR_OUT_OF_MEMORY`, XLA `RESOURCE_EXHAUSTED`,
   torch `OutOfMemoryError`, Julia `CuError(OUT_OF_MEMORY)`).
5. Exits 0 when the loop completed, 3 on a watchdog hard exit
   (`wp_common.run_watchdogged`, `watchdog.jl`), anything else on a crash. On
   exit 3 the driver reads the progress file, records
   `reason = "abandoned: hard-exit at ordinal k"` for the higher ordinals of that
   leg, and re-invokes the runner with the trials that still have no row. On any
   other non-zero exit the driver records the summary line and moves to the next
   package.
6. `warm` trials: cubie `Solver.compile(...)`; jax
   `jit(f).lower(args).compile()` at the trial's n; MPGOS nvcc into the build
   cache; Myokit `load_model`; julia_gpu one solve at n = 8 in the leg's
   process, off the GPU lock; pytorch none. Warm trials are never recorded.
   States legs are never warmed.
7. Records `package_version` and `suite_rev` on every row.

States legs follow the same abandon rule as every other leg; there is no
compile budget. Resume is a filter at generation time; floor is the runner flag
above; runners read no environment variables for either.

### 1.5 Entry point

```
bench.py plan|run [-p pkgs] [-s problems] [-g algorithms] [--mode fixed|adaptive|all]
                  [--for perf,wp,ne,states,overlap] [-n <ceiling|list>] [--setting <list>]
                  [--states <list>] [--transfers both,none] [--tier <list>]
                  [--point <trial id>]... [--resume | --no-overwrite] [--floor]
                  [--cooldown S] [clock flags as today]
```

`plan` writes `trials/<key>/<package>.jsonl` and prints counts per package and
leg; `run` writes the same under `logs/<key>_<stamp>/` and drives the runners.
`--for` predicates expand to trials from `algorithms.csv`, `problems.csv` and
`protocol.toml`; the union is deduplicated by identity, merging `transfers` and
taking the larger `finals`:

| view | packages | algorithms | grid and settings | n | transfers | finals |
|---|---|---|---|---|---|---|
| perf | all | `fixed`/`adaptive` membership | sweep; timing dt or `timing_tol` | N list | both, none | 32768 rows at n = 32768 |
| wp | all | membership | prefix; `dts(algorithm)` or `TOLS` | `n_wp` | none | 1024 rows for cubie NE members |
| ne | cubie, cubie_mlir, julia_cpu | `ne` / `ne_adaptive` rows | prefix; `ne_dts` (non-erk) and `TOLS` | `n_ne` for julia_cpu, `n_wp` for cubie | none | 1024 |
| states | packages listing lorenz96 | membership | sweep; timing setting | `n_states` | both, none | 0 |
| overlap | cubie, cubie_mlir, julia_gpu | rows with `julia_gpu` | sweep at timing dt and `overlap_tol` over the N list; prefix wp grid; prefix NE grid at `n_ne` for julia_gpu | as listed | both, none | 1024 on the NE grid |

Tiers: cubie NE adaptive trials get `default` plus `matched` when
`controllers/<problem>.csv` from julia_cpu holds constants that differ from the
shipped defaults (the `controller` dict is embedded in the trial); overlap
adaptive trials get `default` plus `pi`. `optimize` trials are emitted for cubie
packages per `[optimize]`; `warm` trials for every leg of packages that compile.

Filters: `--resume` drops solve trials whose every requested transfers row is
present; `--no-overwrite` drops those whose rows are all finite; without either,
rows are replaced as they are recorded.

The run loop keeps the clock guard, log directory, manifest and summary.
`bench.py` never invokes an analysis.

### 1.6 Views

All analyses live in `analyses/`, run under the suite interpreter, read only the
store and the goldens, tolerate absent `errored_pct`, `reason`, `samples_ms`,
`finals`, `package_version`, and write under `plots/<key>/<problem>/`:

| script | rows selected |
|---|---|
| `times.py` | setting = timing setting of (problem, mode), tier default, states = problem states; one figure per (key, problem, mode, algorithm, transfers) |
| `wp.py` | n = `n_wp`, transfers none, tier default, finite positive error; error vs time |
| `states.py` | problem lorenz96, n = `n_states`, transfers both; time and `build_s` vs states |
| `ne.py` | finals of cubie packages against julia_cpu at n = `n_ne` on the NE grids, tiers default and matched, masked by `converged` |
| `overlap.py` | cubie packages against julia_gpu at the overlap settings, tiers default and pi, both transfer legs, plus the NE-grid finals; report markdown |
| `pairwise.py` | finals at n = 32768 across packages per problem |

Rows with `errored_pct > plots.max_errored_pct` are dropped where the column is
a number.

## 2. Packets

One worktree, one branch off `main`, one PR per packet. "Done" is the
acceptance line; "Review" is what the PR is read against. A packet replaces
what it touches completely: no compatibility shim, no dual code path, no
partial refactor. Runs on `main` are expected to be broken between packets.

### P1 base cleanup
Depends on: nothing.
Delete: `runner_scripts/cpu/`, `GPU_ODE_Julia/bench_cpu.jl`,
`bench_ensemblegpuarray.jl`, `bench_multi_device.jl`, `sde_examples/`, `MPI/`,
`runner_scripts/gpu/run_sde_*`, `run_egarray_julia.*`, `run_ode_mult_device.*`,
`plot_cpu_comp.jl`, `plot_mult_gpu.jl`, `plot_sde_*.jl`, `paper_artifacts/data`,
the `-d`/`-m` routing in `run_benchmark.sh` and `.bat`, the states compile budget
and marker in `julia_driver.py` and `bench_ode_gpu.jl`, `.idea/`.
Done: tests pass; no file references a deleted path (grep).
Review: nothing else changed; README sections for the deleted paths removed.

### P2 store library
Depends on: nothing (parallel with P1).
Add: `runner_scripts/store.py` per 1.2 with tests (`test_store.py`: schema,
upsert, floor, setting tolerance, lock, finals round-trip, DuckDB read across
keys, CLI); `results.jl` rewritten as the JSON shim with a Julia round-trip test
script; pyarrow and duckdb added to `GPU_ODE_CUBIE/setup_environment.py`;
`suite_python()` in `launch.py`. Try Parquet2.jl in the Julia project and record
the outcome in the PR. `results.py` stays untouched until the runners are ported.
Done: `python -m pytest runner_scripts/tests/test_store.py` and the Julia shim
test pass under the suite interpreter.
Review: schema matches 1.2 column for column; no `analysis` column; NaN never
wins under floor; the CLI accepts a list in `samples_ms`.

### P3 data conversion
Depends on: P2.
Add: `runner_scripts/convert_legacy.py` and one data commit, both keys.
Convert:
- `data/<PKG>/<key>/results.csv` rows: `analysis` dropped, `julia` renamed
  `julia_gpu`.
- overlap `*_timings.csv` rows: tier `fixed` to `default`, `pi` kept,
  `work_precision` at n = 131072 on the prefix grid, `golden_rmse` to `error`,
  no samples.
- NE `julia/` trees: `julia_cpu` finals at n = 1024, row `reason = "untimed"`.
- NE `controller_constants.csv`: `controllers/<problem>.csv`.
- `data/numerical/<key>/<problem>/` files: finals on the n = 32768 times row;
  cubie `_unadaptive` = classical-rk4 fixed, `_adaptive` = tsit5 adaptive,
  `jax.csv` = tsit5 fixed, `pytorch.csv` = classical-rk4 fixed,
  `myokit_cuda.csv` = euler fixed.
Drop: every `cpp` row and `mpgos*.csv`; wp rows with `transfers != none`; jax
`kvaerno3` rows; `julia_fixed.csv` and `julia_adaptive.csv`; overlap
`numerical` phase metrics and finals. Old trees deleted; goldens untouched.
Done: DuckDB counts per (key, package) equal the script's converted counts;
the PR body carries the converted and dropped counts per rule.
Review: no row invented; dropped inputs listed; `.gitignore` no longer ignores
anything under the new layout.

### P4 trial model and entry point
Depends on: P2.
Add: `runner_scripts/trials.py` (dataclass, JSONL read and write, `leg` and
`ordinal` derivation, identity and id), the `--for` expansion per 1.5,
`bench.py` rewritten with `plan` and `run`, the run loop per 1.4 step 5, the
runner registry in `launch.py` (package to argv; unported packages raise a
named error), `algorithms.csv` with `julia` replaced by `julia_gpu` and
`julia_cpu` added to the `fixed`/`adaptive` membership of every `ne` row.
Delete: `resume.py`, `resume.jl`, `--resume-from`, the env-var contract, the
`Point` class and `ne_package()`.
Done: `test_trials.py` covers expansion for every view, deduplication, ordinal
order, resume filters against a scratch store; `bench.py plan` on the real
tables prints counts.
Review: every trial field of 1.3 present; the abandon-on-exit-3 loop is in
`bench.py`, not in a runner; no analysis is invoked.

### P5 runner core and cubie
Depends on: P4.
Add: `runner_scripts/runner.py`, the shared loop of 1.4 built on `timed_min_ms`,
with an adapter interface (`build_leg`, `solve`, `finals`, `compile`,
`optimize`, `version`); `cubie_bench.py` reduced to that adapter; NE finals go
through `Store.record_finals`; the `numerical_names` 32768 special case goes;
`test_runner.py` with a fake adapter covering every outcome and the abandon rule.
Delete: `wp_common.parse_bench_args`, `results.Leg` users in cubie.
Done: tests pass; `bench.py run -p cubie -s lorenz -n 32 --for perf,wp` completes
at tiny N on the 4070 and the rows read back through DuckDB.
Review: the abandon rule is one function; warm uses `Solver.compile`; optimize
still records to `optimize.csv`; `reason` is set on every NaN row.

### P6 jax, pytorch, myokit_cuda
Depends on: P5.
Each bench script becomes an adapter on `runner.py`; jax warm keeps
`lower().compile()`, Myokit warm builds the module only, pytorch warm is a no-op
trial. `WP_PER_LEG` process splitting stays as the driver's per-leg invocation.
Done: tests; tiny-N runs where the platform allows (jax on WSL).
Review: no runner-local sweep logic remains; OOM classification per 1.4.

### P7 julia_gpu
Depends on: P4 (contract) and P2; may start alongside P5.
`julia_driver.py` reads the trial file, writes one JSONL per leg, spawns
`bench_ode_gpu.jl --trials <leg file>` under the GPU lock with
`BENCH_JULIA_JOBS` concurrency, applies the exit-3 abandon from each process's
progress file. `bench_ode_gpu.jl` consumes trials, records through `results.jl`,
implements the abandon rule, warms at n = 8. `julia_systems.jl` unchanged.
Delete: `resume.jl` users, the 32768 `write_finals` special case, the states
budget remnants.
Done: `test_julia_driver.py` on a fake julia; a tiny-N run on the 4070.
Review: the Julia abandon rule matches `runner.py` outcome for outcome.

### P8 julia_cpu
Depends on: P4 and P2; parallel with P5 and P7.
Add: `GPU_ODE_Julia/bench_ode_cpu.jl` from `ne_diffeq.jl`: trials in, timings
and 1024-row finals out through the store, `controllers/<problem>.csv` written
per problem, Float32 discipline kept, `EnsembleThreads`.
Delete: `ne_diffeq.jl`, `ne_common.py` writers and file-path helpers,
`ne_grid.jl` if unused.
Done: a lorenz run at n = 1024 lands finals and rows; the NE tests updated.
Review: the sweep is the prefix grid; `converged` comes from the retcode.

### P9 cpp
Depends on: P4 and P2; parallel with P5, P7, P8.
`run_ode_cpp.ps1` and `.sh` take `--trials`; a Python helper
(`mpgos_trials.py`) lists the builds and points from the file; `Bench.cu`
records rows with `reason`, `package_version` and `suite_rev` through the store
CLI; warm builds every binary the file needs.
Delete: `Test-ResumeSkip`, `Add-NanRow` paths replaced by the abandon rule.
Done: a tiny-N run on Windows lands rows; the Linux script mirrors it.
Review: exit 42 (device budget) maps to `timeout`; no point is silently absent.

### P10 analyses
Depends on: P2 and P3; parallel with P5 to P9.
Add: `analyses/` per 1.6 in Python with matplotlib, one CLI each, plus
`analyses/common.py` for the DuckDB views and the group logic (per key, per os,
per gpu, all). Delete: `runner_scripts/plot/*.jl`,
`compare_numerical_equivalence.py`, `compare_numerical_results.py`,
`run_cubie_julia_overlap.py`, `runner_scripts/cubie_julia_overlap/`,
`run_numerical_equivalence.*`, `ne_common.py` readers.
Done: every script runs on the converted data and writes figures; tests for
the view selectors on a scratch store.
Review: no analysis reads anything but the store and goldens; missing columns
do not raise.

### P11 membership, docs, tests
Depends on: P5 to P10.
`README.md` rewritten to about 100 wrapped lines (what, setup, `bench.py`,
store, analyses, goldens); `SETUP.md` aligned; `docs/unification-plan.md`
reduced to the contracts section; dead tests removed; `results.py`,
`wp_common` leftovers and `launch.py` command builders removed.
Done: tests pass; grep finds no reference to a removed script.

### P12 end-to-end smoke and reruns
Depends on: P11.
`bench.py run -n 128 --for perf,wp,ne,states,overlap` on the 4070 for every
package; fallout fixed in small PRs. Then the reruns as data PRs, per key:
`cpp` in full; `wp` for julia_gpu, cpp and myokit_cuda; jax `kvaerno3` in full
(WSL); julia_gpu finals at n = 32768; julia_cpu `ne` for nand_gate and
ring_modulator_index2; overlap `ne`-grid finals for julia_gpu; cubie and
cubie_mlir in full.

## 3. Schedule

```
wave 1:  P1 | P2
wave 2:  P3 | P4                      (both need P2)
wave 3:  P5 | P7 | P8 | P9 | P10      (P5, P7, P8, P9 need P4; P10 needs P3)
wave 4:  P6                           (needs P5)
wave 5:  P11, then P12
```

Conflict hotspots: `launch.py` (P4 owns the registry; runner packets each add
one line), `algorithms.csv` (P4 only), `results.jl` (P2 only), `README.md`
(P1 removes sections, P11 rewrites; runner packets do not touch it).

## 4. Fixed decisions

- Store: parquet leg files read with DuckDB; no database service; cross-machine
  sync deferred.
- Analyses: Python and matplotlib only.
- Suite interpreter: `GPU_ODE_CUBIE/venv`.
