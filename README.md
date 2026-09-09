# GPUODEBenchmarks
Comparison of Julia's GPU-based ensemble ODE solvers with other
open-source implementations in C++, JAX, PyTorch, CUBIE, and Myokit
CUDA. These artifacts are part of the paper:
> Automated Translation and Accelerated Solving of Differential Equations on Multiple GPU Platforms

**_NOTE:_**  This repository is meant to contain scripts for benchmarking existing ensemble ODE solvers. For external purposes, one can directly use the solvers from the respective libraries. 

### Performance comparison with other open-source ensemble ODE solvers
<img src="https://github.com/utkarsh530/GPUODEBenchmarks/blob/main/paper_artifacts/figures/Lorenz_unadaptive.png" alt="drawing" width="50%"/>

### Works with NVIDIA, Intel, AMD, and Apple GPUs
<img src="https://github.com/utkarsh530/GPUODEBenchmarks/blob/main/paper_artifacts/figures/Multi_GPU_unadaptive.png" alt="drawing" width="50%"/>

# Reproduction of the benchmarks

The methods are written in Julia and are part of the repository
<https://github.com/SciML/DiffEqGPU.jl>. The benchmark suite also
consists of the raw data, such as simulation times and plots mentioned
in the paper. The benchmark suite is supported on Linux, Windows, and macOS.

**Windows Users:** Windows batch (.bat) versions of all run scripts are provided alongside the bash (.sh) scripts. All benchmark commands documented below have both Linux/macOS and Windows examples.

## Quick Setup (Cross-Platform)

For a streamlined setup experience on any platform, use the Python-based setup scripts:

```bash
python3 setup_all_environments.py
```

The Julia environment is pinned: `Project.toml` and `Manifest.toml` are
committed and `setup_julia.py` instantiates that exact version set. Pass
`--update` to re-resolve to the newest compatible releases and rewrite both
files.

This will set up all environments (CUBIE, CUBIE-MLIR, JAX, PyTorch,
Myokit-CUDA, and Julia) automatically. For more details and individual
package setup instructions, see [SETUP.md](SETUP.md).

## Installing Julia

Firstly, we will need to install Julia. The user can download the
binaries from the official JuliaLang website
[`https://julialang.org/downloads/`](https://julialang.org/downloads/).
Alternatively, one can use the convenience of a Julia version
multiplexer, <https://github.com/JuliaLang/juliaup>. The recommended OS
for installation is Linux. The recommended Julia installation version is
v1.8. To use AMD GPUs, please install v1.9. The Julia installation
should also be added to the user's path.

## Setting up DiffEqGPU.jl

### Installing backends

The user must install the GPU backend library for testing
DiffEqGPU.jl-related code.

```julia
    julia> using Pkg
    julia> #Run either of them
    julia> Pkg.add("CUDA") # NVIDIA GPUs
    julia> Pkg.add("AMDGPU") #AMD GPUs
    julia> Pkg.add("oneAPI") #Intel GPUs
    julia> Pkg.add("Metal") #Apple M series GPUs
```
### Continuous Integration and Development

DiffEqGPU.jl is a fully featured library with regression testing, semver
versioning, and version control. The tests are performed on cloud
machines having a multitude of different GPUs
[`https://buildkite.com/julialang/diffeqgpu-dot-jl/builds/705`](https://buildkite.com/julialang/diffeqgpu-dot-jl/builds/705).
These tests are approximately complete in 30 minutes. The publicly visible
testing framework serves as a testimonial of compatibility with multiple
platforms and said features in the paper.

## Testing GPU-accelerated ODE Benchmarks with other programs

### Running All Benchmarks

To run all GPU ODE benchmarks (Julia, C++, JAX, PyTorch, CUBIE,
CUBIE-MLIR, and Myokit-CUDA) sequentially in one command:

**On Linux/macOS:**
```bash
    $ bash ./run_all_benchmarks.sh
```

**On Windows:**
```cmd
    > run_all_benchmarks.bat
```

This script will execute all benchmarks one after another, allowing for set-and-forget benchmarking. The optional `-n N` flag sets the upper bound of the trajectory sweep (8, 32, ... ≤ N); a comma list runs exactly those trajectory counts instead:

**On Linux/macOS:**
```bash
    $ bash ./run_all_benchmarks.sh -n $((2**20))
    $ bash ./run_all_benchmarks.sh -n $((2**23)),$((2**27))   # only N = 2^23 and 2^27
```

**On Windows:**
```cmd
    > run_all_benchmarks.bat -n 1048576
    > run_all_benchmarks.bat -n 8388608,134217728
```

Each benchmark typically takes around 20 minutes, so running all of them may take several hours. The script will continue running subsequent benchmarks even if one fails.

`-a` selects which analysis to run:

* `-a work-precision` — the work-precision (error vs. runtime) sweeps for every
  package and their plot (see [Analyses](#analyses)).
* `-a numerical` — the numerical-equivalence suite (cubie vs.
  DifferentialEquations.jl, error vs. dt and vs. tolerance per algorithm; see
  [Analyses](#analyses)).
* `-a all` — every analysis above, plus the timing sweeps.

`-p`, `-a`, `-g` and `-n` accept comma lists selecting a subset of packages,
analyses, algorithms and trajectory counts:

```bash
    $ bash ./run_all_benchmarks.sh -p cubie,julia -a performance,work-precision \
          -g euler,tsit5 -n $((2**23)),$((2**27))
```

`-p` restricts the run to the listed packages; `-g <algorithms>` restricts the
timing and work-precision sweeps to the listed integration algorithms (see
[Algorithm-matched subsets](#algorithm-matched-subsets) below).

### Algorithm-matched subsets

`runner_scripts/algorithms.csv` is the algorithm registry, one row per
integration algorithm in the cubie vocabulary: `fixed` and `adaptive` name the
frameworks that time it in each mode, `ne` and `ne_adaptive` place it in the
numerical-equivalence sweeps, `julia_cpu` is its DifferentialEquations.jl
constructor, and `julia_gpu` its DiffEqGPU constructor for the overlap suite.
Both `algorithms.py` and `algorithms.jl` read that file, and every suite takes
its set from it. Each timing figure contains only packages running the same
method:

| Subset | Mode | Algorithm | Members |
|---|---|---|---|
| A | fixed | `euler` | CUBIE, CUBIE_MLIR, JAX, PYTORCH, MYOKIT_CUDA |
| B | fixed | `classical-rk4` | CUBIE, CUBIE_MLIR, JAX, PYTORCH, MPGOS |
| C | fixed | `tsit5` | CUBIE, CUBIE_MLIR, JAX, PYTORCH, Julia |
| D | adaptive | `tsit5` | CUBIE, CUBIE_MLIR, JAX, Julia |
| E | adaptive | `cash-karp-54` | CUBIE, CUBIE_MLIR, MPGOS |
| F | fixed | `rosenbrock23_sciml` | CUBIE, CUBIE_MLIR, Julia |
| G | adaptive | `rosenbrock23_sciml` | CUBIE, CUBIE_MLIR, Julia |
| H | fixed | `kvaerno3` | CUBIE, CUBIE_MLIR, JAX, Julia |
| I | adaptive | `kvaerno3` | CUBIE, CUBIE_MLIR, JAX, Julia |
| J | fixed | `radau_iia_5` | CUBIE, CUBIE_MLIR |
| K | adaptive | `radau_iia_5` | CUBIE, CUBIE_MLIR |

Myokit exposes Euler only and MPGOS exposes RK4/RKCK45 only, so no single
figure can contain every package. JAX's classical RK4 and PyTorch's
fixed-grid Tsit5 are custom solvers built from the standard tableaus inside
the bench scripts. Julia's implicit entries are the DiffEqGPU kernel solvers
`GPURosenbrock23` and `GPUKvaerno3` with `autodiff=Val(false)`, and JAX's is
`diffrax.Kvaerno3`. Subsets D, G and I match the tableau but not the error
controller: each framework uses its own step controller, so step counts
differ at equal tolerance.

Every algorithm is run against every problem its framework defines. An
algorithm that cannot integrate a system records a NaN time and a NaN error
for that point and the sweep continues; the plots drop non-finite points.

### Adaptive settings

Every framework is given the same tolerance and its own step controller: the
comparison is what each package delivers for a requested accuracy, which is
why the figures plot achieved error rather than step counts. Adaptive points
take `atol = rtol` from `adaptive.timing_tol` in `runner_scripts/protocol.toml`
for the N-sweep and from its `tol_k` grid for work-precision, and start from the
problem's timing dt; cubie and the DifferentialEquations.jl NE sweep floor the
step at `duration * adaptive.dt_min_fraction`, DiffEqGPU's kernels at 1e-14.
Nothing else is set: every package runs its shipped step-controller defaults.

Controllers are matched in one place only, the cubie against
DifferentialEquations.jl overlap suite, which repeats each comparison with
cubie's controller set to the DIRK PI defaults (`pi_tier_controller` in
`runner_scripts/cubie_adapter.py`).

`eps(Float32)` is 1.2e-7, so the tightest points of the tolerance grid and
the 1e-8 `TIMING_TOL` ask for more than the working precision resolves.

### Implicit stage solves

The `[newton]` table of `runner_scripts/protocol.toml` scales the fixed-step
Newton termination test, `eta * rms(dz / (atol + rtol |u|)) < 0.01`, in
OrdinaryDiffEq (`abstol`/`reltol` with `adaptive = false`), diffrax
(`VeryChord(rtol, atol, norm = rms_norm)`) and cubie
(`newton_atol`/`newton_rtol`). Adaptive solves scale it by the step tolerance
in all three; cubie warns `newton_rtol is at or above the step controller
rtol` at every implicit solver build and floors `newton_rtol` at
`4 eps(Float32)`. DiffEqGPU's kernels stop at an unscaled residual rms below
`100 eps(Float32)` and take no tolerance.

All benchmark entry points accept `-g <algorithms>` (default `all`, meaning
every algorithm the framework supports; a comma list runs the listed ones);
a framework that does not support a requested algorithm skips cleanly:

```bash
    $ bash ./run_benchmark.sh -p cubie -g tsit5
    $ bash ./run_benchmark.sh -p cubie -g euler,tsit5
    $ bash ./run_all_benchmarks.sh -g classical-rk4
    $ ./run_full_dataset.sh --algorithm euler
```

Every timed point is one row of `data/<package>/<os>_<gpu>/results.csv`, the
result store described under "Result store" below.

### Repeat count

Every timed leg is a minimum over repeated runs after one untimed warm-up.
The first timed run picks a floor and ceiling from the schedule below; the
leg always runs to the floor and extends toward the ceiling while
median/min − 1 is above 2%.

| first timed run | floor | ceiling |
| --- | --- | --- |
| < 100 ms | 20 | 20 |
| 0.1 – 3 s | 10 | 10 |
| 3 – 5 s | 5 | 10 |
| > 5 s | 3 | 10 |

The schedule, its spread and the repeat cap (`repeats.cap`, 20) are the
`[repeats]` table of `runner_scripts/protocol.toml`.

### Protocol file

`runner_scripts/protocol.toml` holds every grid, tolerance, ensemble size,
repeat rule and watchdog value. Python reads it through
`runner_scripts/protocol.py`, Julia through `runner_scripts/protocol.jl`, and
the MPGOS launchers generate `GPU_ODE_MPGOS/protocol.h` from it before each
build. `python runner_scripts/protocol.py get <table.key>` prints one value.

### Result store

`data/<package>/<os>_<gpu>/results.csv` holds one row per timed point and
transfer leg, written by `runner_scripts/results.py` (Python writers and the
MPGOS launcher and binary) and `runner_scripts/results.jl` (Julia writers).
The identity columns are `package, key, analysis, problem, algorithm, mode,
setting_kind, setting, n, states, tier, transfers`; `analysis` is `times`,
`wp` or `states`, `setting` the dt or tolerance the point ran at, `n` the
ensemble size and `states` the state count. `transfers` is `both` (h2d and
d2h) or `none`; the N and states sweeps record both legs, work-precision
`none` only. The value columns are `min_ms`, `samples_ms` (every attempt in
ms, `;`-joined, warm-up first; `min_ms` is the minimum after the warm-up),
`errored_pct`, `error` (wp rows) and `build_s` (states rows). Readers compute
spread from `samples_ms` (`results.samples_of`, `result_samples`). A row with
the same identity replaces the recorded one; under `--floor` the lower
`min_ms` stays. `python runner_scripts/results.py` offers `record`, `nan`,
`status` and `clear`.

### Problems

`runner_scripts/problems.csv` is the problem registry: one row per benchmark
ODE or DAE, giving its state count, duration, swept parameter, range and
scale, stiffness class, DAE index, dt-grid exponents, golden method and
tolerance, and the frameworks expected to run it. Both `problems.py` and
`problems.jl` read that one file, and every dt grid is a dyadic fraction of
the problem's duration so dt, save and end boundaries stay exact in binary
floating point.

| Problem | States | Duration | Swept parameter | Class |
|---|---|---|---|---|
| `lorenz` | 3 | 1 | `rho` over [0, 21], linear | non-stiff |
| `lorenz96` | 32 | 1 | `F` over [0, 16], linear | non-stiff |
| `lorenz96_20` | 20 | 1 | `F` over [0, 16], linear | non-stiff |
| `pleiades` | 28 | 3 | `m1` over [0.5, 2], linear | non-stiff |
| `pollu` | 20 | 60 | `k1` over [3.5e-2, 3.5], log | stiff |
| `ring_modulator` | 15 | 1e-3 | `Cs` over [2e-13, 2e-9], log | stiff |
| `ring_modulator_index2` | 15 | 1e-3 | `Uin1_amplitude` over [0, 0.5], linear | stiff, index 2 |
| `nand_gate` | 14 | 80 | `c9` over [2.5e-5, 1e-4], linear | implicit DE |

Except for the two Lorenz systems, the problems come from Mazzia and
Magherini's Bari *Test Set for IVP Solvers*, transcribed from its Fortran
sources with their canonical initial states and intervals. Lorenz 96 is the cyclic 32-state forcing model; the Pleiades is the seven-body celestial
mechanics problem with masses (m1, 2, ..., 7); the pollution problem is
Verwer's 25-reaction atmospheric mechanism.

Swept ranges are deliberately wide enough that solvers fail inside them. The
Pleiades `m1` range is the clearest case: past roughly m1 = 1.5 the mass
perturbation drives two-body encounters whose closest approach falls below
Float32 resolution, and adaptive integrators pin at their minimum step for the
rest of the solve. That is the intended test, not an accident — narrowing the
range to keep every trajectory comfortable would report a solver as converging
on a set chosen so it cannot fail. A run that bottoms out on the step floor is
a *failed* run and should be discounted the same way an errored Julia or cubie
solve is, rather than excluded in advance by shrinking the parameter space. An
earlier commit message (`db2f6ca`) describes narrowing this range to
[0.9, 1.1]; that narrowing was intentionally reverted and the message is stale.

The performance sweep runs each package's whole ascending N list inside
one process, one `(problem, algorithm, mode)` leg at a time: kernels
compile once per leg and only the per-size ensembles are rebuilt. MPGOS's
trajectory count is a compile-time constant, so it rebuilds per point, one
solver at a time.

Every benchmark solve runs under `BENCH_WATCHDOG_SECONDS` (default 120): a
run over the cap is recorded as a NaN row and the leg's remaining solves
are abandoned — the remaining work-precision settings, or the remaining
trajectory counts of an N sweep. MPGOS kernels end themselves through a
device-side cycle budget in `problems/stubs.cuh`; a solve that never
returns is caught by a hard watchdog that records every row its process
can no longer reach as NaN and exits with status 3, and the Julia runner
launches one process per (problem, algorithm, mode) so an exit abandons
only that leg. MPGOS runs each sweep size as its own process; a breach
exits with code 42 and the runner NaN-fills the leg's remaining sizes.

Every problem attempts every algorithm its frameworks support; a failed solve is a NaN row. `lorenz96_20` is the 20-state lorenz96 row, the smaller stiff head-to-head.

### States sweep

`run_benchmark -a states` times lorenz96 at 4-128 states
(`BENCH_STATES_GRID=<comma list>` overrides) and a fixed
131072-trajectory ensemble, in every framework and algorithm
the problem's frameworks support, exclusions included. Rows land in the
result store with `analysis = states`, `problem = lorenz96` and the state
count in `states`. `build_s` is the wall time from solver construction to
the first completed solve; the sweep bypasses every compiled-kernel
cache, making it a cold compile on every run. A size with no finite time in
either mode cancels the pending and running larger sizes of that
algorithm; cancelled rows are NaN.

Every Julia analysis runs through `runner_scripts/gpu/julia_driver.py`:
one process per leg — (problem, algorithm, mode) for performance and
work-precision, (size, algorithm) for states — with up to
`BENCH_JULIA_JOBS` (default 4) compiling concurrently while a pidfile
lock serializes every timed GPU section; each leg's first solve carries
its kernel compile outside the lock.

### Compiled-kernel caches

Cubie persists generated source and compiled kernels under `generated/`
(both backends). JAX writes XLA binaries to a persistent compilation
cache under `generated/jax_cache`. Myokit compiles through CuPy's NVRTC
`RawModule`, which keeps its own on-disk kernel cache. MPGOS binaries are
cached under `GPU_ODE_MPGOS/build_cache/<key>/` keyed by problem, solver,
trajectory count, state count and a source hash, so an unchanged point
skips nvcc entirely. torchdiffeq is eager and compiles nothing. DiffEqGPU
kernels are not cached across processes: GPUCompiler's disk cache only
serves code instances with a precompiled build id, which the
ModelingToolkit-generated functions never have; the Julia states sweep
parallelizes compiles across processes instead.

Performance runs compile before they measure: each runner first fills
its package's cache — MPGOS builds every (problem, solver, NT) binary
with up to `BENCH_WARM_JOBS` (default 8) parallel nvcc processes, cubie
compiles each leg once at a tiny ensemble in per-problem child
processes, JAX lowers and compiles each leg at each N, Myokit compiles
each model — then runs the timed sweep against warm caches.
`run_benchmark -a warm` fills every cache the suite can use: timing
solvers, every work-precision setting, and julia's `Pkg.precompile`;
`run_full_dataset -a warm` does that for every package. States-sweep
kernels are never warmed.

Cubie tunes before it warms. `run_benchmark -p cubie -a optimize` runs
`Solver.optimize` on an `optimize.n`-trajectory batch and records the winning
unrolling, buffer placement, block size and residency in
`data/CUBIE/<key>/optimize.csv` (`CUBIE_MLIR` for the MLIR backend); every
later solver of that point is built with those settings. Explicit algorithms
are tuned once per (problem, algorithm, mode) at the timing setting; the
families in `optimize.per_point_families` are tuned at every work-precision
setting as well. The performance and work-precision launchers run the step
first and skip points already recorded, `--keep` preserves the rows,
and the states sweep tunes each size after timing its cold build.

The ring modulator is problem II-3 of the test set: a 15-state circuit model
whose stiffness scales with `1/Cs`. At `Cs = 0` the four capacitor rows
become algebraic and the system is an index-2 DAE, which is a separate row
sweeping the `Uin1` amplitude instead. Cubie derives the mass matrix during
parsing and tears the algebraic states out by structural simplification; the
torn variables are recorded as observables, so the full 15-variable state is
still compared against the golden. Only fully implicit stages integrate it:
cubie rejects the explicit algorithms and `kvaerno3` on a singular mass
matrix, leaving `rosenbrock23_sciml` and `radau_iia_5`.

The Julia systems are defined once as ModelingToolkit models
(`runner_scripts/julia_systems.jl`): `mtkcompile` transforms the raw
equations and every numeric artifact the suites use — right-hand sides,
symbolic jacobians, time gradients, mass matrices, variable orderings — is
generated from the compiled system and handed to DiffEqGPU as plain
callables. The index-2 ring modulator is the same equation set with
`Cs = 0` substituted at definition, which derives its singular mass matrix.

The NAND gate is the test set's index-0 implicit DE `C(y) y' = f(y, t)`
with a state-dependent, non-diagonal capacitance matrix. Cubie takes it in
natural form and ModelingToolkit compiles it to fourteen node potentials
plus eight derivative states behind a constant singular mass matrix, so its
`frameworks` column is `cubie|cubie_mlir|julia`; the remaining frameworks
have no formulation for that left-hand side. The Float64 golden integrates
it as a fully implicit DFBDF `DAEProblem` with tstops on the pulse corners.
Golden references are Float64 solves under each problem's
`golden_algorithm` at its `golden_tol`, checked against the published
test-set values by `runner_scripts/golden/verify_references.jl`.

Every entry point takes `-s <problem>` (default `all`, or a comma list), and a
framework skips cleanly when a requested problem is not in its list:

```bash
    $ bash ./run_benchmark.sh -p cubie -s lorenz
    $ bash ./run_all_benchmarks.sh -s ring_modulator -g kvaerno3
    $ ./run_full_dataset.sh -s lorenz
```

Adding a problem means one CSV row plus its right-hand side in each
framework's system module: `runner_scripts/{cubie,jax,torch,julia}_systems.*`
and `reference_systems.jl` for the Float64 golden, a
`GPU_ODE_MPGOS/problems/<name>.cuh` header, and a CellML model under
`GPU_ODE_MYOKIT_CUDA/models/` for the Myokit suite.

### Generating the complete dataset

`bench.py` runs every stage: cubie optimize and warm, the timing, states and
work-precision sweeps, numerical equivalence, the overlap comparison, plots and
reports. Every axis (package, analysis, algorithm, problem, mode, N) takes a
comma list and `--point` retakes single points; the `run_*.sh`/`.bat` scripts
forward to it. Without `--keep` a run first drops only the store rows it is
about to record.

```bash
    $ python3 bench.py                           # everything, nmax = 2^24
    $ python3 bench.py -n $((2**25))             # larger ceiling
    $ python3 bench.py -n $((2**23)),$((2**27))  # exact trajectory counts only
    $ python3 bench.py -a performance            # one analysis
    $ python3 bench.py -a optimize,warm -p cubie # tune and fill the cubie caches only
    $ python3 bench.py -p cpp                    # one package
    $ python3 bench.py -p cubie,julia -g euler,tsit5   # subsets of both
    $ python3 bench.py --mode adaptive -s pollu  # one mode of one problem
    $ python3 bench.py --resume                  # skip every point already on disk
    $ python3 bench.py --no-overwrite            # keep finite results, retry NaN and absent points
    $ python3 bench.py --resume-from jax         # restart the perf sweep at a package
    $ python3 bench.py --resume \
        --resume-from cubie:ring_modulator_index2:rosenbrock23_sciml:adaptive:262144
                                                 # ...or at an exact (problem, algorithm, mode, N)
    $ python3 bench.py --floor -s lorenz         # re-run and keep the lower time per point
    $ python3 bench.py --point times:cubie:lorenz:tsit5:fixed:32768 \
                       --point wp:julia:pollu:kvaerno3 --point states:cpp:lorenz96:classical-rk4:16
    $ python3 bench.py --points-file retakes.txt # one point per line
```

A point is `<times|wp|states>:<package>:<problem>:<algorithm>[:<mode>][:<N or
state count>]`; a run of points replaces only those rows and redraws the plots,
and a point without a mode re-measures both modes. `JULIA` overrides the
`julia +1.13` launcher.

`--resume` skips every (problem, algorithm, mode, N) point whose row is
already in the result store and deletes nothing; NaN rows count as recorded.
`--no-overwrite` skips only points with a finite recorded time; NaN and
absent rows rerun, and a rerun point replaces its row. `--keep` gives the
no-deletion behaviour on its own.
`--resume-from` places a cursor in the run order (problems.csv order, then
algorithms.csv order, fixed before adaptive, N ascending) and skips
everything before it — use it to step over a point that hangs, since a hung
point leaves no row for `--resume` to skip. `--floor` re-runs the selected
points (it skips nothing, and implies `--keep`) and merges each result into
the store by keeping the row with the lower time, per transfer leg, so a
re-run can only tighten a recorded minimum. Which points re-run comes from the flags that
already select work (`-s`, `-g`, `-n`). All four flags are also accepted by
`run_benchmark.sh` / `run_benchmark.bat`, where `--resume-from` starts at
the problem:

```bash
    $ bash ./run_benchmark.sh -p cubie --resume     # fill only the gaps
    $ bash ./run_benchmark.sh -p cubie --resume \
        --resume-from ring_modulator_index2:rosenbrock23_sciml:adaptive:262144
```

**On Windows** the same flags apply:

```cmd
    > python bench.py -n 16777216 -a performance,work-precision
```

At high trajectory counts some frameworks will exhaust GPU memory. Each
framework runs as its own process tree, so an OOM ends only that framework's
sweep: the smaller-N points already written to disk are kept, the remaining N
values are left absent, and the run moves on to the next framework. Stages are
independent in the same way — a failed stage never aborts the others.
Cubie always runs first, and the stage's plots are redrawn as each package lands.

Every run writes a timestamped `logs/<dataset-key>_<stamp>/` directory holding
a per-step log, a `run_manifest.txt` recording the git revision, GPU and
parameters, and a `summary.tsv`. The run finishes by printing a summary table
marking each step `OK`, `PARTIAL` (with the largest N reached), `FAILED`, or
`SKIPPED`, so a truncated sweep is visible rather than looking like a plain
failure. A non-zero exit is therefore expected when frameworks OOM at high N.

The run refuses to start if `nvidia-smi` cannot identify the GPU, since every
output file is keyed by `<os>_<gpu>` and the whole dataset would otherwise be
mislabelled `unknown-gpu`; override with `--allow-unknown-gpu`.

#### Clock stability

The run pins the SM and memory clocks to the per-GPU rate in
`runner_scripts/gpu_clocks.conf` and releases the lock on any exit, Ctrl-C
included:

```bash
    $ ./run_full_dataset.sh --lock-clocks 1470,6801  # override the target (SM[,MEM])
    $ ./run_full_dataset.sh --no-lock-clocks         # measure but do not pin
    $ ./run_full_dataset.sh --clock-tolerance 30     # widen the drift threshold (MHz)
```

Locking needs passwordless `sudo nvidia-smi` (an Administrator console on
Windows). Without it the run continues unlocked, still logs and reports what
the clocks did, and prints the `nvidia-smi` commands to lock by hand.

Heat or the power cap can override a lock mid-run, so clocks are sampled at
1 Hz into `logs/<dataset-key>_<stamp>/clocks.csv` and each stage is checked
against its own slice of the log, ignoring idle samples. The final
`CLOCK STABILITY` table marks each stage `OK`, `BLIP` (a stray sample off
target), or `DRIFT` (sustained deviation or a throttle reason asserted).
`DRIFT` in a timed stage (timing, work-precision, overlap) fails the run —
lower the target and re-run those stages with `--resume-from`; in the
accuracy-only stages it is only a warning.

#### Calibrating a new machine

```bash
    $ python3 runner_scripts/calibrate/calibrate_clocks.py
```

Runs a 15 minute load (Linux or Windows) and prints the `gpu_clocks.conf` row
to paste in; the 1 Hz log is kept in `data/clocks/`.

### Benchmarking Julia (DiffEqGPU.jl) methods
We will need to install CUDA.jl for benchmarking. It is the only backend
compatible with the ODE solvers in JAX, PyTorch, and MPGOS. To do so,
one can follow the below process in the Julia Terminal:
```julia
    $ julia
    julia> using Pkg
    julia> Pkg.add("CUDA")
```
Let's clone the benchmark suite repository to start benchmarking;
```bash
    $ git clone https://github.com/utkarsh530\
    /GPUODEBenchmarks.git
```
We will instantiate and pre-compile all the packages beforehand to avoid
the wait times during benchmarking. The folder ./GPU_ODE_Julia contains
all the related scripts for the GPU solvers.
```bash
    $ cd ./GPUODEBenchmarks
    $ julia --project=./GPU_ODE_Julia --threads=auto
    julia> using Pkg
    julia> Pkg.instantiate()
    julia> Pkg.precompile()
    julia> exit()
```
It may take a few minutes to complete (\< 10 minutes). After this, we
can generate the timings of ODE solvers written in Julia. There is a
script to benchmark ODE solvers for the different number of trajectories
to demonstrate scalability and performance. The script invocation and
timings can be generated through the following:

**On Linux/macOS:**
```bash
    $ bash ./run_benchmark.sh -p julia
```

**On Windows:**
```cmd
    > run_benchmark.bat -p julia
```

It might take around 20 minutes to finish. The flag `-n N` can be used
to specify the upper bound of the trajectories to benchmark. By default
$N = 2^{24}$, where the simulation runs for $n \in 8 \le n < N$, with
the multiples of $4$.

Rows land in `data/Julia/<os>_<gpu>/results.csv` with `analysis = times`, two
per (problem, algorithm, mode, N): `transfers = both` times h2d + solve + d2h,
`transfers = none` the resident solve alone. See "Result store" above.

### Benchmarking C++ (MPGOS) ODE solvers

Benchmarking MPGOS ODE solvers requires the CUDA C++ compiler to be
installed correctly. The recommended CUDA Toolkit version is \>= 11. The
installation can be checked through:
```bash
    $ nvcc
    If the installation exists, it will return 
    something like this:
    nvcc fatal   : No input files specified; 
    use option --help for more information
```
If `nvcc` is not found, the user must install the CUDA Toolkit. The
NVIDIA's website lists the resource
[`https://developer.nvidia.com/cuda-downloads`](https://developer.nvidia.com/cuda-downloads)
for installation.

The MPGOS scripts are in the `GPU_ODE_MPGOS` folder. The file
`GPU_ODE_MPGOS/Bench.cu` is the main executed code; the problem header,
solver and trajectory count are compile-time `-D` definitions, so each point
is a rebuild. The MPGOS
programs can be run with the same script by changing the arguments as:

**On Linux/macOS:**
```bash
    $ bash ./run_benchmark.sh -p cpp
```

**On Windows:**
```cmd
    > run_benchmark.bat -p cpp
```

Its rows land in `data/CPP/<os>_<gpu>/results.csv`.

**Note for Windows:** The C++ runner script uses PowerShell for file manipulation. Ensure PowerShell is available and that the execution policy allows running scripts.

### Benchmarking JAX (Diffrax) ODE solvers

Benchmarking JAX-based ODE solvers require installing Python 3.9 and
`conda`. First, we will install all the Python packages for
benchmarking:
```bash
    $ conda env create -f environment.yml
    $ conda activate venv_jax
```
It should install the correct version of JAX with CUDA enabled and the
Diffrax library. The GitHub
[`https://github.com/google/jax#installation`](https://github.com/google/jax#installation)
is a guide to follow if the installation fails.

For our purposes, we can benchmark the solvers by:

**On Linux/macOS:**
```bash
    $ bash ./run_benchmark.sh -p jax
```

**On Windows:**
```cmd
    > run_benchmark.bat -p jax
```

#### A note on JIT ordering in JAX

The JIT ordering JAX matters and sometimes can enhance performance if done correctly. We have tested that vmap and JIT ordering does not make a noticeable difference in our case. The results are available at this [Colab notebook](https://colab.research.google.com/drive/1d7G-O5JX31lHbg7jTzzozbo5-Gp7DBEv?usp=sharing).

### Benchmarking PyTorch (torchdiffeq) ODE solvers

Benchmarking PyTorch-based ODE solvers is a similar process compared to
JAX ones.
```bash
    $ python3 GPU_ODE_PyTorch/setup_environment.py
```
`torchdiffeq` does not fully support vectorized maps with ODE solvers.
To circumvent this, we extended the functionality by rewriting some
library parts, so the setup script installs the fork at a pinned commit:
```bash
    pip install git+https://github.com/\
    utkarsh530/torchdiffeq.git@4f4524f719a619c9bd65b722e5f7bf699ff75f62
```
The setup script runs a `torch.vmap` solve through the fork and fails if
it does not work.

Then run the benchmarks by:

**On Linux/macOS:**
```bash
    $ bash ./run_benchmark.sh -p pytorch
```

**On Windows:**
```cmd
    > run_benchmark.bat -p pytorch
```

### Benchmarking CUBIE ODE solvers

CUBIE is benchmarked twice: once on the stock `numba-cuda` compilation
pipeline (`cubie`) and once on the `numba-cuda-mlir` pipeline (`cubie_mlir`).
Both run from a **single shared virtual environment** holding one PyPI install
of `cubie` with both backends present; `runner_scripts/cubie_adapter.py`
sets `CUBIE_CUDA_BACKEND` from the package name before cubie is imported
(`numba-cuda` and `mlir` respectively) and names each package's systems so
both suites, the overlap worker and the NE sweep share one generated-code
cache per backend.
`GPU_ODE_CUBIE_MLIR/venv` is a link to `GPU_ODE_CUBIE/venv`. Set it up with
`setup_all_environments.py` or the individual `setup_environment.py`
scripts (see [SETUP.md](SETUP.md)), then run:

**On Linux/macOS:**
```bash
    $ bash ./run_benchmark.sh -p cubie
    $ bash ./run_benchmark.sh -p cubie_mlir
```

**On Windows:**
```cmd
    > run_benchmark.bat -p cubie
    > run_benchmark.bat -p cubie_mlir
```

Results are written to `data/CUBIE/` and `data/CUBIE_MLIR/` respectively,
so the MLIR and non-MLIR pipelines appear as separate series in the
comparison plots.

### Benchmarking Myokit-CUDA ODE solvers

The Myokit-CUDA benchmark imports the Lorenz CellML model, exports Myokit's
CUDA device code, and launches the generated equations as a GPU ensemble.
Myokit's CUDA exporter supports float32 forward Euler only, so this
benchmark contributes fixed-step timing and work-precision curves only. It
does not produce an adaptive series.

Set up the environment with `setup_all_environments.py` or
`GPU_ODE_MYOKIT_CUDA/setup_environment.py`, then run either accepted
language spelling:

**On Linux/macOS:**
```bash
    $ bash ./run_benchmark.sh -p myokit_cuda
    $ bash ./run_benchmark.sh -p myokit-cuda
```

**On Windows:**
```cmd
    > run_benchmark.bat -p myokit_cuda
    > run_benchmark.bat -p myokit-cuda
```

Results are written to `data/MYOKIT_CUDA/` with the `Myokit_cuda` filename
prefix.

## Analyses

Errors are computed offline from finals. The golden of a problem is the
julia_cpu float64 finals row of the same problem, construction parameters and
duration under any key. Both scripts run under `GPU_ODE_CUBIE/venv` and select
rows by set or by a SQL predicate over the spec columns; every row of the
selected steppings and packages under every key comes along, whatever its grid:

```bash
python analyses/timing.py --x n      --set perf         # min_ms against n
python analyses/timing.py --x states --set states       # min_ms and cold build against states
python analyses/timing.py --x error  --set golden_grid  # min_ms against error
python analyses/agreement.py --set golden_grid          # errors and package-pair differences
python analyses/timing.py --x n --where "problem = 'lorenz' AND algorithm = 'tsit5'"
```

`timing.py`: one figure and CSV per (key, problem, transfers, stepping), one
series per package. `agreement.py`: per (key, problem), `agreement.csv`,
`agreement_pairs.csv` and one figure per leg. A comparison pairs trajectories
by exact float32 parameter value and takes the RMS over every state of the
difference over the trajectories neither side flags as errored. Rows with
`errored_pct` above 10 are dropped, untimed rows too on the timing axes.
Output: `plots/<key>/<problem>/`.
