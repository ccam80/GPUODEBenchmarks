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
### Testing DiffEqGPU.jl

DiffEqGPU.jl is a test suite that regularly checks functionality by
testing features like multiple backend support, event handling, and
automatic differentiation. To test the functionality, one can follow the
below instructions. The user needs to specify the \"backend\" for
example \"CUDA\" for NVIDIA, \"AMDGPU\" for AMD, \"oneAPI\" for Intel
, and \"Metal\" for Apple GPUs. The estimated time of completion is 20
minutes.
```julia
    $ julia --project=.
    julia> using Pkg
    julia> Pkg.instantiate()
    julia> Pkg.precompile()
```
Finally, test the package with this command
```bash
    $ backend="CUDA"
    $ julia --project=. test_DiffEqGPU.jl $backend
```
Additionally, the GitHub discussion
[`https://github.com/SciML/DiffEqGPU.jl/issues/224#issuecomment-1453769679`](https://github.com/SciML/DiffEqGPU.jl/issues/224#issuecomment-1453769679)
highlights the use of textured memory with ODE solvers, accelerates the
code by $2\times$ over CPU.

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
  package and their plot (see
  [Work-Precision Benchmarks](#work-precision-error-vs-runtime-benchmarks)).
* `-a numerical` — the numerical-equivalence suite (cubie vs.
  DifferentialEquations.jl, error vs. dt and vs. tolerance per algorithm; see
  [Numerical Equivalence](#numerical-equivalence-error-vs-dt--cubie-vs-differentialequationsjl)).
  This delegates to `run_numerical_equivalence.sh`/`.bat`, which can also be
  run standalone.
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

`runner_scripts/algorithms.csv` is the algorithm registry: one row per
integration algorithm, naming the frameworks that run it fixed-step and the
frameworks that run it adaptively, in the cubie vocabulary. Both
`algorithms.py` and `algorithms.jl` read that file, and every bench script
takes its supported set from it. Each figure contains only packages running
the same method:

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
take `atol = rtol` from `TIMING_TOL` in `runner_scripts/wp_common.py` for the
N-sweep and from the `TOLS` grid for work-precision, and start from the
problem's timing dt. Nothing else is set: every package runs its shipped
step-controller defaults.

Controllers are matched in one place only, the cubie against
DifferentialEquations.jl overlap suite, which repeats each comparison with
cubie's controller set to Julia's (`pi_controller` in
`runner_scripts/cubie_julia_overlap/common.py`).

`eps(Float32)` is 1.2e-7, so the tightest points of the tolerance grid and
the 1e-8 `TIMING_TOL` ask for more than the working precision resolves.
Cubie warns `newton_rtol is at or above the step controller rtol` from 1e-7
down. A fixed step leaves diffrax's implicit solvers nothing to take their
root-finder tolerances from, so the bench passes the run's tolerance the way
an adaptive controller would; its chord iteration still diverges on lorenz,
and that point records NaN.

All benchmark entry points accept `-g <algorithms>` (default `all`, meaning
every algorithm the framework supports; a comma list runs the listed ones);
a framework that does not support a requested algorithm skips cleanly:

```bash
    $ bash ./run_benchmark.sh -p cubie -g tsit5
    $ bash ./run_benchmark.sh -p cubie -g euler,tsit5
    $ bash ./run_all_benchmarks.sh -g classical-rk4
    $ ./run_full_dataset.sh --algorithm euler
```

Timing files are named
`data/<package>/<os>_<gpu>/<problem>/<Prefix>_times_<fixed|adaptive>_<algorithm>.txt`
(work-precision files use `_wp_` in place of `_times_`). Data without the
algorithm field is regenerated fresh rather than migrated.

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
| `nand_gate` | 14 | 80 | `VDD` over [4, 6], linear | implicit DE |

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
can no longer reach as NaN and exits, and the Julia runner launches one
process per problem and algorithm so an exit abandons only that pair.

Every problem attempts every algorithm its frameworks support; a failed solve is a NaN row. `lorenz96_20` is the 20-state lorenz96 row, the smaller stiff head-to-head.

### States sweep

`run_benchmark -a states` times lorenz96 at 4-128 states
(`BENCH_STATES_GRID=<comma list>` overrides) and a fixed
131072-trajectory ensemble, in every framework and algorithm
the problem's frameworks support, exclusions included. Rows are
`states t_ms t_dev_ms build_s` in
`<Prefix>_states_<fixed|adaptive>_<algorithm>.txt` under the lorenz96
data directory. `build_s` is the wall time from solver construction to
the first completed solve; the sweep bypasses every compiled-kernel
cache, making it a cold compile on every run. A size with no finite time in
either mode cancels the pending and running larger sizes of that
algorithm; cancelled rows are NaN. `BENCH_STATES_BUDGET` (seconds, unset
disables) kills any process whose first kernel has not compiled within
the budget.

Every Julia analysis runs through `runner_scripts/gpu/julia_driver.py`:
one process per leg — (problem, algorithm) for performance and
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

`run_full_dataset.sh` drives every suite in one set-and-forget run — the
timing sweeps, the work-precision sweeps, the numerical-equivalence suite, the
per-algorithm cubie vs. DiffEqGPU overlap comparison, and finally the plots
and comparison reports:

```bash
    $ ./run_full_dataset.sh                     # everything, nmax = 2^24
    $ ./run_full_dataset.sh -n $((2**25))       # larger ceiling
    $ ./run_full_dataset.sh -n $((2**23)),$((2**27))  # exact trajectory counts only
    $ ./run_full_dataset.sh -a performance      # one analysis
    $ ./run_full_dataset.sh -p cpp              # one package
    $ ./run_full_dataset.sh -p cubie,julia -g euler,tsit5   # subsets of both
    $ ./run_full_dataset.sh --resume-from jax   # restart a part-finished sweep
```

**On Windows** the same flags apply through `run_full_dataset.bat`, a wrapper
for `run_full_dataset.ps1`:

```cmd
    > run_full_dataset.bat -n 16777216 -a performance,work-precision
```

At high trajectory counts some frameworks will exhaust GPU memory. Each
framework runs as its own process tree, so an OOM ends only that framework's
sweep: the smaller-N points already written to disk are kept, the remaining N
values are left absent, and the run moves on to the next framework. Stages are
independent in the same way — a failed stage never aborts the others.

Every run writes a timestamped `logs/<dataset-key>_<stamp>/` directory holding
a per-step log, a `run_manifest.txt` recording the git revision, GPU and
parameters, and a `summary.tsv`. The run finishes by printing a summary table
marking each step `OK`, `PARTIAL` (with the largest N reached), `FAILED`, or
`SKIPPED`, so a truncated sweep is visible rather than looking like a plain
failure. A non-zero exit is therefore expected when frameworks OOM at high N.

The run refuses to start if `nvidia-smi` cannot identify the GPU, since every
output file is keyed by `<os>_<gpu>` and the whole dataset would otherwise be
mislabelled `unknown-gpu`; override with `--allow-unknown-gpu`.

Overlap CSVs stay in `data/cubie_julia_overlap/<os>_<gpu>/`; its figures are `plots/<os>_<gpu>/overlap_*.png` and its report `plots/<os>_<gpu>/cubie_julia_overlap.md`, both redrawn by `-a plots`.

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
    $ bash ./run_benchmark.sh -p julia -d gpu -m ode
```

**On Windows:**
```cmd
    > run_benchmark.bat -p julia -d gpu -m ode
```

It might take around 20 minutes to finish. The flag `-n N` can be used
to specify the upper bound of the trajectories to benchmark. By default
$N = 2^{24}$, where the simulation runs for $n \in 8 \le n < N$, with
the multiples of $4$.

The data will be generated in the `data/Julia` directory, with two files
for fixed and adaptive time-stepping simulations. Each \".txt\" row is
`N time_ms time_device_only_ms`: the number of trajectories, the
end-to-end time (h2d + solve + d2h) in milliseconds, and the same solve
with the inputs already resident and the results left on the device.
Every framework's timing files share this format.

Additionally, to benchmark ODE solvers for other backends:

**On Linux/macOS:**
```bash
    $ N=$((2**24))
    $ backend="Metal"
    $ ./runner_scripts/gpu/run_ode_mult_device.sh $N $backend
```

**On Windows:**
```cmd
    > set N=16777216
    > set backend=Metal
    > runner_scripts\gpu\run_ode_mult_device.bat %N% %backend%
```
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
    $ bash ./run_benchmark.sh -p cpp -d gpu -m ode
```

**On Windows:**
```cmd
    > run_benchmark.bat -p cpp -d gpu -m ode
```

It will generate the data files in the `data/cpp` folder.

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
    $ bash ./run_benchmark.sh -p jax -d gpu -m ode
```

**On Windows:**
```cmd
    > run_benchmark.bat -p jax -d gpu -m ode
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
    $ bash ./run_benchmark.sh -p pytorch -d gpu -m ode
```

**On Windows:**
```cmd
    > run_benchmark.bat -p pytorch -d gpu -m ode
```

### Benchmarking CUBIE ODE solvers

CUBIE is benchmarked twice: once on the stock `numba-cuda` compilation
pipeline (`cubie`) and once on the `numba-cuda-mlir` pipeline (`cubie_mlir`).
Both run from a **single shared virtual environment** holding one PyPI install
of `cubie` with both backends present; the active backend is chosen at import
time by the `CUBIE_CUDA_BACKEND` environment variable, which each launcher
exports for you (`numba-cuda` and `mlir` respectively).
`GPU_ODE_CUBIE_MLIR/venv` is a link to `GPU_ODE_CUBIE/venv`. Set it up with
`setup_all_environments.py` or the individual `setup_environment.py`
scripts (see [SETUP.md](SETUP.md)), then run:

**On Linux/macOS:**
```bash
    $ bash ./run_benchmark.sh -p cubie -d gpu -m ode
    $ bash ./run_benchmark.sh -p cubie_mlir -d gpu -m ode
```

**On Windows:**
```cmd
    > run_benchmark.bat -p cubie -d gpu -m ode
    > run_benchmark.bat -p cubie_mlir -d gpu -m ode
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
    $ bash ./run_benchmark.sh -p myokit_cuda -d gpu -m ode
    $ bash ./run_benchmark.sh -p myokit-cuda -d gpu -m ode
```

**On Windows:**
```cmd
    > run_benchmark.bat -p myokit_cuda -d gpu -m ode
    > run_benchmark.bat -p myokit-cuda -d gpu -m ode
```

Results are written to `data/MYOKIT_CUDA/` with the `Myokit_cuda` filename
prefix.

For the Fabbri-Linder CellML comparison against CuBIE, run the dedicated
two-environment harness with the Myokit-CUDA Python executable and point it
at an installed CuBIE environment:

```text
GPU_ODE_MYOKIT_CUDA/venv/Scripts/python.exe runner_scripts/cubie_myokit_fabbri/compare_fabbri.py --cellml <cubie>/tests/fixtures/cellml/Fabbri_Linder.cellml --cubie-python <cubie>/.venv/Scripts/python.exe --trajectory-counts 512 2048 8192 32768 131072
```

It compares synchronized float32 forward-Euler solves, checks all 35 final
states, and writes per-count details plus an aggregate scaling table under
`data/Fabbri_Myokit_CUDA/`. See
[`runner_scripts/cubie_myokit_fabbri/README.md`](runner_scripts/cubie_myokit_fabbri/README.md)
for the CellML compatibility normalization and timing protocol.

## Comparing GPU acceleration of ODEs with CPUs

The benchmark suite can also be used to test the GPU acceleration of ODE
solvers in comparison with CPUs. The process for generating simulation
times for GPUs can be done by following the GPU section mentioned earlier. The following script
allows the generation of CPU simulation times for ODEs:

**On Linux/macOS:**
```bash
    $ bash ./run_benchmark.sh -p julia -d cpu -m ode
```

**On Windows:**
```cmd
    > run_benchmark.bat -p julia -d cpu -m ode
```

The simulation times will be generated in `data/CPU`. Each of the
workflow takes approximately 20 minutes to finish.

## Benchmarking GPU acceleration of SDEs with CPUs

The SDE solvers in Julia are benchmarked by comparing them to the
CPU-accelerated simulation. This will benchmark the linear SDE with
three states, as described in the \"Benchmarks and case studies\"
section. To generate simulation times for GPU, do the following:

**On Linux/macOS:**
```bash
    $ bash ./run_benchmark.sh -p julia -d gpu -m sde
```

**On Windows:**
```cmd
    > run_benchmark.bat -p julia -d gpu -m sde
```

We can generate the simulation times for CPU-accelerated codes through the following:

**On Linux/macOS:**
```bash
    $ bash ./run_benchmark.sh -p julia -d cpu -m sde
```

**On Windows:**
```cmd
    > run_benchmark.bat -p julia -d cpu -m sde
```

The results will get generated in `data/SDE` and `data/CPU/SDE`, taking
around 10 minutes to complete.

## Composability with MPI

Julia supports Message Passing Interface (MPI) to allow Single Program
Multiple Data (SPMD) type parallel programming. The composability of the
GPU ODE solvers enable seamless integration with MPI, enabling scaling
the ODE solvers to clusters on multiple nodes.
```julia
    $ julia --project=./GPU_ODE_Julia
    julia> using Pkg
    # install MPI.jl
    julia> Pkg.add("MPI")
```
An example script solving the Lorenz problem for approximately 1 billion
parameters are available in the `MPI` folder. A SLURM-based script is
shown below.
```bash
    #!/bin/bash
    # Slurm Sbatch Options
    # Reqeust no. of GPUs/node
    #SBATCH --gres=gpu:volta:1
    # 1 process per node 
    #SBATCH -n 5 -N 5
    #SBATCH --output="./mpi_scatter_test.log-%j"
    # Loading the required module

    # MPI.jl requires a memory pool to be disabled
    export JULIA_CUDA_MEMORY_POOL=none
    export JULIA_MPI_BINARY=system
    # Use local CUDA toolkit installation
    export JULIA_CUDA_USE_BINARYBUILDER=false

    source $HOME/.bashrc
    module load cuda mpi

    srun hostname > hostfile
    time mpiexec julia --project=./GPU_ODE_Julia\ 
    ./MPI/gpu_ode_mpi.jl
```
## Plotting Results

The plotting scripts to visualize the simulation times. The scripts are
located in the `runner_scripts/plot` folder. These scripts replicate the
benchmark figures in the paper. The benchmark suite contains the
simulation data generated by authors, which can be used to verify the
plots. Various benchmarks can be plotted, which are described in the
different sections. The plotting scripts are based on Julia. As a
preliminary step:
```julia
    $ cd GPUODEBenchmarks
    $ julia project=.
    julia> using Pkg
    julia> Pkg.instantiate()
    julia> Pkg.precompile()
```
The plot comparison between Julia, C++, JAX, and PyTorch mentioned in
the paper can be generated by using the below command:
```bash
    $ julia --project=. ./runner_scripts/plot\
    /plot_ode_comp.jl
```
The plot will get saved in the `plots` folder.

Similarly, the other plots in the paper can be generated by running the
different scripts in the folder `runner_scripts/plot`.
```bash
    plot performance of GPU ODE solvers 
    with multiple backends
    $ julia --project=. ./runner_scripts/plot\
    /plot_mult_gpu.jl 
    plot GPU ODE solvers comparsion with CPUs
    $ julia --project=. ./runner_scripts/plot\
    /plot_ode_comp.jl 
    plot GPU SDE solvers comparsion with CPUs
    $ julia --project=. ./runner_scripts/plot\
    /plot_sde_comp.jl 
    plot CRN Network sim comparison with CPUs
    $ julia --project=. ./runner_scripts/plot\
    /plot_sde_crn.jl 
```
To plot data generated by running the scripts, specify the location of
the `data` as the argument to the mentioned command.
```bash
    $ julia --project=. ./runner_scripts/plot/\
    plot_mult_gpu.jl /path/to/data/
```

## Comparing Numerical Results

The benchmark suite includes a feature to compare the numerical accuracy of
results between different integration packages. When running benchmarks with
exactly 32768 trajectories, each package automatically saves its final state
arrays to CSV files in the `data/numerical/` directory.

### Running Benchmarks for Numerical Comparison

To generate the numerical comparison data, run each benchmark with 32768 
trajectories:

```bash
# CUBIE
source ./GPU_ODE_CUBIE/venv/bin/activate
python3 ./GPU_ODE_CUBIE/bench_cubie.py 32768
deactivate

# CUBIE-MLIR (cubie on the numba-cuda-mlir backend)
source ./GPU_ODE_CUBIE_MLIR/venv/bin/activate
python3 ./GPU_ODE_CUBIE_MLIR/bench_cubie_mlir.py 32768
deactivate

# Myokit-CUDA (float32 forward Euler)
source ./GPU_ODE_MYOKIT_CUDA/venv/bin/activate
python3 ./GPU_ODE_MYOKIT_CUDA/bench_myokit_cuda.py 32768
deactivate

# JAX
source ./GPU_ODE_JAX/venv/bin/activate
python3 ./GPU_ODE_JAX/bench_diffrax.py 32768
deactivate

# PyTorch
source ./GPU_ODE_PyTorch/venv/bin/activate
python3 ./GPU_ODE_PyTorch/bench_torchdiffeq.py 32768
deactivate

# Julia
julia --project=. ./GPU_ODE_Julia/bench_ode_gpu.jl 32768
```

### Analyzing Numerical Differences

After generating the CSV files, use the comparison script to analyze differences:

```bash
source ./GPU_ODE_CUBIE/venv/bin/activate
python3 compare_numerical_results.py
deactivate
```

The script performs pairwise comparisons using `numpy.allclose()` and provides:
- Maximum, mean, minimum, and standard deviation of absolute differences
- Maximum, mean, minimum, and standard deviation of relative differences  
- Per-state statistics (for the Lorenz system: x, y, z)
- List of worst mismatches
- Summary of which packages pass the allclose test

For more details, see `data/numerical/README.md`.

## Work-Precision (error vs. runtime) Benchmarks

The trajectory-scaling benchmarks above measure *time only*; the
work-precision (`wp`) mode additionally measures *solution error* against a
golden reference, giving DiffEqDevTools-style error-vs-runtime curves for
every framework at a fixed ensemble size of N = 131072.

### Golden reference

The reference is a Float64 CPU solve (Vern9, abstol = reltol = 1e-13) of the
same Lorenz ensemble, over the float64 rho linspace rounded through float32
(the grid the float32 frameworks actually integrate). Generate it once per
checkout:

```bash
julia -t auto --project=. runner_scripts/golden/generate_golden.jl
# -> data/numerical/golden_<problem>_131072.csv (machine independent, no dataset key)
# An existing file is kept; --force regenerates it, --problem selects one.
```

Because the frameworks build their rho grids independently (and differ by
~1 ulp of float32), and integrate in float32, the meaningful error floor of
the curves is roughly 1e-6 — inherent to benchmarking float32 solvers, not an
artifact.

### Running the sweeps

Each framework's `wp` mode sweeps the controls it supports, once per
supported algorithm (narrow with `-g <algorithm>`): fixed-step sweeps use
dyadic dt from 1/16 to 1/8192 (1/256 to 1/131072 for forward Euler), while
adaptive sweeps use rtol = atol from 1e-2 to 1e-8. Each setting uses the
usual timing protocol
(untimed warm-up, repeated solves, best time) and computes the ensemble l2
error of the final states against the golden reference. Protocol constants
live in `runner_scripts/wp_common.py` (mirrored in the Julia and MPGOS
writers).

```bash
./run_benchmark.sh -p cubie      -d gpu -m ode -a work-precision
./run_benchmark.sh -p cubie-mlir -d gpu -m ode -a work-precision
./run_benchmark.sh -p myokit-cuda -d gpu -m ode -a work-precision  # float32 forward Euler only
./run_benchmark.sh -p julia      -d gpu -m ode -a work-precision
./run_benchmark.sh -p pytorch    -d gpu -m ode -a work-precision   # fixed-dt only: torch.vmap cannot trace adaptive solvers
./run_benchmark.sh -p jax        -d gpu -m ode -a work-precision   # Linux/WSL2 only (no CUDA jaxlib on native Windows)
./run_benchmark.sh -p cpp        -d gpu -m ode -a work-precision   # MPGOS: rebuilds RK4 + RKCK45 once each at NT=131072
```

(`run_benchmark.bat -p <package> -d gpu -m ode -a work-precision` on Windows.) To run every
package's work-precision sweeps and the plot in one go:
`./run_all_benchmarks.sh -a work-precision` (`run_all_benchmarks.bat -a work-precision`).

Results are written per machine as
`data/<package>/<os>_<gpu>/<problem>/<Prefix>_wp_<fixed|adaptive>_<algorithm>.txt`
with rows `<setting> <time_ms> <error>`. Notes:

* The wp timings synchronize the device before stopping the clock (JAX
  `block_until_ready`, torch `cuda.synchronize`), unlike the historical
  N-sweep timings for those frameworks.
* torchdiffeq has no adaptive sweep (data-dependent control flow cannot be
  `torch.vmap`ed — the same reason its N-sweep is fixed-step only).
* Myokit-CUDA has no adaptive sweep because Myokit's CUDA exporter provides
  float32 forward Euler only.

### Plotting

```bash
julia --project=. ./runner_scripts/plot/plot_ode_wp.jl
```

discovers all keyed wp files and writes one algorithm-matched figure per
(mode, algorithm), `plots/<group>/<problem>/wp_<mode>_<algorithm>.png`, plus a
`plots/<group>/<problem>/wp_all.png` overview, for the same groups as
`plot_ode_comp.jl`.

## Numerical Equivalence (error vs. dt) — cubie vs. DifferentialEquations.jl

The work-precision curves compare error against *runtime*; the
numerical-equivalence (`ne`) suite instead compares error against *dt*, per
algorithm, to answer a different question: **does each cubie algorithm
actually calculate what its named method should?** Every implicit-family
algorithm mutually supported by cubie and DifferentialEquations.jl (the
mapping lives in `runner_scripts/numerical_equivalence/algorithms.csv`)
integrates the same Lorenz ensemble (N = 1024, rho in [0, 21], t in [0, 1])
fixed-step at every dyadic dt from 1/2 to 1/8192 — **both stacks in
Float32** — and the final states are compared against the Float64 golden
reference and against each other. erk-family algorithms run only the
adaptive sweep. The small-dt end of the grid resolves the fp-precision
tail.

Float32 discipline on the Julia side is enforced, not assumed: u0, tspan, dt
and the parameter vector are constructed as Float32 (the rho grid is read
from the golden file, whose values are exactly representable in Float32) and
every trajectory's final state is asserted to still be Float32, so a silent
promotion to Float64 aborts the run.

### Running the suite

One command runs everything (golden reference if missing, both reference
sweeps, both cubie sweeps, comparison report + plots):

```bash
./run_numerical_equivalence.sh              # Linux/macOS/WSL
run_numerical_equivalence.bat               # Windows
```

Both take `--controller fixed|adaptive|all` (default `all`) to run just one
of the two sweep types, `-p julia|cubie|all` to run one side of the
comparison, and exit non-zero when any step fails.
`run_all_benchmarks.sh -a numerical` appends the same suite to a full
benchmark run.

Reproducibility: the golden reference and the DifferentialEquations.jl
outputs under `data/numerical_equivalence/julia/<problem>/` are machine-independent
CPU results — once committed, a fresh machine (or cubie's CI) can skip
Julia entirely and only re-run the cubie sweeps + comparison against the
committed reference. The golden is regenerated only if its file is missing
(~1 s of solve time); the Julia sweeps take ~1–3 minutes; the cubie sweeps
dominate the wall time (one kernel compile per algorithm/setting point,
~20–25 minutes per mode on an RTX 4070 SUPER). On a fresh checkout run
`python setup_julia.py` once first (Project.toml is intentionally not
committed; the setup script builds the Julia environment, including the
OrdinaryDiffEq solver sub-libraries this suite needs).

The suite's four steps, each also runnable by hand (the two sweep runners
take the same optional `fixed|adaptive|all` mode argument):

```bash
julia -t auto --project=. runner_scripts/numerical_equivalence/generate_golden_ne.jl
#   -> data/numerical/golden_ne_<problem>_1024.csv  (Float64, machine independent)
julia -t auto --project=. runner_scripts/numerical_equivalence/ne_diffeq.jl
#   -> data/numerical_equivalence/julia/<problem>/<algorithm>.csv            (fixed sweep)
#   -> data/numerical_equivalence/julia/<problem>/<algorithm>_adaptive.csv   (adaptive sweep)
#   -> data/numerical_equivalence/julia/<problem>/controller_constants.csv   (resolved defaults)
GPU_ODE_CUBIE/venv/*/python GPU_ODE_CUBIE/numerical_equivalence.py
#   -> data/numerical_equivalence/cubie/<os>_<gpu>/<problem>/<algorithm>.csv
#   -> data/numerical_equivalence/cubie/<os>_<gpu>/<problem>/<algorithm>_adaptive_<tier>.csv
GPU_ODE_CUBIE/venv/*/python compare_numerical_equivalence.py
#   -> plots/<os>_<gpu>/numerical_equivalence_fixed.csv
#   -> plots/<os>_<gpu>/numerical_equivalence_adaptive.csv
#   -> plots/<os>_<gpu>/numerical_equivalence.png (+ _adaptive variant)
```

### Adaptive sweeps

The fixed-step sweep deliberately removes the step-size controller to
isolate each tableau; the adaptive sweep tests the opposite composite —
embedded estimator + error norm + controller — under real controller
dynamics. Every algorithm with an embedded error estimate on *both* sides
(the `adaptive` column of `algorithms.csv`, cross-checked at runtime
against cubie's `tableau.has_error_estimate` and OrdinaryDiffEq's
`isadaptive`) integrates the ensemble at atol = rtol over 1e-2 .. 1e-8, in
Float32, with pinned initial dt and dt bounds, and errors are compared
against the golden reference as error-vs-tolerance curves. Both runners
skip algorithms outside that mutual set.

Cubie runs each algorithm twice:

* **default** — cubie's shipped controller defaults, compared to the
  Julia run's own per-algorithm defaults as curve tracking within a
  factor.
* **matched** — controller type, gains, safety factor and gain clamps
  mirrored from the constants DifferentialEquations.jl resolved for that
  algorithm (exported to `controller_constants.csv`; the gain mapping
  `kp = beta1*(order+1)`, `ki = -beta2*(order+1)` accounts for the two
  stacks' different exponent conventions — derivation in
  `GPU_ODE_CUBIE/numerical_equivalence.py`). This tier exists to isolate
  how much of the difference between the two stacks comes from the step
  controller rather than the algorithm. When the matched constants equal
  cubie's own defaults, the matched file is written from the default
  tier's results.

Both sweeps write per-algorithm CSVs holding, per dt or per tolerance, the
ensemble l2 error of each implementation against the golden reference and
their ratio `err_cubie / err_julia`, alongside the per-side non-converged
trajectory counts. Errors use only the trajectories both stacks solved.
Each package runs its own implementation and its own defaults; nothing is
pinned on one stack to make it resemble the other.

The golden file and the DifferentialEquations.jl outputs are machine
independent and cheap to regenerate (seconds and ~1 minute respectively), so
the suite is cheap to re-run against a fixed reference: commit (or
regenerate) the golden + Julia reference CSVs, then run only the cubie
sweep and the comparison.
