# Environment setup

Every package runs in its own environment; `bench.py` picks the interpreter
per package from `runner_scripts/launch.py`. One command builds them all:

```
python setup_all_environments.py
```

It runs, in order, the cubie, cubie-MLIR, JAX, PyTorch and Myokit-CUDA
setup scripts under `GPU_ODE_*/` and then `setup_julia.py`, and reports which
failed. Each script can be run on its own and reuses an existing venv.

## Prerequisites

- An NVIDIA GPU with a driver that reports CUDA 13 in `nvidia-smi`. The JAX,
  PyTorch and Myokit setups read the CUDA major from `nvcc` or `nvidia-smi`
  and refuse any other; the cubie setup targets CUDA 13 wheels unless
  `CUBIE_CUDA_MAJOR=12` is set.
- Python 3.11 or newer for the suite (`tomllib`); the PyTorch setup looks for
  the newest CPython that torch publishes wheels for (3.14 at the pin) and
  recreates its venv with it.
- The CUDA toolkit (`nvcc`) for MPGOS; on Windows also Visual Studio with the
  C++ workload (`run_ode_cpp.ps1` enters it through `vswhere`).
- Julia through `juliaup`: `setup_julia.py` pins this directory to the 1.13
  channel and every launcher runs `julia +1.13` (override with `JULIA`).
- Locking GPU clocks needs an elevated shell (passwordless `sudo nvidia-smi`
  on Linux, an Administrator console on Windows); without it a run measures
  unlocked and reports the clocks it saw.
- `rclone` with the `box:` remote (`sync/README.md`), or `--no-sync` on every run and analysis.

On a fresh Linux machine:

```
sudo apt install build-essential nvidia-cuda-toolkit python3-venv python3-pip git
curl -fsSL https://install.julialang.org | sh -s -- --yes
```

## Packages

### cubie and cubie_mlir

```
python GPU_ODE_CUBIE/setup_environment.py
python GPU_ODE_CUBIE_MLIR/setup_environment.py
```

The first builds `GPU_ODE_CUBIE/venv` with `uv`: `cubie` from PyPI with both
backends (`cuda13` and `mlir-cuda13` extras) and its test extra, plus
`pyarrow`, `duckdb` and `matplotlib`. That venv is also the suite
interpreter: `bench.py`, the store CLI, the Julia and C++ runners' store
writes and the analyses all run under it. The second script only links
`GPU_ODE_CUBIE_MLIR/venv` to it (a symlink, or a junction on Windows) and
checks the MLIR backend imports. `runner_scripts/cubie_adapter.py` sets
`CUBIE_CUDA_BACKEND` from the package name (`numba-cuda` or `mlir`) before
cubie is imported.

### jax

```
python GPU_ODE_JAX/setup_environment.py
```

Linux only; elsewhere the script prints a skip and exits 0.
Installs the pinned
`jax[cuda13]`, Diffrax and Equinox with `pyarrow` and `tzdata`, and fails
when jax-cuda plugins of two CUDA generations coexist in the venv.

### pytorch

```
python GPU_ODE_PyTorch/setup_environment.py
```

Installs the pinned torch from the `cu132` wheel index, the vmap-capable
torchdiffeq fork at its pinned commit, `pyarrow` and `tzdata`, then runs a
`torch.vmap` solve through the fork on CUDA and fails if it does not work.

### myokit_cuda

```
python GPU_ODE_MYOKIT_CUDA/setup_environment.py
```

Installs `GPU_ODE_MYOKIT_CUDA/requirements.txt` (pinned Myokit), the pinned
`cupy-cuda13x`, `pyarrow` and `tzdata`, and checks that CuPy sees a device.
See `GPU_ODE_MYOKIT_CUDA/README.md` for what the kernel does.

### cpp

MPGOS needs no environment: `runner_scripts/gpu/run_ode_cpp.ps1` and `.sh`
compile `GPU_ODE_MPGOS/Bench.cu` with `nvcc` per (problem, solver,
trajectory count, precision) into `GPU_ODE_MPGOS/build_cache/<key>/` and
record through the suite interpreter. `runner_scripts/mpgos_trials.py`
refreshes `GPU_ODE_MPGOS/protocol.h` from `protocol.toml` before a build.

### julia_gpu and julia_cpu

```
python setup_julia.py            # instantiate the committed Manifest.toml
python setup_julia.py --update   # re-resolve and rewrite Project/Manifest
```

`Project.toml` and `Manifest.toml` at the repo root are the one Julia
project; `GPU_ODE_JuliaKernels` is a path dependency whose precompile
workload warms the DiffEqGPU kernels (`GPU_ODE_JuliaKernels/README.md`).
`runner_scripts/gpu/julia_driver.py` runs `Pkg.instantiate()` and
`Pkg.precompile()` once before its legs. `bench_ode_cpu.jl` and
`bench_ode_gpu.jl` record through `runner_scripts/results.jl`, which calls
the store CLI under the suite interpreter, or under the `python` on PATH
when the cubie venv is absent; that interpreter then needs `pyarrow` and
`duckdb`.

## Pinned versions

Change these in the setup scripts.

| package | pin |
| --- | --- |
| torch | 2.13.0 (`cu132` index), torchdiffeq `utkarsh530/torchdiffeq@4f4524f` |
| jax / jaxlib | 0.11.1 (`cuda13` extra), diffrax 0.7.2, equinox 0.13.8 |
| myokit | 1.39.2 (`GPU_ODE_MYOKIT_CUDA/requirements.txt`), cupy-cuda13x 14.2.0 |
| cubie | newest PyPI release at setup time; `package_version` records it per row |
| Julia | `Manifest.toml` (julia 1.13); DiffEqGPU and OrdinaryDiffEq versions are recorded per row |

## Verification

```
GPU_ODE_CUBIE/venv/bin/python -c "import cubie, pyarrow, duckdb, matplotlib"
GPU_ODE_JAX/venv/bin/python -c "import jax, diffrax; print(jax.default_backend())"
GPU_ODE_PyTorch/venv/bin/python -c "import torch, torchdiffeq; print(torch.cuda.is_available())"
GPU_ODE_MYOKIT_CUDA/venv/bin/python -c "import cupy, myokit"
julia +1.13 --project=. -e "using DiffEqGPU, CUDA, GPU_ODE_JuliaKernels"
nvidia-smi && nvcc --version
python bench.py plan --set perf
```

On Windows the interpreters are `venv\Scripts\python.exe`. `plan` does no
GPU work but refuses to run when `nvidia-smi` cannot name the GPU.

## Troubleshooting

To rebuild a Python environment, remove its venv and re-run the setup
script. `GPU_ODE_CUBIE_MLIR/venv` is a link to `GPU_ODE_CUBIE/venv`: remove
the link itself, not its contents.

```
rm GPU_ODE_CUBIE_MLIR/venv && rm -rf GPU_ODE_CUBIE/venv         # Linux, macOS
cmd /c rmdir GPU_ODE_CUBIE_MLIR\venv                             # Windows: drops the junction only
Remove-Item -Recurse -Force GPU_ODE_CUBIE\venv                   # then the venv itself
```

If the Julia project fails to instantiate, `setup_julia.py --update`
re-resolves it under the 1.13 channel.

Caches safe to delete: `generated/` (cubie code, `jax_cache`),
`GPU_ODE_MYOKIT_CUDA/models/generated/` and `GPU_ODE_MPGOS/build_cache/`.
