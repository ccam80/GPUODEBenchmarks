# Myokit-CUDA benchmark

This environment imports the Lorenz CellML model with Myokit 1.39.2, exports
Myokit's single-cell `cuda-kernel` forward-Euler function, and compiles it
with an appended ensemble launcher through CuPy/NVRTC. The generated Myokit
equations are not rewritten. The Lorenz `rho` sweep is supplied through the
exporter's required `diffusion_current` binding.

Create the environment:

```text
python GPU_ODE_MYOKIT_CUDA/setup_environment.py
```

The setup detects CUDA 12 or 13 from `nvcc` (falling back to `nvidia-smi`)
and installs the matching `cupy-cuda12x` or `cupy-cuda13x` wheel.

The runner consumes a trial file written by `bench.py`; from the repository
root:

```text
python bench.py run --set perf -p myokit_cuda
```

or directly, with a trial file already planned:

```text
GPU_ODE_MYOKIT_CUDA/venv/Scripts/python.exe GPU_ODE_MYOKIT_CUDA/bench_myokit_cuda.py --trials trials/<key>/myokit_cuda.jsonl
```

On Linux, use `venv/bin/python`. Myokit's CUDA exporter implements forward
Euler in float32 only, so a trial naming another algorithm, precision or
controller lands as a failed row with that reason.
