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

Run it through `bench.py`:

```text
python bench.py run --set perf -p myokit_cuda
```

or on a planned trial file (`venv/bin/python` on Linux):

```text
GPU_ODE_MYOKIT_CUDA/venv/Scripts/python.exe GPU_ODE_MYOKIT_CUDA/bench_myokit_cuda.py --trials trials/<key>/myokit_cuda.jsonl
```

The kernel is fixed-step forward Euler in float32; any other algorithm,
precision or controller records a failed row.
