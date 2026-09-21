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

For the Fabbri-Linder model (`runner_scripts/models/fabbri_linder.cellml`,
the published model with the interface declarations Myokit's parser needs)
the adapter sets `Rate_modulation_experiments.ANS` to 1 and promotes
`ACh_cas` and `Iso_cas` to zero-derivative states, so the kernel integrates
37 states with each trajectory's inputs in rows 36 and 37 of the initial
state array (`runner_scripts/fabbri.py`); the finals return the 35 model
states in reference order. A traced run launches `myokit_cuda_trace`, which
keeps the state after each block of steps spanning the 1 ms sample
interval, so the step must divide it.
