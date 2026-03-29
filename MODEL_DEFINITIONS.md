# ODE Model Definitions

This repository now includes centralized model definitions that can be used across different benchmark frameworks (Julia, Python, C++).

## Available Models

1. **Lorenz System** (default)
   - 3-dimensional chaotic system
   - Parameter range: ρ ∈ [0, 21]
   - Initial conditions: x=1.0, y=0.0, z=0.0
   - Time span: [0, 1]

2. **Van der Pol Oscillator**
   - 2-dimensional nonlinear oscillator
   - Parameter range: μ ∈ [0.1, 5.0]
   - Initial conditions: x=2.0, y=0.0
   - Time span: [0, 2π]

## Usage

### Julia Benchmarks

The Julia benchmarks accept an optional second argument to specify the model:

```bash
# Default (Lorenz)
julia --project=./GPU_ODE_Julia bench_cpu.jl 8192

# Explicitly specify Lorenz
julia --project=./GPU_ODE_Julia bench_cpu.jl 8192 lorenz

# Use Van der Pol
julia --project=./GPU_ODE_Julia bench_cpu.jl 8192 vanderpol
```

The same applies to GPU benchmarks:

```bash
julia --project=./GPU_ODE_Julia bench_lorenz_gpu.jl 8192 vanderpol
```

### Python Benchmarks

Python benchmarks (CUBIE, JAX, PyTorch) also accept an optional model argument:

```bash
# CUBIE
python3 GPU_ODE_CUBIE/bench_cubie.py 8192 lorenz
python3 GPU_ODE_CUBIE/bench_cubie.py 8192 vanderpol

# JAX
python3 GPU_ODE_JAX/bench_diffrax.py 8192 lorenz
python3 GPU_ODE_JAX/bench_diffrax.py 8192 vanderpol

# PyTorch
python3 GPU_ODE_PyTorch/bench_torchdiffeq.py 8192 lorenz
python3 GPU_ODE_PyTorch/bench_torchdiffeq.py 8192 vanderpol
```

### C++ Benchmarks (MPGOS)

C++ benchmarks require compile-time model selection. The system definition is included at compile time:

For Lorenz (default):
```bash
nvcc -I./SourceCodes Lorenz.cu -o Lorenz
./Lorenz
```

For Van der Pol, you would need to:
1. Create a new `.cu` file (e.g., `VanDerPol.cu`)
2. Include `VanDerPol_SystemDefinition.cuh` instead of `Lorenz_SystemDefinition.cuh`
3. Update system dimension constants (SD=2 for Van der Pol vs SD=3 for Lorenz)
4. Compile and run the new executable

## Model Definition Files

### Julia
- Location: `GPU_ODE_Julia/src/model_definitions.jl`
- Functions: `get_model(model_name, Type)`, `get_parameter_range(model_name, npoints, Type)`

### Python
- Location: `model_definitions.py` (root directory)
- Functions:
  - `get_model_params(model_name, precision)` - Get model parameters
  - `get_parameter_list(model_name, n_trajectories, precision)` - Get parameter sweep
  - `get_cubie_system(model_name, precision)` - CUBIE-specific system string
  - `get_jax_model_class(model_name)` - JAX model class
  - `get_pytorch_model_class(model_name)` - PyTorch model class

### C++
- Location: `GPU_ODE_MPGOS/*_SystemDefinition.cuh`
- Files:
  - `Lorenz_SystemDefinition.cuh` - Lorenz system
  - `VanDerPol_SystemDefinition.cuh` - Van der Pol oscillator

## Adding New Models

To add a new ODE system:

1. **Julia**: Add the ODE function and parameters to `GPU_ODE_Julia/src/model_definitions.jl`
2. **Python**: Add model-specific functions to `model_definitions.py`
3. **C++**: Create a new `*_SystemDefinition.cuh` file following the template structure

Each model definition should include:
- ODE equations (right-hand side function)
- Default initial conditions
- Parameter name and default value
- Suggested parameter range for ensemble simulations
- Time span for integration
