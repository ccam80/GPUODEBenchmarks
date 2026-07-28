"""Compile and launch Myokit's generated CUDA Euler kernel.

Myokit's ``cuda-kernel`` exporter generates a single-cell
``iterate_euler_cu`` device function.  This module leaves that function and
its generated equations intact, appends an ensemble launch kernel, and
compiles the combined source with CuPy's NVRTC-backed ``RawModule``.
"""

import tempfile
from pathlib import Path

import numpy as np


_LAUNCH_KERNEL = r"""

extern "C" __global__
void myokit_cuda_integrate(
    Real *states,
    const Real *diffusion_current,
    const int cell_count,
    const Real dt,
    const int step_count)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= cell_count) {
        return;
    }

    Real state[NDIM];
    for (int state_index = 0; state_index < NDIM; ++state_index) {
        state[state_index] =
            states[state_index * cell_count + cell];
    }

    const Real input = diffusion_current[cell];
    for (int step = 0; step < step_count; ++step) {
        iterate_euler_cu(dt, state, input, (Real *)0);
    }

    for (int state_index = 0; state_index < NDIM; ++state_index) {
        states[state_index * cell_count + cell] =
            state[state_index];
    }
}
"""

_UNSUPPORTED_NVRTC_INCLUDE = "#include <float.h>\n"


def _import_dependencies():
    """Import optional runtime dependencies with an actionable error."""
    try:
        import myokit
    except ImportError as exc:
        raise ImportError(
            "Myokit is required; run GPU_ODE_MYOKIT_CUDA/"
            "setup_environment.py first."
        ) from exc
    try:
        import cupy
    except ImportError as exc:
        raise ImportError(
            "A CUDA-major-matched CuPy package is required; run "
            "GPU_ODE_MYOKIT_CUDA/setup_environment.py first."
        ) from exc
    return myokit, cupy


def _ensure_diffusion_binding(model, variable_qname):
    """Give the exporter its required ``diffusion_current`` binding."""
    existing = model.binding("diffusion_current")
    if existing is not None:
        if variable_qname is not None and existing.qname() != variable_qname:
            raise ValueError(
                "model already binds diffusion_current to {0}, not {1}"
                .format(existing.qname(), variable_qname)
            )
        return existing

    if variable_qname is not None:
        variable = model.get(variable_qname)
        if variable.is_state():
            raise ValueError(
                "diffusion_current cannot be bound to state variable {0}"
                .format(variable_qname)
            )
    else:
        component_name = "myokit_cuda_input"
        suffix = 2
        while model.has_component(component_name):
            component_name = "myokit_cuda_input_{0}".format(suffix)
            suffix += 1
        component = model.add_component(component_name)
        variable = component.add_variable("diffusion_current")
        variable.set_rhs(0)

    if variable.binding() is not None:
        raise ValueError(
            "{0} already has binding {1}".format(
                variable.qname(), variable.binding()
            )
        )
    variable.set_binding("diffusion_current")
    return variable


def _validate_float32(array, name, expected_shape=None):
    """Return a contiguous float32 array with an optional shape check."""
    result = np.ascontiguousarray(array, dtype=np.float32)
    if expected_shape is not None and result.shape != expected_shape:
        raise ValueError(
            "{0} has shape {1}, expected {2}".format(
                name, result.shape, expected_shape
            )
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("{0} contains non-finite values".format(name))
    return result


class MyokitCudaModel:
    """An ensemble wrapper around a Myokit-generated CUDA Euler step.

    Parameters
    ----------
    cellml_path : path-like
        CellML model to import with Myokit.
    diffusion_variable : str, optional
        Qualified Myokit variable name to bind to ``diffusion_current``.
        The value passed for each trajectory then replaces this variable in
        the exported equations.  When omitted, a zero-valued unused binding
        is added for models such as Fabbri-Linder.
    block_size : int, optional
        CUDA launch block size.
    """

    def __init__(
        self,
        cellml_path,
        diffusion_variable=None,
        block_size=128,
    ):
        myokit, cupy = _import_dependencies()
        self._cupy = cupy
        self.cellml_path = Path(cellml_path).resolve()
        if not self.cellml_path.is_file():
            raise FileNotFoundError(str(self.cellml_path))
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = int(block_size)

        importer = myokit.formats.importer("cellml")
        model = importer.model(str(self.cellml_path))
        _ensure_diffusion_binding(model, diffusion_variable)
        model.validate()

        self.state_names = tuple(state.qname() for state in model.states())
        self.initial_state = _validate_float32(
            model.initial_values(as_floats=True),
            "initial state",
            (len(self.state_names),),
        )
        self.diffusion_variable = diffusion_variable

        with tempfile.TemporaryDirectory(
            prefix="myokit_cuda_export_"
        ) as export_dir:
            exporter = myokit.formats.exporter("cuda-kernel")
            exporter.runnable(export_dir, model)
            generated_path = Path(export_dir) / "kernel.cu"
            self.generated_source = generated_path.read_text(
                encoding="utf-8"
            )

        include_count = self.generated_source.count(
            _UNSUPPORTED_NVRTC_INCLUDE
        )
        if include_count != 1:
            raise RuntimeError(
                "expected exactly one Myokit float.h include, found {0}"
                .format(include_count)
            )
        # NVRTC has no host C standard-library include path.  Myokit's
        # generated single-precision kernel does not use anything from this
        # header, so remove only that exact include and preserve all generated
        # equations byte-for-byte.
        nvrtc_source = self.generated_source.replace(
            _UNSUPPORTED_NVRTC_INCLUDE,
            "",
            1,
        )
        self.cuda_source = nvrtc_source + _LAUNCH_KERNEL
        self._module = cupy.RawModule(
            code=self.cuda_source,
            options=("--std=c++11",),
            name_expressions=("myokit_cuda_integrate",),
        )
        self._kernel = self._module.get_function(
            "myokit_cuda_integrate"
        )

    @property
    def state_count(self):
        """Number of ODE states in the imported model."""
        return len(self.state_names)

    def initial_states(self, cell_count):
        """Return identical initial states with shape ``(states, cells)``."""
        if cell_count <= 0:
            raise ValueError("cell_count must be positive")
        return np.repeat(
            self.initial_state[:, np.newaxis],
            int(cell_count),
            axis=1,
        )

    def solve(
        self,
        dt,
        step_count,
        cell_count=None,
        initial_states=None,
        diffusion_values=None,
    ):
        """Run a fixed number of generated forward-Euler steps.

        The returned host array has shape ``(cells, states)``.  Compilation
        happens when the object is constructed and is therefore separable
        from timed calls to this method.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if int(step_count) != step_count or step_count <= 0:
            raise ValueError("step_count must be a positive integer")
        step_count = int(step_count)

        if initial_states is None:
            if cell_count is None:
                raise ValueError(
                    "cell_count is required when initial_states is omitted"
                )
            host_states = self.initial_states(int(cell_count))
        else:
            host_states = _validate_float32(
                initial_states, "initial_states"
            )
            if host_states.ndim != 2:
                raise ValueError(
                    "initial_states must have shape (states, cells)"
                )
            if host_states.shape[0] != self.state_count:
                raise ValueError(
                    "initial_states has {0} states, expected {1}".format(
                        host_states.shape[0], self.state_count
                    )
                )
            if cell_count is not None and host_states.shape[1] != cell_count:
                raise ValueError(
                    "cell_count does not match initial_states"
                )
            cell_count = host_states.shape[1]

        cell_count = int(cell_count)
        if diffusion_values is None:
            host_diffusion = np.zeros(cell_count, dtype=np.float32)
        else:
            host_diffusion = _validate_float32(
                diffusion_values,
                "diffusion_values",
                (cell_count,),
            )

        device_states = self._cupy.asarray(host_states)
        device_diffusion = self._cupy.asarray(host_diffusion)
        grid_size = (
            (cell_count + self.block_size - 1) // self.block_size
        )
        self._kernel(
            (grid_size,),
            (self.block_size,),
            (
                device_states,
                device_diffusion,
                np.int32(cell_count),
                np.float32(dt),
                np.int32(step_count),
            ),
        )
        self._cupy.cuda.get_current_stream().synchronize()
        return self._cupy.asnumpy(device_states).T
