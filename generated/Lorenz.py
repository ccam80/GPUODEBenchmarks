#-2016637299862493869

# This file was generated automatically by Cubie. Don't make changes in here - they'll just be overwritten! Instead, modify the sympy input which you used to define the file.
from numba import cuda, int32
import math
from cubie.cuda_simsafe import *


# AUTO-GENERATED DXDT FACTORY
def dxdt_factory(constants, precision):
    """Auto-generated dxdt factory."""
    sigma = precision(constants['sigma'])
    beta = precision(constants['beta'])
    
    @cuda.jit((precision[::1],
               precision[::1],
               precision[::1],
               precision[::1],
               precision[::1],
               precision),
              device=True,
              inline=True)
    def dxdt(state, parameters, drivers, observables, out, t):
        out[2] = -beta*state[2] + state[0]*state[1]
        _cse0 = -state[1]
        out[1] = _cse0 + state[0]*(parameters[0] - state[2])
        out[0] = sigma*(-_cse0 - state[0])
    
    return dxdt

# AUTO-GENERATED OBSERVABLES FACTORY
def observables_factory(constants, precision):
    """Auto-generated observables factory."""
    sigma = precision(constants['sigma'])
    beta = precision(constants['beta'])
    @cuda.jit((precision[::1],
               precision[::1],
               precision[::1],
               precision[::1],
               precision),
              device=True,
              inline=True)
    def get_observables(state, parameters, drivers, observables, t):
        pass
    
    return get_observables
