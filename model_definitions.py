"""
Model definitions for ODE benchmarks across different Python frameworks.
Contains various ODE systems that can be used for benchmarking.
"""

import numpy as np


def get_lorenz_params(precision=np.float32):
    """
    Get Lorenz system parameters.
    
    Lorenz system equations:
        dx/dt = σ(y - x)
        dy/dt = ρx - y - xz
        dz/dt = xy - βz
    
    where σ = 10.0, β = 8/3, and ρ is the varied parameter
    
    Returns:
        dict: System parameters including initial conditions, constants, etc.
    """
    return {
        'name': 'Lorenz',
        'dimension': 3,
        'initial_conditions': {'x': 1.0, 'y': 0.0, 'z': 0.0},
        'constants': {'sigma': 10.0, 'beta': 8.0/3.0},
        'parameter_name': 'rho',
        'parameter_default': 21.0,
        'parameter_range': (0.0, 21.0),
        'tspan': (0.0, 1.0),
        'precision': precision
    }


def get_vanderpol_params(precision=np.float32):
    """
    Get Van der Pol oscillator parameters.
    
    Van der Pol oscillator equations:
        dx/dt = y
        dy/dt = μ(1 - x²)y - x
    
    where μ is the parameter controlling nonlinearity
    
    Returns:
        dict: System parameters including initial conditions, constants, etc.
    """
    return {
        'name': 'VanDerPol',
        'dimension': 2,
        'initial_conditions': {'x': 2.0, 'y': 0.0},
        'constants': {},
        'parameter_name': 'mu',
        'parameter_default': 1.0,
        'parameter_range': (0.1, 5.0),
        'tspan': (0.0, 2.0 * np.pi),
        'precision': precision
    }


def get_model_params(model_name, precision=np.float32):
    """
    Get model parameters by name.
    
    Args:
        model_name (str): Name of the model ('lorenz' or 'vanderpol')
        precision: NumPy dtype for precision (np.float32 or np.float64)
    
    Returns:
        dict: Model parameters
    """
    model_name_lower = model_name.lower()
    if model_name_lower == 'lorenz':
        return get_lorenz_params(precision)
    elif model_name_lower == 'vanderpol':
        return get_vanderpol_params(precision)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available models: lorenz, vanderpol")


def get_parameter_list(model_name, n_trajectories, precision=np.float32):
    """
    Get parameter list for ensemble simulations.
    
    Args:
        model_name (str): Name of the model
        n_trajectories (int): Number of trajectories/parameter values
        precision: NumPy dtype for precision
    
    Returns:
        np.ndarray: Array of parameter values
    """
    params = get_model_params(model_name, precision)
    param_min, param_max = params['parameter_range']
    return np.linspace(param_min, param_max, n_trajectories, dtype=precision)


# CUBIE-specific model definitions
def get_cubie_system(model_name, precision=np.float32):
    """
    Get CUBIE system definition string and parameters.
    
    Args:
        model_name (str): Name of the model
        precision: NumPy dtype for precision
    
    Returns:
        tuple: (system_string, initial_conditions, parameter_name, constants)
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower == 'lorenz':
        system_string = """
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        """
        params = get_lorenz_params(precision)
        return (
            system_string,
            params['initial_conditions'],
            params['parameter_name'],
            params['constants']
        )
    elif model_name_lower == 'vanderpol':
        system_string = """
        dx = y
        dy = mu * (1.0 - x * x) * y - x
        """
        params = get_vanderpol_params(precision)
        return (
            system_string,
            params['initial_conditions'],
            params['parameter_name'],
            params['constants']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# JAX-specific model class definitions
def get_jax_model_class(model_name):
    """
    Get JAX model class definition.
    
    Args:
        model_name (str): Name of the model
    
    Returns:
        class: JAX model class
    """
    import jax.numpy as jnp
    import equinox as eqx
    
    model_name_lower = model_name.lower()
    
    if model_name_lower == 'lorenz':
        class Lorenz(eqx.Module):
            rho: float

            def __call__(self, t, y, args):
                f0 = 10.0 * (y[1] - y[0])
                f1 = self.rho * y[0] - y[1] - y[0] * y[2]
                f2 = y[0] * y[1] - (8/3) * y[2]
                return jnp.stack([f0, f1, f2])
        return Lorenz
    
    elif model_name_lower == 'vanderpol':
        class VanDerPol(eqx.Module):
            mu: float

            def __call__(self, t, y, args):
                f0 = y[1]
                f1 = self.mu * (1.0 - y[0] * y[0]) * y[1] - y[0]
                return jnp.stack([f0, f1])
        return VanDerPol
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


# PyTorch-specific model class definitions
def get_pytorch_model_class(model_name):
    """
    Get PyTorch model class definition.
    
    Args:
        model_name (str): Name of the model
    
    Returns:
        class: PyTorch model class
    """
    import torch
    import torch.nn as nn
    
    model_name_lower = model_name.lower()
    
    if model_name_lower == 'lorenz':
        class LorenzODE(torch.nn.Module):
            def __init__(self, rho=torch.tensor(21.0)):
                super(LorenzODE, self).__init__()
                self.sigma = nn.Parameter(torch.as_tensor([10.0]))
                self.rho = nn.Parameter(rho)
                self.beta = nn.Parameter(torch.as_tensor([8/3]))

            def forward(self, t, u):
                x, y, z = u[0], u[1], u[2]
                du1 = self.sigma[0] * (y - x)
                du2 = x * (self.rho - z) - y
                du3 = x * y - self.beta[0] * z
                return torch.stack([du1, du2, du3])
        return LorenzODE
    
    elif model_name_lower == 'vanderpol':
        class VanDerPolODE(torch.nn.Module):
            def __init__(self, mu=torch.tensor(1.0)):
                super(VanDerPolODE, self).__init__()
                self.mu = nn.Parameter(mu)

            def forward(self, t, u):
                x, y = u[0], u[1]
                du1 = y
                du2 = self.mu * (1.0 - x * x) * y - x
                return torch.stack([du1, du2])
        return VanDerPolODE
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
