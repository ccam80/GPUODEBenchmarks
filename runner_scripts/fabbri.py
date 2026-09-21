"""The Fabbri-Linder problem shared by every package: the CellML file, the reference state order, the cascade inputs and the map from a grid value (an index in 0..131071, rounded and clamped) to an (ACh, Iso) point: indices below 1024 are a 32 x 32 lattice with both range ends (ACh = 100 nM x i/31, Iso = 1000 nM x j/31, ACh slow), the rest fill the plane by 17-bit reversal (top 8 bits an ACh level of 256, low 9 bits an Iso level of 512); fabbri.jl matches it bit for bit."""

import os

import numpy as np

PROBLEM = "fabbri_linder"
PARAMETER = "ach_iso"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "fabbri_linder.cellml")

# The Linder cAMP-cascade switch; the analogue inputs are inert while it is 0.
ANS_CONSTANT = "Rate_modulation_experiments_ANS"
ACH_PARAMETER = "Rate_modulation_experiments_ACh_cas"
ISO_PARAMETER = "Rate_modulation_experiments_Iso_cas"
# Myokit's spelling of the same variables.
ANS_QNAME = "Rate_modulation_experiments.ANS"
ACH_QNAME = "Rate_modulation_experiments.ACh_cas"
ISO_QNAME = "Rate_modulation_experiments.Iso_cas"
# cellmlmanip's spelling of the membrane voltage, for cubie's GHK singularity rewrite.
VOLTAGE_RAW = "Membrane$V_ode"

INDEX_BITS = 17
ACH_BITS = 8
ISO_BITS = INDEX_BITS - ACH_BITS
ACH_LEVELS = 1 << ACH_BITS
ISO_LEVELS = 1 << ISO_BITS
LATTICE_POINTS = 1 << INDEX_BITS
ACH_MAX_NM = 100.0
ISO_MAX_NM = 1000.0
# The traced lattice at the head of the grid: 32 ACh levels x 32 Iso levels, range ends included.
TRACE_SIDE = 32
TRACE_POINTS = TRACE_SIDE * TRACE_SIDE

# The 35 states in CellML document order: the order the finals of every package are stored in.
STATE_ORDER = (
    "Membrane_V_ode", "Nai_concentration_Nai_", "i_f_y_gate_y", "i_Na_m_gate_m", "i_Na_h_gate_h",
    "i_CaL_dL_gate_dL", "i_CaL_fL_gate_fL", "i_CaL_fCa_gate_fCa", "i_CaT_dT_gate_dT", "i_CaT_fT_gate_fT",
    "Ca_SR_release_R", "Ca_SR_release_O", "Ca_SR_release_I", "Ca_SR_release_RI", "Ca_buffering_fTMM",
    "Ca_buffering_fCMi", "Ca_buffering_fCMs", "Ca_buffering_fTC", "Ca_buffering_fTMC", "Ca_buffering_fCQ",
    "Ca_dynamics_Cai", "Ca_dynamics_Ca_nsr", "Ca_dynamics_Ca_jsr", "Ca_dynamics_Ca_sub",
    "i_Kur_rKur_gate_r_Kur", "i_Kur_sKur_gate_s_Kur", "i_to_q_gate_q", "i_to_r_gate_r", "i_Kr_pa_gate_paS",
    "i_Kr_pa_gate_paF", "i_Kr_pi_gate_piy", "i_Ks_n_gate_n", "i_KACh_a_gate_a", "cAMP_cAMP", "PLBp_PLBp",
)


def myokit_name(qname):
    """cubie's flattening of a Myokit qualified name: the component dot becomes an underscore."""
    return qname.replace(".", "_")


def lattice_index(values):
    """The lattice index of each grid value: rounded to the nearest integer (ties to even) and clamped to 0..131071."""
    rounded = np.rint(np.asarray(values, dtype=np.float64))
    return np.clip(rounded, 0, LATTICE_POINTS - 1).astype(np.int64)


def bit_reverse(index):
    """The INDEX_BITS-bit reversal of each index."""
    index = np.asarray(index, dtype=np.int64)
    out = np.zeros_like(index)
    for bit in range(INDEX_BITS):
        out |= ((index >> bit) & 1) << (INDEX_BITS - 1 - bit)
    return out


def levels(values):
    """(ach, iso) of each grid value as fractions of the ranges in float64: the head lattice by row and column, every later index by its bit reversal."""
    index = lattice_index(values)
    head = index < TRACE_POINTS
    ach = np.empty(index.shape, dtype=np.float64)
    iso = np.empty(index.shape, dtype=np.float64)
    ach[head] = (index[head] // TRACE_SIDE) / (TRACE_SIDE - 1)
    iso[head] = (index[head] % TRACE_SIDE) / (TRACE_SIDE - 1)
    reversed_index = bit_reverse(index[~head])
    ach[~head] = (reversed_index >> ISO_BITS) / (ACH_LEVELS - 1)
    iso[~head] = (reversed_index & (ISO_LEVELS - 1)) / (ISO_LEVELS - 1)
    return ach, iso


def inputs(values, precision=np.float32):
    """(ach_nm, iso_nm) of each grid value in the run precision, from the float64 fractions."""
    ach, iso = levels(values)
    return (ach * ACH_MAX_NM).astype(precision), (iso * ISO_MAX_NM).astype(precision)


def parameters(values, precision=np.float32):
    """The cubie parameter dict of a grid: the two cascade inputs of each point."""
    ach, iso = inputs(values, precision)
    return {ACH_PARAMETER: ach, ISO_PARAMETER: iso}
