# The Fabbri-Linder grid, bit for bit with runner_scripts/fabbri.py: an index in 0..131071, rounded and clamped; the first 1024 are a 32 x 32 lattice with the range ends (ACh slow), every later index fills the plane through its 17-bit reversal (top 8 bits an ACh level of 256, low 9 bits an Iso level of 512).

const FABBRI_INDEX_BITS = 17
const FABBRI_ACH_BITS = 8
const FABBRI_ISO_BITS = FABBRI_INDEX_BITS - FABBRI_ACH_BITS
const FABBRI_ACH_LEVELS = 1 << FABBRI_ACH_BITS
const FABBRI_ISO_LEVELS = 1 << FABBRI_ISO_BITS
const FABBRI_LATTICE_POINTS = 1 << FABBRI_INDEX_BITS
const FABBRI_ACH_MAX_NM = 100.0
const FABBRI_ISO_MAX_NM = 1000.0
const FABBRI_TRACE_SIDE = 32
const FABBRI_TRACE_POINTS = FABBRI_TRACE_SIDE * FABBRI_TRACE_SIDE

"The index of a grid value: rounded to the nearest integer (ties to even) and clamped to 0..131071."
fabbri_lattice_index(value) = clamp(round(Int, Float64(value)), 0, FABBRI_LATTICE_POINTS - 1)

"The 17-bit reversal of an index."
function fabbri_bit_reverse(index::Integer)
    out = 0
    for bit in 0:(FABBRI_INDEX_BITS - 1)
        out |= ((index >> bit) & 1) << (FABBRI_INDEX_BITS - 1 - bit)
    end
    return out
end

"(ach, iso) of a grid value as Float64 fractions of the ranges."
function fabbri_levels(value)
    index = fabbri_lattice_index(value)
    if index < FABBRI_TRACE_POINTS
        return (index ÷ FABBRI_TRACE_SIDE) / (FABBRI_TRACE_SIDE - 1), (index % FABBRI_TRACE_SIDE) / (FABBRI_TRACE_SIDE - 1)
    end
    reversed = fabbri_bit_reverse(index)
    return (reversed >> FABBRI_ISO_BITS) / (FABBRI_ACH_LEVELS - 1), (reversed & (FABBRI_ISO_LEVELS - 1)) / (FABBRI_ISO_LEVELS - 1)
end

"(ach_nm, iso_nm) of a grid value in T, from the Float64 fractions."
function fabbri_inputs(value, ::Type{T}) where {T}
    ach, iso = fabbri_levels(value)
    return T(ach * FABBRI_ACH_MAX_NM), T(iso * FABBRI_ISO_MAX_NM)
end
