# Errored-percent column shared by the Julia writers and the plot scripts.

# Rows errored past this percent are dropped by the plot scripts.
const MAX_ERRORED_PCT = 10.0

"Percent of trajectories with a non-finite final state; a device array reduces in place."
function errored_pct(finals)
    isempty(finals) && return 0.0
    bad = mapreduce(u -> any(!isfinite, u) ? 1 : 0, +, finals; init = 0)
    return 100.0 * bad / length(finals)
end

"True unless the cell is a number past MAX_ERRORED_PCT."
within_error_budget(pct) = !(pct isa Real && pct > MAX_ERRORED_PCT)
