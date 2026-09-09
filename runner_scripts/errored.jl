# The errored-trajectory rule of store.py for the Julia writers: a non-finite final state, a final time off the duration by more than T_FINAL_RTOL relative, or a non-empty retcode.

const T_FINAL_RTOL = 1e-4

"True when the trajectory's final state or time flags it (a NaN time counts)."
_errored(u, t_final, duration) = any(!isfinite, u) ||
    !(abs(Float64(t_final) - Float64(duration)) <= T_FINAL_RTOL * abs(Float64(duration)))

"bool per trajectory over host finals rows (an (n, states) matrix or a vector of state vectors), final times and retcodes (nothing when the package reports none)."
function errored_mask(finals, t_final, retcode, duration)
    rows = finals isa AbstractMatrix ? eachrow(finals) : finals
    length(rows) == length(t_final) || throw(ArgumentError("t_final has one time per finals row"))
    codes = retcode === nothing ? fill("", length(rows)) : string.(retcode)
    length(codes) == length(rows) || throw(ArgumentError("retcode has one code per finals row"))
    return [_errored(u, t_final[i], duration) || !isempty(codes[i]) for (i, u) in enumerate(rows)]
end

"Percent of trajectories the rule flags; reduces on the device when the final states and times live there. NaN when there are none."
function errored_pct(us_end, ts_end, duration; retcode = nothing)
    isempty(us_end) && return NaN
    if retcode !== nothing
        return 100.0 * count(errored_mask(us_end, ts_end, retcode, duration)) / length(us_end)
    end
    d = Float64(duration)
    bad = mapreduce((u, t) -> _errored(u, t, d) ? 1 : 0, +, us_end, ts_end; init = 0)
    return 100.0 * bad / length(us_end)
end
