# The errored-trajectory rule of the result store, for the Julia writers.

const T_FINAL_RTOL = 1e-4

"One flag per trajectory: a non-finite state, a final time off `duration` by more than T_FINAL_RTOL relative, or a non-empty retcode. `states` is n x k, `t_final` and `retcode` have n entries."
function errored_mask(states, t_final, retcode, duration)
    n = length(t_final)
    size(states, 1) == n || throw(ArgumentError("states has one row per t_final"))
    length(retcode) == n || throw(ArgumentError("retcode has one code per t_final"))
    d = Float64(duration)
    return [any(!isfinite, @view(states[i, :])) ||
            !(abs(Float64(t_final[i]) - d) <= T_FINAL_RTOL * abs(d)) ||
            !isempty(retcode[i]) for i in 1:n]
end

"Percent of trajectories errored_mask marks; NaN for an empty ensemble."
function errored_pct(states, t_final, retcode, duration)
    mask = errored_mask(states, t_final, retcode, duration)
    return isempty(mask) ? NaN : 100.0 * count(mask) / length(mask)
end
