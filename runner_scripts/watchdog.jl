# Per-run wall-clock watchdog shared by the Julia GPU writers.

include(joinpath(@__DIR__, "protocol.jl"))

"Exit without unwinding; on Windows through TerminateProcess, which skips the DLL detach a live kernel blocks."
function hard_exit(code)
    if Sys.iswindows()
        handle = ccall((:GetCurrentProcess, "kernel32"), Ptr{Cvoid}, ())
        ccall((:TerminateProcess, "kernel32"), Cint, (Ptr{Cvoid}, Cuint), handle, code)
    end
    ccall(:_exit, Cvoid, (Cint,), code)
end

"Run f() under the watchdog; when it has not returned after budget_s (the soft cap plus 30 s), run on_breach() and hard-exit."
function run_watchdogged(f, on_breach; budget_s = WATCHDOG_SECONDS + 30.0)
    finished = Threads.Atomic{Bool}(false)
    timer = Timer(budget_s) do _
        finished[] && return
        try
            on_breach()
        finally
            flush(stdout)
            flush(stderr)
            # A hung kernel blocks every exit path except a hard exit.
            hard_exit(WATCHDOG_EXIT_CODE)
        end
    end
    try
        return f()
    finally
        finished[] = true
        close(timer)
    end
end

"(floor, ceiling) repeats for a leg whose first timed run took first_s seconds, both capped at cap."
function repeat_bounds(first_s, cap)
    for (limit, lo, hi) in REPEAT_SCHEDULE
        first_s < limit && return (min(lo, cap), min(hi, cap))
    end
end

"Median without a Statistics dependency."
function _median(values)
    sorted = sort(values)
    half = length(sorted) ÷ 2
    return isodd(length(sorted)) ? sorted[half + 1] :
           (sorted[half] + sorted[half + 1]) / 2
end

"True when the timed runs so far settle the leg's minimum: the ceiling is reached, or the floor is and median/min - 1 is within REPEAT_SPREAD."
function repeats_done(timed_s, lo, hi)
    length(timed_s) >= hi && return true
    length(timed_s) < lo && return false
    return _median(timed_s) / minimum(timed_s) - 1.0 <= REPEAT_SPREAD
end

"(ms, samples, result) after one warm-up; ms is NaN when a run breaches cap_s, result is the last solve's return value. samples holds every attempt in ms, warm-up first. The repeat count follows the first timed run's duration, capped at `repeats`."
function watchdogged_min_ms(f, on_breach, repeats; cap_s = WATCHDOG_SECONDS)
    samples = Float64[]
    timed = Float64[]
    lo = hi = 0
    result = nothing
    while true
        elapsed = @elapsed result = run_watchdogged(f, on_breach; budget_s = cap_s + 30.0)
        push!(samples, elapsed * 1000.0)
        elapsed > cap_s && return (NaN, samples, result)
        length(samples) == 1 && continue   # the warm-up carries the compile
        push!(timed, elapsed)
        length(timed) == 1 && ((lo, hi) = repeat_bounds(timed[1], repeats))
        repeats_done(timed, lo, hi) &&
            return (minimum(timed) * 1000.0, samples, result)
    end
end
