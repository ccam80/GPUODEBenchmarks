# Per-run wall-clock watchdog shared by the Julia GPU writers.

const WATCHDOG_SECONDS = parse(Float64,
    get(ENV, "BENCH_WATCHDOG_SECONDS", "120"))

# Exit status of the hard-exit path.
const WATCHDOG_EXIT_CODE = Cint(3)

"Run f() under the watchdog; when it never returns, run on_breach() and hard-exit."
function run_watchdogged(f, on_breach)
    finished = Threads.Atomic{Bool}(false)
    # Margin over the soft cap: only never-returning runs reach the hard exit.
    timer = Timer(WATCHDOG_SECONDS * 2.0 + 30.0) do _
        finished[] && return
        try
            on_breach()
        finally
            flush(stdout)
            flush(stderr)
            # A hung kernel blocks every exit path except a hard exit.
            ccall(:_exit, Cvoid, (Cint,), WATCHDOG_EXIT_CODE)
        end
    end
    try
        return f()
    finally
        finished[] = true
        close(timer)
    end
end

# (limit_s, floor, ceiling) repeat schedule; mirrored in wp_common.py and Bench.cu.
const REPEAT_SCHEDULE = ((0.1, 20, 20), (3.0, 10, 10), (5.0, 5, 10),
    (Inf, 3, 10))
# A leg past its floor stops once median/min - 1 is within this spread.
const REPEAT_SPREAD = 0.02

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

"(ms, samples) after one warm-up; ms is NaN when a run breaches. samples holds every attempt in ms, warm-up first. The repeat count follows the first timed run's duration, capped at `repeats`."
function watchdogged_min_ms(f, on_breach, repeats)
    samples = Float64[]
    timed = Float64[]
    lo = hi = 0
    while true
        elapsed = @elapsed run_watchdogged(f, on_breach)
        push!(samples, elapsed * 1000.0)
        elapsed > WATCHDOG_SECONDS && return (NaN, samples)
        length(samples) == 1 && continue   # the warm-up carries the compile
        push!(timed, elapsed)
        length(timed) == 1 && ((lo, hi) = repeat_bounds(timed[1], repeats))
        repeats_done(timed, lo, hi) &&
            return (minimum(timed) * 1000.0, samples)
    end
end
