# Per-run wall-clock watchdog shared by the Julia GPU writers.

const WATCHDOG_SECONDS = parse(Float64,
    get(ENV, "BENCH_WATCHDOG_SECONDS", "120"))

"Run f() under the watchdog; when it never returns, run on_breach() and hard-exit."
function run_watchdogged(f, on_breach)
    finished = Threads.Atomic{Bool}(false)
    # Margin over the soft cap: a run finishing just past it is still
    # recorded in-process; the hard exit is for runs that never return.
    timer = Timer(WATCHDOG_SECONDS * 2.0 + 30.0) do _
        finished[] && return
        try
            on_breach()
        finally
            flush(stdout)
            flush(stderr)
            # A hung kernel blocks every exit path except a hard exit.
            ccall(:_exit, Cvoid, (Cint,), 0)
        end
    end
    try
        return f()
    finally
        finished[] = true
        close(timer)
    end
end

"Best-of-repeats wall time in ms after one warm-up; NaN when a run breaches."
function watchdogged_min_ms(f, on_breach, repeats)
    best = Inf
    for attempt in 0:repeats
        elapsed = @elapsed run_watchdogged(f, on_breach)
        elapsed > WATCHDOG_SECONDS && return NaN
        attempt > 0 && (best = min(best, elapsed))
    end
    return best * 1000.0
end
