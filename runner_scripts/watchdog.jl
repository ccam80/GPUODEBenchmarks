# Per-run wall-clock watchdog shared by the Julia GPU writers.

const WATCHDOG_SECONDS = parse(Float64,
    get(ENV, "BENCH_WATCHDOG_SECONDS", "120"))

# Exit status of the hard-exit path, so drivers record the leg as failed.
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

"(ms, samples) after one warm-up; ms is NaN when a run breaches. samples holds every attempt in ms, warm-up first."
function watchdogged_min_ms(f, on_breach, repeats)
    best = Inf
    samples = Float64[]
    for attempt in 0:repeats
        elapsed = @elapsed run_watchdogged(f, on_breach)
        push!(samples, elapsed * 1000.0)
        elapsed > WATCHDOG_SECONDS && return (NaN, samples)
        attempt > 0 && (best = min(best, elapsed))
    end
    return (best * 1000.0, samples)
end
