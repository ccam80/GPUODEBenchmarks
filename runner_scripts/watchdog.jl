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

const DEADLINE_PATH = Ref("")
const KILLER = Ref{Union{Nothing, Base.Process}}(nothing)

# The killer's loop: every second, exit when the parent is gone, else end it with the watchdog code once the deadline file's time has passed.
const KILLER_SCRIPT = """
h = ccall((:OpenProcess, "kernel32"), Ptr{Cvoid}, (UInt32, Cint, UInt32), 0x00100001, 0, PID)
h == C_NULL && exit()
while true
    sleep(1)
    ccall((:WaitForSingleObject, "kernel32"), UInt32, (Ptr{Cvoid}, UInt32), h, 0) == 0 && exit()
    deadline = try parse(Float64, read(PATH, String)) catch; 0.0 end
    if deadline > 0 && time() > deadline
        ccall((:TerminateProcess, "kernel32"), Cint, (Ptr{Cvoid}, UInt32), h, CODE)
        rm(PATH; force = true)
        exit()
    end
end
"""

"On Windows, start the killer process once; it ends this process from outside when an armed deadline passes."
function start_killer()
    Sys.iswindows() || return
    KILLER[] === nothing || return
    DEADLINE_PATH[] = joinpath(tempdir(), "watchdog_$(getpid()).deadline")
    write(DEADLINE_PATH[], "0")
    script = replace(KILLER_SCRIPT, "PID" => string(getpid()), "PATH" => repr(DEADLINE_PATH[]),
                     "CODE" => string(WATCHDOG_EXIT_CODE))
    KILLER[] = run(`$(Base.julia_cmd()) --threads=1 --startup-file=no -e $script`; wait = false)
    atexit() do
        KILLER[] === nothing || kill(KILLER[])
        rm(DEADLINE_PATH[]; force = true)
    end
end

"Set the killer's deadline budget_s from now."
function arm_deadline(budget_s)
    start_killer()
    isempty(DEADLINE_PATH[]) || write(DEADLINE_PATH[], string(time() + budget_s))
end

"Clear the killer's deadline."
function clear_deadline()
    isempty(DEADLINE_PATH[]) || write(DEADLINE_PATH[], "0")
end

"Run f() under the watchdog; when it has not returned after budget_s (the soft cap plus 30 s), run on_breach() and hard-exit. `external` also arms the killer, which a caller timing f arms itself outside the timed call."
function run_watchdogged(f, on_breach; budget_s = WATCHDOG_SECONDS + 30.0, external = true)
    external && arm_deadline(budget_s)
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
        external && clear_deadline()
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
        arm_deadline(cap_s + 30.0)
        elapsed = @elapsed result = run_watchdogged(f, on_breach; budget_s = cap_s + 30.0, external = false)
        clear_deadline()
        push!(samples, elapsed * 1000.0)
        elapsed > cap_s && return (NaN, samples, result)
        length(samples) == 1 && continue   # the warm-up carries the compile
        push!(timed, elapsed)
        length(timed) == 1 && ((lo, hi) = repeat_bounds(timed[1], repeats))
        repeats_done(timed, lo, hi) &&
            return (minimum(timed) * 1000.0, samples, result)
    end
end
