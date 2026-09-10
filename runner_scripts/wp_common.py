"""Timing and watchdog helpers shared by the Python runners; the repeat schedule and the watchdog come from protocol.toml."""

import os
import sys
import threading

from protocol import REPEAT_SCHEDULE, REPEAT_SPREAD, WATCHDOG_EXIT_CODE, WATCHDOG_SECONDS


def run_watchdogged(run, on_breach, budget_s=None):
    """Run run(); when it has not returned after budget_s (default: the soft cap plus 30 s), run on_breach() and hard-exit with WATCHDOG_EXIT_CODE, as watchdog.jl does."""
    finished = threading.Event()

    def fire():
        if finished.is_set():
            return
        try:
            on_breach()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            # A hung kernel blocks every exit path except a hard exit.
            os._exit(WATCHDOG_EXIT_CODE)

    # Margin over the soft cap: only never-returning runs reach the hard exit.
    timer = threading.Timer(WATCHDOG_SECONDS + 30.0 if budget_s is None else budget_s, fire)
    timer.daemon = True
    timer.start()
    try:
        return run()
    finally:
        finished.set()
        timer.cancel()


def repeat_bounds(first_s, cap):
    """(floor, ceiling) repeats for a leg whose first timed run took first_s seconds, both capped at cap."""
    for limit, floor, ceiling in REPEAT_SCHEDULE:
        if first_s < limit:
            return min(floor, cap), min(ceiling, cap)


def repeats_done(timed_s, floor, ceiling):
    """True when the timed runs so far settle the leg's minimum: the ceiling is reached, or the floor is and median/min - 1 is within REPEAT_SPREAD."""
    if len(timed_s) >= ceiling:
        return True
    if len(timed_s) < floor:
        return False
    import statistics
    return statistics.median(timed_s) / min(timed_s) - 1.0 <= REPEAT_SPREAD


def timed_min_ms(run, repeats, on_breach=None, setup=None, cap_s=None):
    """(best_ms, result, samples) after one warm-up; best_ms None on a breach of cap_s (default WATCHDOG_SECONDS). samples holds every attempt in ms, warm-up first. The repeat count follows the first timed run's duration, capped at `repeats`. With on_breach, a run that never returns hard-exits through run_watchdogged at the cap plus 30 s. setup() runs untimed before every attempt after the first."""
    import timeit
    cap = WATCHDOG_SECONDS if cap_s is None else float(cap_s)
    samples = []
    timed = []
    floor = ceiling = None
    while True:
        if setup is not None and samples:
            setup()
        elapsed = timeit.default_timer()
        result = (run() if on_breach is None
                  else run_watchdogged(run, on_breach, cap + 30.0))
        elapsed = timeit.default_timer() - elapsed
        samples.append(elapsed * 1000.0)
        if elapsed > cap:
            return None, result, samples
        if len(samples) == 1:
            continue                     # the warm-up carries the compile
        timed.append(elapsed)
        if floor is None:
            floor, ceiling = repeat_bounds(timed[0], repeats)
        if repeats_done(timed, floor, ceiling):
            return min(timed) * 1000.0, result, samples
