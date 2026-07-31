#!/bin/bash
# Drive the complete benchmark dataset in one run:
#
#   1. performance    - N-sweep timing benchmarks
#   2. work-precision - error-vs-time sweeps
#   3. numerical      - Float32 fixed/adaptive sweeps, cubie vs DifferentialEquations.jl
#   4. overlap        - per-algorithm cubie vs DiffEqGPU comparison
#   5. plots          - plots + pairwise comparison reports
#
# A package that runs out of GPU memory stops only its own sweep; completed
# points stay on disk and the run continues. A failed analysis never aborts
# the others. Use --resume-from to continue a part-finished sweep.
#
# GPU clocks are pinned for the whole run (see runner_scripts/clock_guard.sh);
# without passwordless root the run proceeds unlocked and reports any drift.
#
# Usage:
#   ./run_full_dataset.sh                           # every analysis, every package
#   ./run_full_dataset.sh -n 33554432               # larger ceiling
#   ./run_full_dataset.sh -p cpp                    # one package
#   ./run_full_dataset.sh -a overlap                # one analysis
#   ./run_full_dataset.sh -a performance,numerical  # several analyses
#   ./run_full_dataset.sh --resume-from jax         # restart at a package
#   ./run_full_dataset.sh --lock-clocks 1470,6801   # override the clock target
#   ./run_full_dataset.sh --no-lock-clocks          # sample clocks but do not pin
#   ./run_full_dataset.sh --clock-tolerance 30      # widen the drift threshold (MHz)
#
#   -p, --package   all (default) | julia | cpp | pytorch | jax | cubie | cubie_mlir | myokit_cuda
#   -a, --analysis  all (default) | performance | work-precision | numerical | overlap | plots
#   -n, --nmax      largest trajectory count for a performance sweep (default 16777216)
#
# On Windows, run_full_dataset.bat takes the same flags.
#
# Exit code: 0 if every analysis and package succeeded, 1 if any did not.
# Clock drift in a timed analysis also fails the run.

set -u
cd "$(dirname "$0")" || exit 1

NMAX=16777216
DO_PERF=true
DO_WP=true
DO_NE=true
DO_OVERLAP=true
DO_PLOTS=true
OVERLAP_PROFILE="full"
PACKAGE="all"
COOLDOWN=15
RESUME_FROM=""
ALLOW_UNKNOWN_GPU=false
LOCK_CLOCKS=true
CLOCK_TARGET=""          # "SM[,MEM]"; empty means use the per-GPU table

ALL_PACKAGES=(julia cpp pytorch jax cubie cubie_mlir myokit_cuda)

source ./runner_scripts/clock_guard.sh

usage() {
    sed -n '2,35p' "$0" | sed 's/^# \?//'
    exit "${1:-0}"
}

# `-a plots` redraws from disk; otherwise a plot follows the data this run made.
PLOT_ALL=false

set_analyses() {
    DO_PERF=false; DO_WP=false; DO_NE=false; DO_OVERLAP=false; DO_PLOTS=false
    local item
    for item in ${1//,/ }; do
        case "$item" in
            all) DO_PERF=true; DO_WP=true; DO_NE=true; DO_OVERLAP=true; DO_PLOTS=true;;
            performance) DO_PERF=true; DO_PLOTS=true;;
            work-precision) DO_WP=true; DO_PLOTS=true;;
            numerical) DO_NE=true;;
            overlap) DO_OVERLAP=true;;
            plots) DO_PLOTS=true; PLOT_ALL=true;;
            *) echo "Unknown analysis '$item' (all|performance|work-precision|numerical|overlap|plots)"
               exit 1;;
        esac
    done
}

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--nmax) [ $# -ge 2 ] || { echo "$1 requires a value"; exit 1; }
                   NMAX="$2"; shift 2;;
        -p|--package) [ $# -ge 2 ] || { echo "$1 requires a value"; exit 1; }
                   PACKAGE="${2//-/_}"; shift 2;;
        -a|--analysis) [ $# -ge 2 ] || { echo "$1 requires a value"; exit 1; }
                   set_analyses "$2"; shift 2;;
        --profile) [ $# -ge 2 ] || { echo "$1 requires a value"; exit 1; }
                   OVERLAP_PROFILE="$2"; shift 2;;
        --resume-from) [ $# -ge 2 ] || { echo "$1 requires a value"; exit 1; }
                   RESUME_FROM="${2//-/_}"; shift 2;;
        --cooldown) [ $# -ge 2 ] || { echo "$1 requires a value"; exit 1; }
                   COOLDOWN="$2"; shift 2;;
        --allow-unknown-gpu) ALLOW_UNKNOWN_GPU=true; shift;;
        --lock-clocks) [ $# -ge 2 ] || { echo "$1 requires SM[,MEM]"; exit 1; }
                   CLOCK_TARGET="$2"; LOCK_CLOCKS=true; shift 2;;
        --no-lock-clocks) LOCK_CLOCKS=false; shift;;
        --clock-tolerance) [ $# -ge 2 ] || { echo "$1 requires a value in MHz"; exit 1; }
                   CLOCK_TOL_MHZ="$2"; shift 2;;
        -h|--help) usage 0;;
        *) echo "Unknown option $1"; usage 1;;
    esac
done

if [ "$PACKAGE" == "all" ]; then
    LANGUAGES=("${ALL_PACKAGES[@]}")
elif [[ " ${ALL_PACKAGES[*]} " == *" $PACKAGE "* ]]; then
    LANGUAGES=("$PACKAGE")
else
    echo "Unknown package '$PACKAGE' (all|$(IFS='|'; echo "${ALL_PACKAGES[*]}"))"
    exit 1
fi

DATASET_KEY="$(bash ./runner_scripts/bench_key.sh)"

# Every output file is keyed by "<os>_<gpu>". An unidentifiable GPU means
# nvidia-smi could not talk to the driver, so the benchmarks would fail anyway
# and the whole dataset would be mislabelled "unknown-gpu". Stop before doing
# hours of work that has to be thrown away.
if [ "${DATASET_KEY##*_}" = "unknown-gpu" ] && [ "$ALLOW_UNKNOWN_GPU" != true ]; then
    echo "✗ Could not identify the GPU — dataset key would be '$DATASET_KEY'." >&2
    echo "  nvidia-smi reports:" >&2
    nvidia-smi 2>&1 | head -3 | sed 's/^/    /' >&2
    echo "  Fix the driver (a 'Driver/library version mismatch' usually means" >&2
    echo "  the nvidia kernel modules need reloading or the host rebooting)," >&2
    echo "  or pass --allow-unknown-gpu to run anyway." >&2
    exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="logs/${DATASET_KEY}_${STAMP}"
mkdir -p "$LOG_DIR"
RESULTS="$LOG_DIR/summary.tsv"
: > "$RESULTS"

# ------------------------------------------------------------------ GPU clocks
# Pin before any GPU work starts; unpin on every exit path, including Ctrl-C.
CLOCK_STATUS="off"
if $LOCK_CLOCKS; then
    if clocks_configure "$DATASET_KEY" "$CLOCK_TARGET"; then
        if clocks_lock; then
            CLOCK_STATUS="locked SM=$CLOCK_SM${CLOCK_MEM:+ MEM=$CLOCK_MEM}"
        else
            CLOCK_STATUS="unlocked (no root) — target was SM=$CLOCK_SM"
        fi
    else
        CLOCK_STATUS="unlocked (no target configured)"
    fi
fi
trap 'clocks_monitor_stop; clocks_reset' EXIT INT TERM
clocks_monitor_start "$LOG_DIR/clocks.csv" || true

# data/<dir>/<prefix>_times_*.txt per framework, for the progress report.
data_dir_for() {
    case "$1" in
        julia) echo "Julia";;
        cpp) echo "CPP";;
        *) echo "${1^^}";;
    esac
}
data_prefix_for() {
    case "$1" in
        julia) echo "Julia";; cpp) echo "MPGOS";; jax) echo "Jax";;
        pytorch) echo "Torch";; cubie) echo "Cubie";;
        cubie_mlir) echo "Cubie_mlir";; myokit_cuda) echo "Myokit_cuda";;
    esac
}

# Largest N actually recorded for a framework, so a truncated sweep is visible
# in the summary rather than silently looking like a plain failure.
max_n_reached() {
    local dir prefix f best=0 n
    dir="data/$(data_dir_for "$1")"
    prefix="$(data_prefix_for "$1")"
    [ -d "$dir" ] || { echo 0; return; }
    for f in "$dir/${DATASET_KEY}/${prefix}"_times_*.txt; do
        [ -f "$f" ] || continue
        n=$(awk 'NF{print $1}' "$f" | sort -n | tail -1)
        [ -n "$n" ] && [ "${n%.*}" -gt "$best" ] 2>/dev/null && best="${n%.*}"
    done
    echo "$best"
}

record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$RESULTS"; }

hr() { printf '=%.0s' {1..60}; echo; }

# CLOCK_CRITICAL: drift fails the step. CLOCK_CHECK=false skips non-GPU steps.
CLOCK_CRITICAL=true
CLOCK_CHECK=true
CLOCK_FAILURES=0
STEP_LABEL=""

# Run one labelled step, tee'd to its own log, never aborting the outer run.
run_step() {
    local label="$1" logfile="$2"; shift 2
    local start end status cstart cend
    hr; echo "[$(date -u +%H:%M:%SZ)] $label"; hr
    start=$(date +%s)
    cstart="$(clocks_stamp)"
    "$@" 2>&1 | tee "$LOG_DIR/$logfile"
    status=${PIPESTATUS[0]}
    cend="$(clocks_stamp end)"
    end=$(date +%s)
    if [ "$status" -eq 0 ]; then
        echo "✓ $label  ($((end - start))s)"
    else
        echo "✗ $label failed with exit $status  ($((end - start))s) — continuing"
    fi
    # Check this step's slice of the whole-run clock log.
    if $CLOCK_CHECK; then
        clocks_check "$cstart" "$cend" "${STEP_LABEL:-$label}" "$CLOCK_CRITICAL" \
            || CLOCK_FAILURES=$((CLOCK_FAILURES + 1))
    fi
    return "$status"
}

echo "Dataset key : $DATASET_KEY"
echo "nmax        : $NMAX"
echo "Overlap     : profile=$OVERLAP_PROFILE"
echo "Packages    : ${LANGUAGES[*]}"
echo "Log dir     : $LOG_DIR"
echo "Analyses    : performance=$DO_PERF work-precision=$DO_WP numerical=$DO_NE overlap=$DO_OVERLAP plots=$DO_PLOTS"
echo "Clocks      : $CLOCK_STATUS"
[ -n "$RESUME_FROM" ] && echo "Resume from : $RESUME_FROM"
echo

# Provenance for the dataset this run produces.
{
    echo "dataset_key=$DATASET_KEY"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "nmax=$NMAX"
    echo "overlap_profile=$OVERLAP_PROFILE"
    echo "packages=${LANGUAGES[*]}"
    echo "git_rev=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty=$(test -n "$(git status --porcelain 2>/dev/null)" && echo yes || echo no)"
    echo "host=$(uname -a)"
    echo "clocks=$CLOCK_STATUS"
    echo "clock_target_sm=${CLOCK_SM:-none}"
    echo "clock_target_mem=${CLOCK_MEM:-none}"
    echo "clock_tolerance_mhz=$CLOCK_TOL_MHZ"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
        | sed 's/^/gpu=/'
} > "$LOG_DIR/run_manifest.txt"

skipping=false
[ -n "$RESUME_FROM" ] && skipping=true

# ---------------------------------------------------------------- performance
if $DO_PERF; then
    for lang in "${LANGUAGES[@]}"; do
        if $skipping; then
            if [ "$lang" == "$RESUME_FROM" ]; then skipping=false; else
                echo "-- skipping $lang (before --resume-from $RESUME_FROM)"
                record "perf:$lang" "SKIPPED" "-" "-"
                continue
            fi
        fi
        CLOCK_CRITICAL=true; STEP_LABEL="perf:$lang"
        run_step "Performance sweep: $lang (nmax=$NMAX)" "perf_${lang}.log" \
            bash ./run_benchmark.sh -p "$lang" -d gpu -m ode -a performance -n "$NMAX"
        status=$?
        reached=$(max_n_reached "$lang")
        if [ "$status" -eq 0 ]; then
            record "perf:$lang" "OK" "maxN=$reached" "${status}"
        elif [ "$reached" -gt 0 ]; then
            # Partial data survived: the sweep died partway (typically OOM).
            record "perf:$lang" "PARTIAL" "maxN=$reached" "${status}"
            echo "  → kept results up to N=$reached; higher N left empty"
        else
            record "perf:$lang" "FAILED" "no data" "${status}"
        fi
        sleep "$COOLDOWN"
    done
fi

# ------------------------------------------------------------- work-precision
if $DO_WP; then
    # The work-precision sweeps score every point against a Float64 golden
    # reference. It is machine independent and generated once, but nothing else
    # creates it -- without it *every* framework's wp run aborts immediately on
    # a missing-file error, so generate it up front rather than failing seven
    # times over.
    if [ ! -f data/numerical/golden_lorenz_32768.csv ]; then
        # Reference generation is scored on accuracy, not speed.
        CLOCK_CRITICAL=false; STEP_LABEL="wp:golden"
        run_step "Golden reference for work-precision" "wp_golden.log" \
            julia -t auto --project=. ./runner_scripts/golden/generate_golden.jl
        status=$?
        if [ "$status" -eq 0 ]; then
            record "wp:golden" "OK" "-" "$status"
        else
            record "wp:golden" "FAILED" "wp sweeps cannot score" "$status"
            echo "  → work-precision sweeps will fail without the golden reference"
        fi
    else
        record "wp:golden" "OK" "already present" "0"
    fi

    skipping=false
    for lang in "${LANGUAGES[@]}"; do
        CLOCK_CRITICAL=true; STEP_LABEL="wp:$lang"
        run_step "Work-precision sweep: $lang" "wp_${lang}.log" \
            bash ./run_benchmark.sh -p "$lang" -d gpu -m ode -a work-precision
        status=$?
        [ "$status" -eq 0 ] && record "wp:$lang" "OK" "-" "$status" \
                            || record "wp:$lang" "FAILED" "-" "$status"
        sleep "$COOLDOWN"
    done
fi

# ------------------------------------------------------ numerical equivalence
if $DO_NE; then
    # Equivalence is a correctness check; its clock does not have to be stable.
    CLOCK_CRITICAL=false; STEP_LABEL="ne"
    run_step "Numerical-equivalence suite (all)" "numerical_equivalence.log" \
        bash ./run_numerical_equivalence.sh
    status=$?
    # Exit 2 means the suite ran but found a mismatching/divergent algorithm:
    # a real result to inspect, not an infrastructure failure.
    case "$status" in
        0) record "ne" "OK" "all equivalent" "$status";;
        2) record "ne" "MISMATCH" "see numerical_equivalence_*.md" "$status";;
        *) record "ne" "FAILED" "-" "$status";;
    esac
fi

# --------------------------------------------------- cubie vs DiffEqGPU overlap
if $DO_OVERLAP && [[ ! " all cubie julia " == *" $PACKAGE "* ]]; then
    record "overlap" "SKIPPED" "$PACKAGE is not in the overlap suite" "-"
elif $DO_OVERLAP; then
    PY=./GPU_ODE_CUBIE/venv/bin/python
    [ -x "$PY" ] || PY=python3
    CLOCK_CRITICAL=true; STEP_LABEL="overlap"
    run_step "Cubie vs DiffEqGPU overlap ($OVERLAP_PROFILE, nmax=$NMAX)" \
        "cubie_julia_overlap.log" \
        "$PY" ./run_cubie_julia_overlap.py \
            --profile "$OVERLAP_PROFILE" -a all -p "$PACKAGE" -n "$NMAX"
    status=$?
    # The launcher already records per-framework failures and keeps going, so a
    # non-zero exit here means at least one worker died, not that all did.
    [ "$status" -eq 0 ] && record "overlap" "OK" "-" "$status" \
                        || record "overlap" "PARTIAL" "a worker failed; see manifest.json" "$status"
fi

# ---------------------------------------------------------- plots and reports
if $DO_PLOTS; then
    # Plotting and reporting do no timed GPU work.
    CLOCK_CRITICAL=false; CLOCK_CHECK=false; STEP_LABEL=""
    if $DO_PERF || $PLOT_ALL; then
        run_step "Timing comparison plot" "plot_ode_comp.log" \
            julia --project=. ./runner_scripts/plot/plot_ode_comp.jl
        status=$?
        [ "$status" -eq 0 ] && record "plot:timing" "OK" "-" "$status" \
                            || record "plot:timing" "FAILED" "-" "$status"
    fi

    if $DO_WP || $PLOT_ALL; then
        run_step "Work-precision plot" "plot_ode_wp.log" \
            julia --project=. ./runner_scripts/plot/plot_ode_wp.jl
        status=$?
        [ "$status" -eq 0 ] && record "plot:wp" "OK" "-" "$status" \
                            || record "plot:wp" "FAILED" "-" "$status"
    fi

    # Exit 3 is "nothing to compare", normal on a single machine. Any other
    # non-zero is a real failure and must count.
    PY=./GPU_ODE_CUBIE/venv/bin/python
    if [ -x "$PY" ]; then
        run_step "Pairwise numerical comparison" "compare_numerical.log" \
            "$PY" compare_numerical_results.py
        status=$?
        case "$status" in
            0) record "compare:pairwise" "OK" "-" "$status";;
            3) record "compare:pairwise" "SKIPPED" "needs >=2 keyed datasets" "$status";;
            *) record "compare:pairwise" "FAILED" "-" "$status";;
        esac
    else
        record "compare:pairwise" "SKIPPED" "cubie venv missing" "-"
    fi
fi

# ---------------------------------------------------------------------- summary
echo
hr
echo "RUN SUMMARY  ($DATASET_KEY)"
hr
printf '%-26s %-16s %s\n' "STAGE" "STATUS" "DETAIL"
printf '%-26s %-16s %s\n' "-----" "------" "------"
while IFS=$'\t' read -r stage status detail _; do
    printf '%-26s %-16s %s\n' "$stage" "$status" "$detail"
done < "$RESULTS"
hr

clocks_monitor_stop
clocks_report && hr

failures=$(awk -F'\t' '$2=="FAILED"' "$RESULTS" | wc -l)
partials=$(awk -F'\t' '$2=="PARTIAL"' "$RESULTS" | wc -l)
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_DIR/run_manifest.txt"
echo "Logs: $LOG_DIR"
echo "Data: ./data    Plots: ./plots"
echo "Clocks: $CLOCK_STATUS  (1 Hz log in $LOG_DIR/clocks.csv)"
[ "$partials" -gt 0 ] && echo "$partials stage(s) partial — expected when frameworks OOM at high N."
if [ "$CLOCK_FAILURES" -gt 0 ]; then
    echo "✗ $CLOCK_FAILURES timed stage(s) drifted. Lower the lock in $CLOCK_CONF"
    echo "  and re-run them with --resume-from."
fi
if [ "$failures" -gt 0 ]; then
    echo "$failures stage(s) failed outright."
    exit 1
fi
[ "$CLOCK_FAILURES" -gt 0 ] && exit 1
exit 0
