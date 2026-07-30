#!/bin/bash
# Clock stability guard for timed benchmarks.
#
# A GPU boosts while cold and slows once the heatsink saturates, so whichever
# framework runs first measures faster. Pinning the clocks removes that bias, and
# because heat or the power cap can override a lock mid-run, the run is sampled at
# 1 Hz and each timed step checked against its own slice of the log.
#
# Sourced, not executed:
#
#   clocks_configure <dataset_key> [sm,mem]  resolve + validate target clocks
#   clocks_lock                              persistence mode + pin (needs root)
#   clocks_reset                             restore the prior state
#   clocks_monitor_start <csv>                begin 1 Hz sampling
#   clocks_monitor_stop                       end sampling
#   clocks_stamp [end]                        timestamp in nvidia-smi's format
#   clocks_check <from> <to> <label> <crit>   drift report for one time window
#   clocks_report                             final per-step stability table
#
# Locking and resetting need root. Sampling, checking and reporting do not, so an
# unlocked run still records what the clocks did.

CLOCK_SM=""              # target SM/graphics clock, MHz ("" = not configured)
CLOCK_MEM=""             # target memory clock, MHz ("" = not locked/checked)
CLOCK_TOL_MHZ=15         # one clock step; anything beyond this is real drift
CLOCK_DRIFT_PCT=1        # >this% of busy samples off target escalates to error
CLOCK_LOCKED=false
CLOCK_PM_RESTORE=""      # persistence mode to put back on reset
CLOCK_CSV=""
CLOCK_MONITOR_PID=""
CLOCK_REPORT_TSV=""
CLOCK_CONF="${CLOCK_CONF:-runner_scripts/gpu_clocks.conf}"

# nvidia-smi for the write operations. Passwordless sudo only: a password prompt
# halfway into an unattended run would hang it.
_clock_smi_priv() {
    if [ "$(id -u)" = 0 ]; then
        nvidia-smi "$@"
    else
        sudo -n nvidia-smi "$@"
    fi
}

_clock_supported() {   # _clock_supported gr|mem <mhz>
    nvidia-smi --query-supported-clocks="$1" --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' ' | grep -qx "$2"
}

# Apply a setting, returning failure if it did not take.
#
# nvidia-smi exits 0 for "Setting locked Memory clocks is not supported", printing
# the refusal and then "All done." Trusting the exit status would leave the drift
# check comparing every sample against a target nothing is holding.
_clock_apply() {
    local out rc
    out="$(_clock_smi_priv "$@" 2>&1)"; rc=$?
    [ "$rc" -eq 0 ] || return 1
    case "$out" in
        *"not supported"*|*"Insufficient Permissions"*|*"Unable to"*) return 1;;
    esac
    return 0
}

# Resolve the target clocks for this machine. An explicit "sm,mem" wins over the
# per-GPU table. Both are validated against the clocks the card offers: -lgc/-lmc
# silently reject an unsupported value, leaving the run unlocked.
clocks_configure() {
    local key="$1" explicit="${2:-}" gpu sm mem
    gpu="${key#*_}"

    if [ -n "$explicit" ]; then
        sm="${explicit%%,*}"
        mem="${explicit#*,}"
        [ "$mem" = "$explicit" ] && mem=""      # "1500" with no memory clock
    elif [ -f "$CLOCK_CONF" ]; then
        read -r sm mem <<<"$(awk -v g="$gpu" '
            $1 ~ /^#/ || NF < 2 { next }
            $1 == g { print $2, $3; exit }' "$CLOCK_CONF")"
    fi

    if [ -z "${sm:-}" ]; then
        echo "⚠ No clock target for '$gpu' in $CLOCK_CONF and none given." >&2
        echo "  Measure one: python3 runner_scripts/calibrate/calibrate_clocks.py" >&2
        echo "  or pass --lock-clocks SM[,MEM]. Continuing unlocked." >&2
        return 1
    fi

    if ! _clock_supported gr "$sm"; then
        echo "✗ SM clock ${sm} MHz is not a supported clock on this GPU." >&2
        echo "  List them: nvidia-smi --query-supported-clocks=gr --format=csv" >&2
        return 1
    fi
    if [ -n "${mem:-}" ] && ! _clock_supported mem "$mem"; then
        echo "✗ Memory clock ${mem} MHz is not a supported clock on this GPU." >&2
        echo "  Supported: $(nvidia-smi --query-supported-clocks=mem --format=csv,noheader,nounits \
                              | tr -d ' ' | sort -un | tr '\n' ' ')" >&2
        return 1
    fi

    CLOCK_SM="$sm"
    CLOCK_MEM="${mem:-}"
    return 0
}

# Pin the clocks. Persistence mode goes on first so the settings survive the gap
# between one framework's process exiting and the next one starting.
clocks_lock() {
    [ -n "$CLOCK_SM" ] || return 1

    if ! _clock_smi_priv -L >/dev/null 2>&1; then
        echo "⚠ No passwordless root for nvidia-smi — clocks stay unlocked." >&2
        echo "  Run this first, then re-run:" >&2
        echo "    sudo nvidia-smi -pm 1" >&2
        echo "    sudo nvidia-smi -lgc $CLOCK_SM,$CLOCK_SM" >&2
        [ -n "$CLOCK_MEM" ] && \
        echo "    sudo nvidia-smi -lmc $CLOCK_MEM,$CLOCK_MEM" >&2
        return 1
    fi

    CLOCK_PM_RESTORE="$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null)"
    _clock_apply -pm 1 || echo "⚠ Could not enable persistence mode." >&2

    if ! _clock_apply -lgc "$CLOCK_SM,$CLOCK_SM"; then
        echo "✗ Failed to lock the SM clock to $CLOCK_SM MHz." >&2
        return 1
    fi
    CLOCK_LOCKED=true

    # Some cards expose only the coarse P-state memory clocks, and some drivers
    # refuse -lmc outright. Drop the memory clock from the drift check rather
    # than failing the run over something that was never pinned.
    if [ -n "$CLOCK_MEM" ]; then
        if ! _clock_apply -lmc "$CLOCK_MEM,$CLOCK_MEM"; then
            echo "⚠ Could not lock the memory clock to $CLOCK_MEM MHz; left on the" >&2
            echo "  driver default and excluded from drift checks." >&2
            CLOCK_MEM=""
        fi
    fi

    local now
    now="$(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    echo "✓ Clocks locked: SM=${CLOCK_SM} MHz${CLOCK_MEM:+, MEM=${CLOCK_MEM} MHz}  (now reading ${now})"
    return 0
}

clocks_reset() {
    $CLOCK_LOCKED || return 0
    CLOCK_LOCKED=false
    _clock_apply -rgc && echo "✓ SM clock unlocked" \
        || echo "⚠ Failed to reset the SM clock — run 'sudo nvidia-smi -rgc'" >&2
    _clock_apply -rmc
    case "$CLOCK_PM_RESTORE" in
        Disabled) _clock_smi_priv -pm 0 >/dev/null 2>&1;;
    esac
}

clocks_monitor_start() {
    CLOCK_CSV="$1"
    CLOCK_REPORT_TSV="$(dirname "$1")/clock_stability.tsv"
    : > "$CLOCK_REPORT_TSV"
    # stdbuf keeps the log line-buffered: clocks_check reads it back mid-run,
    # and a block-buffered nvidia-smi could hold a short stage's samples in
    # memory, leaving that stage silently unchecked.
    local smi=(nvidia-smi
        --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,clocks_event_reasons.active
        --format=csv,nounits -lms 1000)
    command -v stdbuf >/dev/null 2>&1 && smi=(stdbuf -oL "${smi[@]}")
    "${smi[@]}" > "$CLOCK_CSV" 2>&1 &
    CLOCK_MONITOR_PID=$!
    # One process for the whole run rather than a per-second nvidia-smi: cheaper,
    # and it cannot fall behind and leave gaps in the record.
    sleep 1
    if ! kill -0 "$CLOCK_MONITOR_PID" 2>/dev/null; then
        echo "⚠ Clock monitor died immediately; drift will not be checked." >&2
        CLOCK_MONITOR_PID=""
        return 1
    fi
    return 0
}

clocks_monitor_stop() {
    [ -n "$CLOCK_MONITOR_PID" ] || return 0
    kill "$CLOCK_MONITOR_PID" 2>/dev/null
    wait "$CLOCK_MONITOR_PID" 2>/dev/null
    CLOCK_MONITOR_PID=""
}

# nvidia-smi stamps its samples in local time as "YYYY/MM/DD HH:MM:SS.mmm",
# which is zero-padded and therefore sorts chronologically as a plain string.
# That is what lets the window filter below be a string comparison -- mawk has
# no mktime(), so parsing these into epoch seconds is not portably available.
clocks_stamp() {
    if [ "${1:-}" = end ]; then
        date +'%Y/%m/%d %H:%M:%S.999'
    else
        date +'%Y/%m/%d %H:%M:%S.000'
    fi
}

# clocks_check <from> <to> <label> <critical>
#
# Verdict for one step's slice of the log. "critical" (true/false) says whether
# drift here invalidates a published number -- timed sweeps yes, accuracy-only
# stages no -- which sets warning versus error.
clocks_check() {
    local from="$1" to="$2" label="$3" critical="${4:-true}"
    [ -n "$CLOCK_CSV" ] && [ -s "$CLOCK_CSV" ] || return 0
    [ -n "$CLOCK_SM" ] || return 0

    local line
    line="$(awk -v from="$from" -v to="$to" -v sm="$CLOCK_SM" -v mem="$CLOCK_MEM" \
                -v tol="$CLOCK_TOL_MHZ" -F',' '
        # nvidia-smi reports the reasons as a 64-bit hex mask. Decode it by hand:
        # mawk has no and()/rshift(), so take the low hex digits into an integer
        # and test bits with integer division.
        function hex2dec(s,   i, n, c, d, v, start) {
            sub(/^[ \t]*0[xX]/, "", s)
            n = length(s); start = (n > 8 ? n - 7 : 1); v = 0
            for (i = start; i <= n; i++) {
                c = tolower(substr(s, i, 1))
                d = index("0123456789abcdef", c) - 1
                if (d < 0) d = 0
                v = v * 16 + d
            }
            return v
        }
        function bit(v, b) { return int(v / b) % 2 }
        function abs(x) { return x < 0 ? -x : x }

        $1 < from || $1 > to { next }
        $2 !~ /^[ \t]*[0-9]+/ { next }                 # header / error text

        {
            reasons = hex2dec($7)
            idle    = bit(reasons, 1)                  # 0x01 GpuIdle
            # 0x02 ApplicationsClocksSetting belongs to the deprecated -ac
            # mechanism and is not asserted by -lgc (driver 595); ignored either
            # way. The bits below override a lock:
            bad = bit(reasons, 4) || bit(reasons, 8) \
               || bit(reasons, 32) || bit(reasons, 64) || bit(reasons, 128)

            n++
            if ($4 + 0 > tmax) tmax = $4 + 0
            if ($5 + 0 > pmax) pmax = $5 + 0

            # A clock drop with no kernel running cannot affect a timing, and
            # the lock restores it before the next one. Busy samples only.
            if (idle) next
            busy++

            dev = abs($2 - sm)
            if (dev > devworst) devworst = dev
            if (dev > tol) drift++
            if (minsm == 0 || $2 + 0 < minsm) minsm = $2 + 0

            if (mem != "" && abs($3 - mem) > tol) memdrift++
            if (bad) throttled++
        }
        END {
            printf "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.0f\t%.1f\n",
                   n, busy, drift, devworst, minsm, memdrift, throttled, tmax, pmax
        }' "$CLOCK_CSV")"

    local n busy drift devworst minsm memdrift throttled tmax pmax
    IFS=$'\t' read -r n busy drift devworst minsm memdrift throttled tmax pmax <<<"$line"

    [ "${n:-0}" -gt 0 ] || return 0

    # Escalate on >CLOCK_DRIFT_PCT of busy samples, and at least 3 of them: a
    # short step has few samples, so one stray reading would otherwise fail it.
    local pct=0 verdict="OK"
    [ "$busy" -gt 0 ] && pct=$(( (drift * 100 + busy - 1) / busy ))
    if [ "$throttled" -gt 0 ] || { [ "$pct" -gt "$CLOCK_DRIFT_PCT" ] && [ "$drift" -ge 3 ]; }; then
        verdict="DRIFT"
    elif [ "$drift" -gt 0 ] || [ "${memdrift:-0}" -gt 0 ]; then
        verdict="BLIP"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$verdict" "$n" "$busy" "$drift" "$devworst" \
        "$minsm" "$memdrift" "$throttled" "${tmax}C/${pmax}W" >> "$CLOCK_REPORT_TSV"

    case "$verdict" in
        OK) ;;
        BLIP)
            echo "  ⚠ clock blip during $label: $drift/$busy busy samples off" \
                 "${CLOCK_SM}MHz (worst ${devworst}MHz, min ${minsm}MHz)" >&2;;
        DRIFT)
            local sev="⚠ WARNING"
            $critical && sev="✗ ERROR"
            echo "  $sev: clocks drifted during $label" >&2
            echo "    $drift/$busy busy samples off ${CLOCK_SM}MHz" \
                 "(${pct}%, worst ${devworst}MHz low, min ${minsm}MHz)" >&2
            [ "${memdrift:-0}" -gt 0 ] && \
                echo "    ${memdrift} samples off the ${CLOCK_MEM}MHz memory clock" >&2
            [ "$throttled" -gt 0 ] && \
                echo "    ${throttled} samples throttled by power/thermal limits" \
                     "(peak ${tmax}C, ${pmax}W)" >&2
            $critical && echo "    → lower the lock and re-run this step" >&2
            ;;
    esac

    # Only a critical step fails the run.
    if [ "$verdict" = DRIFT ] && $critical; then
        return 1
    fi
    return 0
}

# Final table, printed whether or not anything drifted. Returns non-zero when
# there was nothing to report, so the caller can leave out the surrounding rule.
clocks_report() {
    [ -n "$CLOCK_REPORT_TSV" ] && [ -s "$CLOCK_REPORT_TSV" ] || return 1
    echo
    echo "CLOCK STABILITY  (target SM=${CLOCK_SM:-unlocked}${CLOCK_MEM:+ MEM=$CLOCK_MEM}, tol ${CLOCK_TOL_MHZ}MHz)"
    printf '%-26s %-7s %8s %8s %9s %10s\n' "STEP" "STATUS" "SAMPLES" "OFF" "WORST" "PEAK"
    printf '%-26s %-7s %8s %8s %9s %10s\n' "----" "------" "-------" "---" "-----" "----"
    local label verdict n busy drift devworst minsm memdrift throttled peak
    while IFS=$'\t' read -r label verdict n busy drift devworst minsm memdrift throttled peak; do
        printf '%-26s %-7s %8s %8s %9s %10s\n' \
            "$label" "$verdict" "$busy" "$drift" "${devworst}MHz" "$peak"
    done < "$CLOCK_REPORT_TSV"
}
