# Sets ANALYSIS, NMAX and ALGORITHM in the caller: . "$(dirname "$0")/../parse_args.sh" "$@"
ANALYSIS=performance
NMAX=16777216
ALGORITHM=all
while [ $# -gt 0 ]; do
    case "$1" in
        -a|--analysis)
            [ $# -ge 2 ] || { echo "$1 requires a value" >&2; exit 1; }
            ANALYSIS=$2; shift 2;;
        -n|--nmax)
            [ $# -ge 2 ] || { echo "$1 requires a value" >&2; exit 1; }
            NMAX=$2; shift 2;;
        -g|--algorithm)
            [ $# -ge 2 ] || { echo "$1 requires a value" >&2; exit 1; }
            ALGORITHM=$2; shift 2;;
        *) echo "Unknown option $1" >&2; exit 1;;
    esac
done
case "$ANALYSIS" in
    performance|work-precision) ;;
    *) echo "Unknown analysis '$ANALYSIS' (performance|work-precision)" >&2; exit 1;;
esac
case "$ALGORITHM" in
    all|euler|classical-rk4|tsit5|cash-karp-54) ;;
    *) echo "Unknown algorithm '$ALGORITHM' (all|euler|classical-rk4|tsit5|cash-karp-54)" >&2; exit 1;;
esac
case "$NMAX" in
    ''|*[!0-9]*) echo "-n/--nmax must be a positive integer, got '$NMAX'" >&2; exit 1;;
esac
