# Sets ANALYSIS, NMAX, NLIST and ALGORITHM in the caller: . "$(dirname "$0")/../parse_args.sh" "$@"
# -n takes a single sweep ceiling (runs 8, 32, ... <= n) or a comma list of exact
# trajectory counts (runs exactly those). NLIST holds the resulting space-separated
# counts; NMAX is the largest of them.
ANALYSIS=performance
NMAX=16777216
ALGORITHM=all
NLIST=
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
case ",$NMAX," in
    *[!0-9,]*|*,,*)
        echo "-n/--nmax must be a positive integer or a comma list of them, got '$NMAX'" >&2
        exit 1;;
esac
case "$NMAX" in
    *,*)
        NLIST=${NMAX//,/ }
        NMAX=0
        for n in $NLIST; do
            if [ "$n" -gt "$NMAX" ]; then NMAX=$n; fi
        done
        ;;
    *)
        n=8
        while [ "$n" -le "$NMAX" ]; do
            NLIST="$NLIST $n"
            n=$((n * 4))
        done
        ;;
esac
