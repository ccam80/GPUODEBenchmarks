# Sets ANALYSIS, NMAX, NLIST, ALGORITHM and PROBLEM in the caller; -n takes a sweep ceiling or a comma list, -s a problem name, comma list, or all.
ANALYSIS=performance
NMAX=16777216
ALGORITHM=all
PROBLEM=all
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
        -s|--problem)
            [ $# -ge 2 ] || { echo "$1 requires a value" >&2; exit 1; }
            PROBLEM=$2; shift 2;;
        *) echo "Unknown option $1" >&2; exit 1;;
    esac
done
[ -n "$PROBLEM" ] || { echo "-s/--problem requires a value" >&2; exit 1; }
case "$ANALYSIS" in
    performance|work-precision|states) ;;
    *) echo "Unknown analysis '$ANALYSIS' (performance|work-precision|states)" >&2; exit 1;;
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
