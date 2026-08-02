#!/bin/bash
set -e
# Backend is an extra axis here, so this one does not use the shared parser.
NMAX=16777216
BACKEND=
while [ $# -gt 0 ]; do
    case "$1" in
        -n|--nmax) NMAX=$2; shift 2;;
        -b|--backend) BACKEND=$2; shift 2;;
        -a|--analysis)
            [ "$2" == "performance" ] || { echo "multi-device supports -a performance only" >&2; exit 1; }
            shift 2;;
        *) echo "Unknown option $1" >&2; exit 1;;
    esac
done
[ -n "$BACKEND" ] || { echo "Usage: $0 --backend <backend> [-n nmax]" >&2; exit 1; }

rm -rf "./data/devices/${BACKEND}"
mkdir -p "./data/devices/${BACKEND}"

a=8
while [ $a -le $NMAX ]
do
    echo "No. of trajectories = $a"
    julia --project="./GPU_ODE_Julia/" ./GPU_ODE_Julia/bench_multi_device.jl $a "$BACKEND"
    a=$((a*4))
done
