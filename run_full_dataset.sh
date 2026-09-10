#!/bin/bash
# Forwards to bench.py, the benchmark entry point; every flag is the same.
exec python3 "$(dirname "$0")/bench.py" "$@"
