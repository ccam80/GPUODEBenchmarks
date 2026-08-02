#!/usr/bin/env python3
"""Run the direct Cubie <-> DiffEqGPU GPU-ODE overlap benchmark.

Examples:
  python run_cubie_julia_overlap.py --profile full
  python run_cubie_julia_overlap.py -a numerical -p cubie
  python run_cubie_julia_overlap.py -a performance --from-n 2048
  python run_cubie_julia_overlap.py --algorithm kvaerno5 -p cubie

Results land in data/cubie_julia_overlap/<dataset-key>/. A run replaces the
rows it regenerates and leaves the rest. Workers record point failures and
keep going; the analyzer runs after the selected workers finish.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / "runner_scripts" / "cubie_julia_overlap"
sys.path.insert(0, str(ROOT / "runner_scripts"))
sys.path.insert(0, str(SUITE))
from bench_key import dataset_key  # noqa: E402 - repository helper bootstrap
from common import (  # noqa: E402 - suite helper bootstrap
    ANALYSES, FAILURE_FIELDS, METRIC_FIELDS, TIMING_FIELDS, algorithm_names,
    phases_for, prune_csv,
)

CSV_KINDS = (("timings", TIMING_FIELDS), ("metrics", METRIC_FIELDS),
             ("failures", FAILURE_FIELDS))


def existing_python():
    """The cubie interpreter: the repo venv when present, else this one."""
    candidates = [
        ROOT / "GPU_ODE_CUBIE" / "venv" / "Scripts" / "python.exe",
        ROOT / "GPU_ODE_CUBIE" / "venv" / "bin" / "python",
        Path(sys.executable),
    ]
    return next((p for p in candidates if p.exists()), Path(sys.executable))


def parser():
    p = argparse.ArgumentParser(
        description="Cubie versus DiffEqGPU GPU ODE comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--profile", choices=("smoke", "full"), default="smoke",
                   help="Smoke uses every algorithm with reduced N/grids; full uses the published protocol.")
    p.add_argument("-a", "--analysis", choices=ANALYSES + ("all",), default="all",
                   help="Which analysis to run; one not selected keeps its existing rows.")
    p.add_argument("-p", "--package", choices=("all", "cubie", "julia"), default="all")
    p.add_argument("-n", "--nmax", type=int, default=16_777_216,
                   help="Largest performance N; values are 8*4^k not exceeding this value.")
    p.add_argument("--from-n", type=int, default=0,
                   help="Continue the performance analysis at this N; rows below it are kept.")
    p.add_argument("--algorithm", choices=algorithm_names(), default="all",
                   help="Run one algorithm; the others keep their existing rows.")
    return p


def main():
    args = parser().parse_args()
    if args.nmax < 8:
        parser().error("--nmax must be at least 8")
    if args.from_n and args.analysis != "performance":
        parser().error("--from-n continues the performance analysis; pass -a performance")
    key = dataset_key()
    output = (ROOT / "data" / "cubie_julia_overlap" / key).resolve()
    cubie_python = existing_python()
    julia = os.environ.get("JULIA", "julia")
    phases = phases_for(args.analysis)
    packages = ("julia", "cubie") if args.package == "all" else (args.package,)
    shared = ["--output", str(output), "--profile", args.profile,
              "--analysis", args.analysis, "--nmax", str(args.nmax),
              "--from-n", str(args.from_n), "--algorithm", args.algorithm]
    commands = []
    if "julia" in packages:
        commands.append(("julia", [julia, "--startup-file=no", "-t", "auto",
                                   "--project={}".format(ROOT), str(SUITE / "julia_worker.jl")] + shared))
    if "cubie" in packages:
        commands.append(("cubie", [str(cubie_python), str(SUITE / "cubie_worker.py")] + shared))
    commands.append(("analysis", [str(cubie_python), str(SUITE / "analyze.py"), "--output", str(output)]))

    print("Output: {}".format(output))
    for label, command in commands:
        print("{}: {}".format(label, subprocess.list2cmdline(command)))

    # Cubie resolves its CUDA backend at import time from this variable.
    worker_env = dict(os.environ)
    worker_env.setdefault("CUBIE_CUDA_BACKEND", "numba-cuda")
    print("Cubie backend: {}".format(worker_env["CUBIE_CUDA_BACKEND"]))

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SUITE / "diffeqgpu_ode_inventory.csv", output / "diffeqgpu_ode_inventory.csv")
    shutil.copy2(SUITE / "algorithms.csv", output / "overlap_algorithms.csv")

    # Clear the rows this run replaces; the workers only append.
    for framework in packages:
        for kind, fields in CSV_KINDS:
            dropped = prune_csv(output / "{}_{}.csv".format(framework, kind),
                                fields, phases, args.from_n, args.algorithm)
            if dropped:
                print("Replacing {} row(s) in {}_{}.csv".format(dropped, framework, kind))
        if "numerical" in phases:
            stale = output / "finals" / framework
            if args.algorithm != "all":
                stale = stale / args.algorithm
            shutil.rmtree(stale, ignore_errors=True)

    manifest = {
        "dataset_key": key, "profile": args.profile,
        "analysis": args.analysis, "package": args.package,
        "cubie_backend": worker_env["CUBIE_CUDA_BACKEND"],
        "nmax": args.nmax, "from_n": args.from_n, "algorithm": args.algorithm,
        "commands": [c for _, c in commands],
        "started_utc": datetime.now(timezone.utc).isoformat(), "results": {},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    process_failures = 0
    for label, command in commands:
        print("\n=== {} ===".format(label), flush=True)
        try:
            completed = subprocess.run(command, cwd=str(ROOT), check=False,
                                       env=worker_env)
            code = completed.returncode
        except OSError as exc:
            code = 127
            (output / "{}_process_failure.txt".format(label)).write_text(
                "{}: {}\n".format(type(exc).__name__, exc), encoding="utf-8")
        manifest["results"][label] = code
        if code != 0:
            process_failures += 1
            print("{} process failed with exit code {}; continuing.".format(label, code), flush=True)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nRun artifacts: {}".format(output))
    return 1 if process_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
