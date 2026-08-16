#!/usr/bin/env python3
"""Run the direct Cubie <-> DiffEqGPU GPU-ODE overlap benchmark.

Examples:
  python run_cubie_julia_overlap.py -a numerical -p cubie
  python run_cubie_julia_overlap.py -a performance --from-n 2048
  python run_cubie_julia_overlap.py -a performance -n 32768,134217728 -p julia
  python run_cubie_julia_overlap.py --algorithm kvaerno5 -p cubie

Results land in data/cubie_julia_overlap/<dataset-key>/; figures and the report
land in plots/<dataset-key>/. A run replaces the rows it
regenerates and leaves the rest. Workers record point failures and keep going;
the analyzer runs after the selected workers finish.
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
    parse_ns, phases_for, prune_csv,
)
from problems import problem_names, resolve_problems  # noqa: E402

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
    p.add_argument("-a", "--analysis", choices=ANALYSES + ("all",), default="all",
                   help="Which analysis to run; one not selected keeps its existing rows.")
    p.add_argument("-p", "--package", choices=("all", "cubie", "julia"), default="all")
    p.add_argument("-n", "--nmax", default="16777216",
                   help="Sweep ceiling (8, 32, ... <= n) or a comma list of exact trajectory counts.")
    p.add_argument("--from-n", type=int, default=0,
                   help="Continue the performance analysis at this N; rows below it are kept.")
    p.add_argument("--algorithm", choices=algorithm_names(), default="all",
                   help="Run one algorithm; the others keep their existing rows.")
    p.add_argument("-s", "--problem", choices=["all"] + problem_names(),
                   default="all",
                   help="Run one problem; each gets its own output directory.")
    return p


def main():
    args = parser().parse_args()
    try:
        ns = parse_ns(args.nmax, args.from_n)
    except ValueError:
        parser().error("-n/--nmax takes an integer or a comma list of integers, got '{}'"
                       .format(args.nmax))
    if not ns:
        parser().error("-n/--nmax selects no trajectory count of at least 8")
    if args.from_n and args.analysis != "performance":
        parser().error("--from-n continues the performance analysis; pass -a performance")
    key = dataset_key()
    problems = resolve_problems(args.problem, "cubie")
    if not problems:
        parser().error("no requested problem is in the overlap suite")
    cubie_python = existing_python()
    julia = os.environ.get("JULIA", "julia")
    phases = phases_for(args.analysis)
    packages = ("julia", "cubie") if args.package == "all" else (args.package,)
    status = 0
    for problem in problems:
        status |= run_problem(problem, args, ns, key, packages, cubie_python,
                              julia, phases)
    return status


def run_problem(problem, args, ns, key, packages, cubie_python, julia, phases):
    """One problem: the selected workers, then the analyzer."""
    output = (ROOT / "data" / "cubie_julia_overlap" / key /
              problem["problem"]).resolve()
    shared = ["--output", str(output),
              "--analysis", args.analysis, "--nmax", ",".join(str(n) for n in ns),
              "--from-n", str(args.from_n), "--algorithm", args.algorithm,
              "--problem", problem["problem"]]
    commands = []
    if "julia" in packages:
        commands.append(("julia", [julia, "--startup-file=no", "-t", "auto",
                                   "--project={}".format(ROOT), str(SUITE / "julia_worker.jl")] + shared))
    if "cubie" in packages:
        commands.append(("cubie", [str(cubie_python), str(SUITE / "cubie_worker.py")] + shared))
    commands.append(("analysis", [str(cubie_python), str(SUITE / "analyze.py"),
                                  "--output", str(output), "--key", key,
                                  "--problem", problem["problem"]]))

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
                                fields, phases, args.from_n, args.algorithm, ns)
            if dropped:
                print("Replacing {} row(s) in {}_{}.csv".format(dropped, framework, kind))
        if "numerical" in phases:
            stale = output / "finals" / framework
            if args.algorithm != "all":
                stale = stale / args.algorithm
            shutil.rmtree(stale, ignore_errors=True)

    manifest = {
        "dataset_key": key, "problem": problem["problem"],
        "analysis": args.analysis, "package": args.package,
        "cubie_backend": worker_env["CUBIE_CUDA_BACKEND"],
        "nmax": args.nmax, "performance_ns": ns, "from_n": args.from_n,
        "algorithm": args.algorithm,
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
