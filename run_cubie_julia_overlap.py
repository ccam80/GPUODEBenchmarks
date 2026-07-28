#!/usr/bin/env python3
"""Run the exhaustive direct Cubie ↔ DiffEqGPU GPU-ODE overlap benchmark.

Examples:
  python run_cubie_julia_overlap.py --profile smoke
  python run_cubie_julia_overlap.py --profile full --nmax 16777216
  python run_cubie_julia_overlap.py --profile full --phase work_precision --work-repeats 20

The launcher is platform-neutral and intentionally uses subprocess argument
lists (no shell scripts).  Framework workers record point failures and keep
going; the analyzer always runs after the selected workers finish.
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
from bench_key import dataset_key  # noqa: E402 - repository helper bootstrap


def existing_python(explicit=None):
    if explicit:
        return Path(explicit).resolve()
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
    p.add_argument("--phase", choices=("performance", "numerical", "work_precision", "all"), default="all")
    p.add_argument("--framework", choices=("both", "cubie", "julia"), default="both")
    p.add_argument("--nmax", type=int, default=16_777_216,
                   help="Largest performance N; values are 8*4^k not exceeding this value.")
    p.add_argument("--performance-repeats", type=int, default=20)
    p.add_argument("--work-repeats", type=int, default=20)
    p.add_argument("--fixed-dt", type=float, default=2.0 ** -10)
    p.add_argument("--adaptive-tol", type=float, default=1.0e-8)
    p.add_argument("--dataset-key", help="Override the automatic <os>_<gpu> key.")
    p.add_argument("--run-id", help="Output subdirectory; default is a UTC timestamp plus profile.")
    p.add_argument("--output-root", type=Path, default=ROOT / "data" / "cubie_julia_overlap")
    p.add_argument("--cubie-python", help="Python executable containing Cubie.")
    p.add_argument("--analysis-python", help="Python executable containing NumPy and matplotlib.")
    p.add_argument("--julia", default=os.environ.get("JULIA", "julia"))
    p.add_argument("--skip-analyze", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print commands without creating outputs or running workers.")
    return p


def main():
    args = parser().parse_args()
    if args.nmax < 8:
        parser().error("--nmax must be at least 8")
    if args.performance_repeats < 1 or args.work_repeats < 1:
        parser().error("repeat counts must be positive")
    key = args.dataset_key or dataset_key()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_" + args.profile)
    output = (args.output_root / key / run_id).resolve()
    cubie_python = existing_python(args.cubie_python)
    analysis_python = Path(args.analysis_python).resolve() if args.analysis_python else cubie_python
    shared = ["--output", str(output), "--profile", args.profile, "--phase", args.phase,
              "--nmax", str(args.nmax), "--performance-repeats", str(args.performance_repeats),
              "--work-repeats", str(args.work_repeats), "--fixed-dt", repr(args.fixed_dt),
              "--adaptive-tol", repr(args.adaptive_tol), "--reset"]
    commands = []
    if args.framework in ("both", "julia"):
        commands.append(("julia", [args.julia, "--startup-file=no", "-t", "auto",
                                   "--project={}".format(ROOT), str(SUITE / "julia_worker.jl")] + shared))
    if args.framework in ("both", "cubie"):
        commands.append(("cubie", [str(cubie_python), str(SUITE / "cubie_worker.py")] + shared))
    if not args.skip_analyze:
        commands.append(("analysis", [str(analysis_python), str(SUITE / "analyze.py"), "--output", str(output)]))

    print("Output: {}".format(output))
    for label, command in commands:
        print("{}: {}".format(label, subprocess.list2cmdline(command)))
    if args.dry_run:
        return 0

    # The shared cubie venv carries both CUDA backends and resolves one at
    # import time; state which so the run is reproducible. Respect an explicit
    # CUBIE_CUDA_BACKEND from the caller.
    worker_env = dict(os.environ)
    worker_env.setdefault("CUBIE_CUDA_BACKEND", "numba-cuda")
    print("Cubie backend: {}".format(worker_env["CUBIE_CUDA_BACKEND"]))

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SUITE / "diffeqgpu_ode_inventory.csv", output / "diffeqgpu_ode_inventory.csv")
    shutil.copy2(SUITE / "algorithms.csv", output / "overlap_algorithms.csv")
    manifest = {
        "dataset_key": key, "run_id": run_id, "profile": args.profile,
        "phase": args.phase, "framework": args.framework,
        "cubie_backend": worker_env["CUBIE_CUDA_BACKEND"],
        "nmax": args.nmax, "performance_repeats": args.performance_repeats,
        "work_repeats": args.work_repeats, "fixed_dt": args.fixed_dt,
        "adaptive_tol": args.adaptive_tol, "commands": [c for _, c in commands],
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
