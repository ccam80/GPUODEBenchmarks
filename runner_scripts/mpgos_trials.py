"""MPGOS trial-file helper for run_ode_cpp.ps1 and .sh: `context` prints key, source hash, package_version, suite_rev and the watchdog exit code; `builds <trials>` lists binaries; `points <trials>` lists solve trials in run order; `nan <trials> <trial_id> <key> <transfers,...> <reason> [--floor] [--build-s S]` records NaN rows."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import protocol  # noqa: E402
import store  # noqa: E402
import trials as trials_mod  # noqa: E402
from bench_key import dataset_key  # noqa: E402

REPO_ROOT = os.path.dirname(HERE)
MPGOS_DIR = os.path.join(REPO_ROOT, "GPU_ODE_MPGOS")
PROBLEMS_DIR = os.path.join(MPGOS_DIR, "problems")
PROTOCOL_HEADER = os.path.join(MPGOS_DIR, "protocol.h")

# Algorithm to MPGOS solver.
SOLVERS = {"classical-rk4": "RK4", "cash-karp-54": "RKCK45"}
PRECISION_TYPES = {"float32": "float", "float64": "double"}
HASH_HEX = 12

BUILD_COLUMNS = ("problem", "solver", "nt", "sd", "precision", "cold", "leg")
POINT_COLUMNS = ("trial_id", "leg", "ordinal", "problem", "solver", "nt", "sd", "precision",
                 "transfers", "finals", "reason")


def source_files():
    """The source files, sorted."""
    paths = [os.path.join(MPGOS_DIR, name) for name in ("Bench.cu", "grid.cuh", "trial.cuh",
                                                        "protocol.h", "makefile")]
    for folder in ("problems", "SourceCodes"):
        root = os.path.join(MPGOS_DIR, folder)
        for dirpath, _, names in os.walk(root):
            paths += [os.path.join(dirpath, name) for name in names]
    return sorted(p for p in paths if os.path.isfile(p))


def source_hash():
    """The first 12 hex of the sha256 over the source files in sorted order."""
    digest = hashlib.sha256()
    for path in source_files():
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()[:HASH_HEX]


def nvcc_release():
    """The `release X.Y` of nvcc on PATH, or 'unknown'."""
    try:
        out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    match = re.search(r"release\s+(\d+\.\d+)", out.stdout + out.stderr)
    return match.group(1) if match else "unknown"


def package_version(src_hash=None):
    """<source hash>+nvcc<release>."""
    return "{0}+nvcc{1}".format(src_hash or source_hash(), nvcc_release())


def header_states(problem):
    """The PROBLEM_SD a problem header declares; None when the package has no header for the problem."""
    path = os.path.join(PROBLEMS_DIR, problem + ".cuh")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        match = re.search(r"^#define PROBLEM_SD (\d+)", handle.read(), flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def states_param(trial):
    """The states count system_params carries, or None."""
    params = json.loads(trial["system_params"]) if trial["system_params"] else {}
    return int(params["states"]) if "states" in params else None


def states_of(trial):
    """The state count the trial's binary has: system_params, else the header default."""
    given = states_param(trial)
    return given if given is not None else header_states(trial["problem"])


def unrunnable(trial):
    """Why the package cannot run a trial, or '' when it can."""
    if header_states(trial["problem"]) is None:
        return "error: ValueError: cpp has no problem header for " + trial["problem"]
    if trial["algorithm"] not in SOLVERS:
        return "error: ValueError: cpp has no solver for " + trial["algorithm"]
    if trial["precision"] not in PRECISION_TYPES:
        return "error: ValueError: cpp builds float32 or float64, not " + trial["precision"]
    return ""


def build_key(trial):
    """(problem, solver, nt, sd, precision) of the binary a trial runs on; sd is '-' when the header default stands."""
    given = states_param(trial)
    return (trial["problem"], SOLVERS[trial["algorithm"]], int(trial["n"]),
            "-" if given is None else str(given), trial["precision"])


def builds(trial_list):
    """One row per binary in first appearance, cold when a cold warm line asks for it, with that leg."""
    rows = {}
    for trial in trial_list:
        if trial["kind"] not in ("warm", "solve") or unrunnable(trial):
            continue
        key = build_key(trial)
        row = rows.setdefault(key, {"cold": False, "leg": trial["leg"]})
        if trial["kind"] == "warm" and trial["cold"] and not row["cold"]:
            row["cold"] = True
            row["leg"] = trial["leg"]
    out = []
    for (problem, solver, nt, sd, precision), row in rows.items():
        out.append({"problem": problem, "solver": solver, "nt": nt, "sd": sd,
                    "precision": precision, "cold": row["cold"], "leg": row["leg"]})
    return out


def points(trial_list):
    """The solve trials in file order with build key, transfers, finals and the reason they cannot run ('' when they can)."""
    out = []
    for trial in trial_list:
        if trial["kind"] != "solve":
            continue
        reason = unrunnable(trial)
        solver = SOLVERS.get(trial["algorithm"], "-")
        given = states_param(trial)
        out.append({"trial_id": trial["trial_id"], "leg": trial["leg"], "ordinal": trial["ordinal"],
                    "problem": trial["problem"], "solver": solver, "nt": int(trial["n"]),
                    "sd": "-" if given is None else str(given), "precision": trial["precision"],
                    "transfers": ",".join(trial["transfers"]), "finals": trial["finals"],
                    "reason": reason})
    return out


def nan_rows(trial_list, trial_id, key, transfers, reason, build_s=None, src_hash=None,
             suite_rev=None):
    """NaN rows for the named solve trial's transfers with a reason; the caller records them."""
    matched = [t for t in trial_list if t["kind"] == "solve" and t["trial_id"] == trial_id]
    if not matched:
        raise ValueError("no solve trial " + trial_id)
    trial = matched[0]
    states = states_of(trial)
    if states is None:
        raise ValueError("cpp has no problem header for " + trial["problem"])
    spec = {field: trial[field] for field in store.TRIAL_FIELDS}
    version = package_version(src_hash)
    rev = suite_rev if suite_rev is not None else store.suite_rev(REPO_ROOT)
    return [dict(spec, transfers=t, key=key, states=states, reason=reason,
                 build_s=float("nan") if build_s is None else float(build_s),
                 package_version=version, suite_rev=rev)
            for t in transfers]


def _cell(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _print_table(rows, columns):
    for row in rows:
        print("\t".join(_cell(row[c]) for c in columns))


def context():
    """The scripts' constants; refreshes protocol.h."""
    protocol.write_cxx_header(PROTOCOL_HEADER)
    src_hash = source_hash()
    return {"key": dataset_key(), "source_hash": src_hash,
            "package_version": package_version(src_hash), "suite_rev": store.suite_rev(REPO_ROOT),
            "watchdog_exit": protocol.WATCHDOG_EXIT_CODE}


def main(argv):
    parser = argparse.ArgumentParser(prog="mpgos_trials.py", description=__doc__)
    parser.add_argument("--root", default="data")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("context")
    commands.add_parser("builds").add_argument("trials")
    commands.add_parser("points").add_argument("trials")
    nan = commands.add_parser("nan")
    nan.add_argument("trials")
    nan.add_argument("trial_id")
    nan.add_argument("key")
    nan.add_argument("transfers")
    nan.add_argument("reason")
    nan.add_argument("--floor", action="store_true")
    nan.add_argument("--build-s", default=None)
    args = parser.parse_args(argv)
    if args.command == "context":
        for name, value in context().items():
            print("{0}={1}".format(name, value))
        return 0
    trial_list = trials_mod.read_jsonl(args.trials)
    if args.command == "builds":
        _print_table(builds(trial_list), BUILD_COLUMNS)
        return 0
    if args.command == "points":
        _print_table(points(trial_list), POINT_COLUMNS)
        return 0
    transfers = [t for t in args.transfers.split(",") if t]
    rows = nan_rows(trial_list, args.trial_id, args.key, transfers, args.reason, args.build_s)
    store.Store(args.root).record_batch(rows, floor=args.floor)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
