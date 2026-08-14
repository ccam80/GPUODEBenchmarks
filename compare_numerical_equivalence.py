#!/usr/bin/env python3
"""Compare cubie against DifferentialEquations.jl per problem, algorithm and dt; outputs land in plots/<group>/<problem>/."""

import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "runner_scripts"))
sys.path.insert(0, os.path.join(_HERE, "runner_scripts", "numerical_equivalence"))
from bench_key import group_dir  # noqa: E402
from problems import problem_names, resolve_problems  # noqa: E402
from ne_common import (TOLS_NE, dts_ne, load_algorithms, load_golden_ne,
                       ensemble_error_masked, julia_ne_file, cubie_ne_file,
                       julia_ne_adaptive_file, cubie_ne_adaptive_file,
                       read_ne_csv_masked, read_ne_adaptive_csv_masked,
                       CUBIE_NE_DIR)

# Windows consoles default to a legacy codepage (cp1252) that cannot encode
# the glyphs printed below; force UTF-8 where supported.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Roundoff floor relative to the golden scale, drawn on the plots so the
# region where error is float32 noise rather than truncation is visible.
FLOOR_REL = 4e-6



def discover_keys():
    """Dataset keys present in the cubie ne output directory."""
    keys = set()
    if not os.path.isdir(CUBIE_NE_DIR):
        return keys
    for name in os.listdir(CUBIE_NE_DIR):
        if os.path.isdir(os.path.join(CUBIE_NE_DIR, name)):
            keys.add(name)
    return keys


def _ratio(err_c, err_j):
    if err_c is None or err_j in (None, 0.0):
        return None
    if not (np.isfinite(err_c) and np.isfinite(err_j)):
        return None
    return err_c / err_j


def analyse_algorithm(row, key, golden_states, problem):
    """Per-dt errors for one algorithm's fixed-step sweep."""
    alias = row["cubie_alias"]
    jfile = julia_ne_file(alias, problem)
    cfile = cubie_ne_file(alias, key, problem)
    julia = read_ne_csv_masked(jfile) if os.path.isfile(jfile) else None
    cubie = read_ne_csv_masked(cfile) if os.path.isfile(cfile) else None

    # Errors use the trajectories BOTH stacks solved (julia carries a
    # per-trajectory retcode flag; cubie signals a failed solve with NaN).
    # Per-side non-converged counts are reported alongside.
    points = []
    for dt in dts_ne(problem):
        fc = cubie.get(dt) if cubie else None
        fj = julia.get(dt) if julia else None
        c_arr, c_conv = fc if fc is not None else (None, None)
        j_arr, j_conv = fj if fj is not None else (None, None)
        err_c = err_j = None
        mutual_n = None
        nc_c = int((~c_conv).sum()) if c_arr is not None else None
        nc_j = int((~j_conv).sum()) if j_arr is not None else None
        if c_arr is not None and j_arr is not None:
            mutual = c_conv & j_conv
            mutual_n = int(mutual.sum())
            err_c = ensemble_error_masked(c_arr, golden_states, mutual)
            err_j = ensemble_error_masked(j_arr, golden_states, mutual)
        elif c_arr is not None:
            err_c = ensemble_error_masked(c_arr, golden_states, c_conv)
        elif j_arr is not None:
            err_j = ensemble_error_masked(j_arr, golden_states, j_conv)
        points.append({
            "dt": dt, "err_cubie": err_c, "err_julia": err_j,
            "ratio": _ratio(err_c, err_j), "nonconv_cubie": nc_c,
            "nonconv_julia": nc_j, "mutual_n": mutual_n,
        })

    return {"row": row, "points": points}


def analyse_adaptive(row, key, golden_states, problem):
    """Per-tolerance errors for one algorithm's adaptive tiers."""
    alias = row["cubie_alias"]
    jfile = julia_ne_adaptive_file(alias, problem)
    julia = read_ne_adaptive_csv_masked(jfile) if os.path.isfile(jfile) else None
    tiers = {}
    for tier in ("default", "matched"):
        cfile = cubie_ne_adaptive_file(alias, tier, key, problem)
        tiers[tier] = (read_ne_adaptive_csv_masked(cfile)
                       if os.path.isfile(cfile) else None)
    if not any(tiers.values()):
        return None

    points = []
    for tol in TOLS_NE:
        j = julia.get(tol) if julia else None
        dflt = tiers["default"].get(tol) if tiers["default"] else None
        mtch = tiers["matched"].get(tol) if tiers["matched"] else None

        masks = [blk[1] for blk in (j, dflt, mtch) if blk is not None]
        mutual = masks[0].copy() if masks else None
        for m in masks[1:]:
            mutual &= m

        err_j = (ensemble_error_masked(j[0], golden_states, mutual)
                 if j is not None else None)
        entry = {
            "tol": tol, "err_julia": err_j,
            "julia_steps": (float(np.nanmedian(j[2])) if j is not None
                            and j[2] is not None else None),
            "nonconv_julia": int((~j[1]).sum()) if j is not None else None,
            "nonconv_default": (int((~dflt[1]).sum())
                                if dflt is not None else None),
            "nonconv_matched": (int((~mtch[1]).sum())
                                if mtch is not None else None),
            "mutual_n": int(mutual.sum()) if mutual is not None else None,
        }
        for tier, data in (("default", dflt), ("matched", mtch)):
            err = (ensemble_error_masked(data[0], golden_states, mutual)
                   if data is not None else None)
            entry["err_" + tier] = err
            entry["ratio_" + tier] = _ratio(err, err_j)
        points.append(entry)

    return {"row": row, "points": points}


FIXED_COLUMNS = ["algorithm", "family", "order", "dt", "err_cubie",
                 "err_julia", "ratio", "nonconv_cubie", "nonconv_julia",
                 "mutual_n"]

ADAPTIVE_COLUMNS = ["algorithm", "family", "order", "tol", "err_cubie_default",
                    "err_cubie_matched", "err_julia", "ratio_default",
                    "ratio_matched", "julia_steps", "nonconv_default",
                    "nonconv_matched", "nonconv_julia", "mutual_n"]


def write_fixed_csv(results, outfile):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIXED_COLUMNS)
        for res in results:
            row = res["row"]
            for p in res["points"]:
                writer.writerow([
                    row["cubie_alias"], row["family"], row["order"], p["dt"],
                    p["err_cubie"], p["err_julia"], p["ratio"],
                    p["nonconv_cubie"], p["nonconv_julia"], p["mutual_n"]])


def write_adaptive_csv(results, outfile):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(ADAPTIVE_COLUMNS)
        for res in results:
            row = res["row"]
            for p in res["points"]:
                writer.writerow([
                    row["cubie_alias"], row["family"], row["order"], p["tol"],
                    p["err_default"], p["err_matched"], p["err_julia"],
                    p["ratio_default"], p["ratio_matched"], p["julia_steps"],
                    p["nonconv_default"], p["nonconv_matched"],
                    p["nonconv_julia"], p["mutual_n"]])


def _grid(n):
    import matplotlib.pyplot as plt
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.0 * ncols, 3.2 * nrows),
                             sharex=True, sharey=True)
    return fig, np.atleast_2d(axes), nrows, ncols


def write_plot(key, results, scale, outfile, problem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes, nrows, ncols = _grid(len(results))
    for idx, res in enumerate(results):
        ax = axes[idx // ncols][idx % ncols]
        row = res["row"]
        pts = res["points"]
        dts_c = [p["dt"] for p in pts if p["err_cubie"] is not None]
        errs_c = [p["err_cubie"] for p in pts if p["err_cubie"] is not None]
        dts_j = [p["dt"] for p in pts if p["err_julia"] is not None]
        errs_j = [p["err_julia"] for p in pts if p["err_julia"] is not None]
        if dts_c:
            ax.loglog(dts_c, errs_c, "o-", color="tab:blue", label="cubie")
        if dts_j:
            ax.loglog(dts_j, errs_j, "x--", color="tab:red", label="DiffEq.jl")
        ax.axhline(FLOOR_REL * scale, color="gray", linewidth=0.5,
                   linestyle="-.")
        ax.set_title(row["cubie_alias"], fontsize=9)
        ax.grid(True, which="both", alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7)
    for idx in range(len(results), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_axis_off()
    # Clamp the shared y-window: divergent coarse-dt points reach 1e30+ and
    # would squash the meaningful error range into a sliver.
    axes[0][0].set_ylim(FLOOR_REL * scale / 30.0, 100.0 * scale)
    for ax in axes[-1]:
        ax.set_xlabel("dt")
    for r in range(nrows):
        axes[r][0].set_ylabel("ensemble l2 error")
    fig.suptitle("Numerical equivalence, Float32 fixed-step {0} ensemble "
                 "({1})".format(problem["display"], key))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def write_adaptive_plot(key, adaptive_results, scale, outfile, problem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes, nrows, ncols = _grid(len(adaptive_results))
    for idx, res in enumerate(adaptive_results):
        ax = axes[idx // ncols][idx % ncols]
        for label, field, style, color in (
                ("cubie default", "err_default", "o-", "tab:blue"),
                ("cubie matched", "err_matched", "s-", "tab:green"),
                ("DiffEq.jl", "err_julia", "x--", "tab:red")):
            xs = [e["tol"] for e in res["points"] if e[field] is not None]
            ys = [e[field] for e in res["points"] if e[field] is not None]
            if xs:
                ax.loglog(xs, ys, style, color=color, label=label,
                          markersize=4)
        tols = [e["tol"] for e in res["points"]]
        ax.loglog(tols, tols, ":", color="gray", linewidth=1, label="err = tol")
        ax.axhline(FLOOR_REL * scale, color="gray", linewidth=0.5,
                   linestyle="-.")
        ax.set_title(res["row"]["cubie_alias"], fontsize=9)
        ax.grid(True, which="both", alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7)
    for idx in range(len(adaptive_results), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("tolerance (atol = rtol)")
    for r in range(nrows):
        axes[r][0].set_ylabel("ensemble l2 error")
    fig.suptitle("Adaptive numerical equivalence, Float32 {0} ensemble "
                 "({1})".format(problem["display"], key))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def compare_problem(problem, algorithms, keys):
    """Write the report and plots for one problem, once per dataset key."""
    _, golden_states = load_golden_ne(problem)
    scale = float(np.sqrt(np.mean(golden_states ** 2)))

    for key in sorted(keys):
        results = [analyse_algorithm(row, key, golden_states, problem)
                   for row in algorithms]
        adaptive_results = [
            res for res in (analyse_adaptive(row, key, golden_states, problem)
                            for row in algorithms)
            if res is not None]
        outdir = group_dir(key, problem)

        fixed_csv = os.path.join(outdir, "numerical_equivalence_fixed.csv")
        plot = os.path.join(outdir, "numerical_equivalence.png")
        write_fixed_csv(results, fixed_csv)
        write_plot(key, results, scale, plot, problem)
        print("[{0}/{1}] {2} algorithms -> {3}, {4}".format(
            key, problem.name, len(results), fixed_csv, plot))

        if adaptive_results:
            adaptive_csv = os.path.join(
                outdir, "numerical_equivalence_adaptive.csv")
            aplot = os.path.join(outdir, "numerical_equivalence_adaptive.png")
            write_adaptive_csv(adaptive_results, adaptive_csv)
            write_adaptive_plot(key, adaptive_results, scale, aplot, problem)
            print("[{0}/{1}] adaptive: {2} algorithms -> {3}, {4}".format(
                key, problem.name, len(adaptive_results), adaptive_csv, aplot))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["all"] + problem_names(),
                        default="all")
    args = parser.parse_args()

    algorithms = load_algorithms()
    keys = discover_keys()
    if not keys:
        print("No cubie ne outputs found in {0}; run "
              "GPU_ODE_CUBIE/numerical_equivalence.py first."
              .format(CUBIE_NE_DIR))
        return 1

    for problem in resolve_problems(args.problem, "cubie"):
        compare_problem(problem, algorithms, keys)
    return 0


if __name__ == "__main__":
    sys.exit(main())
