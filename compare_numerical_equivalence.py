#!/usr/bin/env python3
"""Compare cubie against DifferentialEquations.jl per algorithm and dt.

Consumes the numerical-equivalence (ne) sweep outputs (see
runner_scripts/numerical_equivalence/): Float32 fixed-step final states of
the Lorenz ensemble for every mutually supported algorithm, from cubie
(per-machine, data/numerical_equivalence/cubie/) and from raw
DifferentialEquations.jl (machine-independent CPU reference,
data/numerical_equivalence/julia/). For each algorithm the error against the
Float64 golden reference is tabulated by dt, the observed convergence order
is estimated, and the two implementations are checked for equivalence.

Outputs, per dataset key found in the cubie directory:
  numerical_equivalence_<os>_<gpu>.md      (report, repo root)
  plots/numerical_equivalence_<os>_<gpu>.png (error-vs-dt grid)

Run from the repo root (inside the GPU_ODE_CUBIE venv):
    python compare_numerical_equivalence.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "runner_scripts",
    "numerical_equivalence"))
from ne_common import (DTS_NE, TOLS_NE, load_algorithms, load_golden_ne,
                       ensemble_error, ensemble_error_masked, julia_ne_file,
                       cubie_ne_file, julia_ne_adaptive_file,
                       cubie_ne_adaptive_file, read_ne_csv, read_ne_adaptive_csv,
                       read_ne_csv_masked, read_ne_adaptive_csv_masked,
                       CUBIE_NE_DIR)

# Windows consoles default to a legacy codepage (cp1252) that cannot encode
# the glyphs printed below; force UTF-8 where supported.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Error bounds (relative to the golden scale). Below FLOOR_REL the error is
# float32 roundoff, not truncation, so nothing can be said. Above ORDER_CAP_REL
# the solve has left the asymptotic regime, so the observed-ORDER estimate is
# meaningless (order estimation only). The equivalence check has NO upper cap:
# once genuinely non-converged trajectories are removed (mutual-convergence
# masking), any point where both stacks converged but still disagree — however
# large the error — is a real mismatch and must count toward the verdict.
FLOOR_REL = 4e-6
ORDER_CAP_REL = 3e-2

# Equivalence: two implementations of the identical tableau differ only by
# roundoff, so their mutual rms distance must stay well below the truncation
# error (factor EQ_FRACTION) once above the roundoff floor allowance. Every
# algorithm is judged the same way: either its trajectories match Julia's to
# within EQ_FRACTION of the truncation error, or they do not. There is no
# softer "same order / comparable magnitude" path for different-tableau pairs
# (e.g. sdirk_2_2, whose Julia SDIRK22 aliases to Trapezoid) — a genuine
# tableau difference is a real mismatch, not a pass.
EQ_FRACTION = 0.05
EQ_FLOOR_MULT = 3.0

# Adaptive tiers are compared on SIGNED accuracy, not trajectory distance: at
# each tolerance take the ratio of cubie's error to Julia's error against the
# golden (both over the mutually-converged trajectories). The mutual rms
# distance is unsigned and penalises cubie even when it is *more* accurate than
# Julia — a single float32 accept/reject flip decouples the dt sequences, yet
# cubie can still land closer to golden — so the distance is reported in the
# tables but never drives the verdict. A tier passes (cubie tracks or beats
# Julia) when cubie's error is at most ADAPTIVE_TOL times Julia's at every
# in-range tolerance; being more accurate always passes. The same rule and
# tolerance apply to the matched tier (Julia's resolved controller) and the
# default tier (cubie's own controller) — there is no separate default-tier
# band.
ADAPTIVE_TOL = 1.1


def observed_orders(errs_by_dt, floor, cap):
    """Median log2 error ratio between successive dt halvings in-region."""
    orders = []
    for k in range(len(DTS_NE) - 1):
        d1, d2 = DTS_NE[k], DTS_NE[k + 1]  # d2 = d1 / 2
        if d1 in errs_by_dt and d2 in errs_by_dt:
            e1, e2 = errs_by_dt[d1], errs_by_dt[d2]
            if (np.isfinite(e1) and np.isfinite(e2)
                    and floor < e1 < cap and floor < e2 < cap and e2 > 0):
                orders.append(np.log2(e1 / e2))
    return float(np.median(orders)) if orders else float("nan")


def discover_keys(aliases):
    """Dataset keys present in the cubie ne output directory."""
    keys = set()
    if not os.path.isdir(CUBIE_NE_DIR):
        return keys
    for fname in os.listdir(CUBIE_NE_DIR):
        if not fname.endswith(".csv"):
            continue
        stem = fname[:-4]
        # Alias may itself contain '_', so match against the known aliases.
        for alias in aliases:
            if not stem.startswith(alias + "_"):
                continue
            rest = stem[len(alias) + 1:]
            # Adaptive outputs carry an extra "adaptive_<tier>_" infix.
            for prefix in ("adaptive_default_", "adaptive_matched_"):
                if rest.startswith(prefix):
                    rest = rest[len(prefix):]
                    break
            keys.add(rest)
    return keys


def analyse_algorithm(row, key, golden_states, scale):
    """Collect the per-dt metric table and verdict for one algorithm."""
    alias = row["cubie_alias"]
    jfile = julia_ne_file(alias)
    cfile = cubie_ne_file(alias, key)
    julia = read_ne_csv_masked(jfile) if os.path.isfile(jfile) else None
    cubie = read_ne_csv_masked(cfile) if os.path.isfile(cfile) else None

    floor = FLOOR_REL * scale
    order_cap = ORDER_CAP_REL * scale

    # Errors and the mutual rms distance are computed over the trajectories
    # BOTH stacks converged on (julia carries a per-trajectory retcode flag;
    # cubie signals a failed solve with NaN). The per-dt non-converged counts
    # are surfaced separately so a cubie stack that solves fewer trajectories
    # than julia is visible rather than hidden by the intersection.
    points = []   # (dt, err_c, err_j, rms_diff, max_diff)
    nonconv = {}  # dt -> (nonconv_cubie, nonconv_julia, mutual_n)
    for dt in DTS_NE:
        fc = cubie.get(dt) if cubie else None
        fj = julia.get(dt) if julia else None
        c_arr, c_conv = fc if fc is not None else (None, None)
        j_arr, j_conv = fj if fj is not None else (None, None)
        err_c = err_j = rms_diff = max_diff = None
        nc_c = int((~c_conv).sum()) if c_arr is not None else None
        nc_j = int((~j_conv).sum()) if j_arr is not None else None
        mutual_n = None
        if c_arr is not None and j_arr is not None:
            mutual = c_conv & j_conv
            mutual_n = int(mutual.sum())
            err_c = ensemble_error_masked(c_arr, golden_states, mutual)
            err_j = ensemble_error_masked(j_arr, golden_states, mutual)
            if mutual.any():
                with np.errstate(invalid="ignore"):
                    d = c_arr[mutual] - j_arr[mutual]
                rms_diff = float(np.sqrt(np.mean(d ** 2)))
                max_diff = float(np.max(np.abs(d)))
        elif c_arr is not None:
            err_c = ensemble_error_masked(c_arr, golden_states, c_conv)
        elif j_arr is not None:
            err_j = ensemble_error_masked(j_arr, golden_states, j_conv)
        points.append((dt, err_c, err_j, rms_diff, max_diff))
        nonconv[dt] = (nc_c, nc_j, mutual_n)

    order_c = observed_orders(
        {dt: e for dt, e, _, _, _ in points if e is not None}, floor,
        order_cap)
    order_j = observed_orders(
        {dt: e for dt, _, e, _, _ in points if e is not None}, floor,
        order_cap)

    # Verdict over dts where both sides are present and the better error is
    # in the convergence region.
    if cubie is None and julia is None:
        verdict = "NO DATA"
    elif cubie is None:
        verdict = "NO CUBIE DATA"
    elif julia is None:
        verdict = "NO JULIA DATA"
    else:
        checked = 0
        julia_valid = 0
        ok = True
        for dt, err_c, err_j, rms_diff, _ in points:
            if err_c is None or err_j is None:
                continue
            if err_j is not None and np.isfinite(err_j) and floor < err_j:
                julia_valid += 1
            if not np.isfinite(err_c) or not np.isfinite(err_j):
                continue
            emax = max(err_c, err_j)
            if not (floor < emax):
                continue
            checked += 1
            if rms_diff > max(EQ_FRACTION * emax, EQ_FLOOR_MULT * floor):
                ok = False
        if checked < 2:
            # Julia produced sane in-region errors that cubie never matched
            # anywhere (out-of-region / non-finite): that is divergence, not
            # missing data.
            verdict = ("MISMATCH" if julia_valid >= 2
                       else "INSUFFICIENT OVERLAP")
        elif ok:
            verdict = "EQUIVALENT"
        else:
            verdict = "MISMATCH"

    return {
        "row": row, "points": points, "order_cubie": order_c,
        "order_julia": order_j, "verdict": verdict, "nonconv": nonconv,
        "missing_cubie": cubie is None, "missing_julia": julia is None,
    }


def signed_tier_verdict(points, err_field, floor, have_data):
    """Signed accuracy verdict for one adaptive tier against Julia.

    Uses the worst-case cubie/julia error ratio over the in-range tolerances
    (both errors are already over the mutually-converged trajectories). Being
    more accurate than Julia (ratio <= 1) always passes; a tier is LESS
    ACCURATE only when cubie's error exceeds ADAPTIVE_TOL x Julia's somewhere.
    """
    if not have_data:
        return "NO DATA"
    worst = None
    for entry in points:
        err_j, err_c = entry["err_julia"], entry.get(err_field)
        if (err_j is None or err_c is None or not np.isfinite(err_j)
                or not np.isfinite(err_c) or err_j <= 0
                or not (floor < max(err_j, err_c))):
            continue
        r = err_c / err_j
        worst = r if worst is None else max(worst, r)
    if worst is None:
        return "INSUFFICIENT OVERLAP"
    if worst > ADAPTIVE_TOL:
        return "LESS ACCURATE (worst ratio {0:.3g})".format(worst)
    if worst <= 1.0:
        return "MORE ACCURATE (worst ratio {0:.2f})".format(worst)
    return "TRACKING (worst ratio {0:.2f})".format(worst)


def analyse_adaptive(row, key, golden_states, scale):
    """Per-tolerance metrics and verdict for one algorithm's adaptive runs."""
    alias = row["cubie_alias"]
    jfile = julia_ne_adaptive_file(alias)
    julia = read_ne_adaptive_csv_masked(jfile) if os.path.isfile(jfile) else None
    tiers = {}
    for tier in ("default", "matched"):
        cfile = cubie_ne_adaptive_file(alias, tier, key)
        tiers[tier] = (read_ne_adaptive_csv_masked(cfile)
                       if os.path.isfile(cfile) else None)
    # Algorithms outside the mutual adaptive set (no cubie data) don't
    # belong in the adaptive report.
    if not any(tiers.values()):
        return None

    floor = FLOOR_REL * scale

    # Masked-read blocks are (finals, converged, naccept, nreject). Errors and
    # the matched-vs-julia distance use only trajectories converged across every
    # present tier (julia + whichever cubie tiers exist); per-tier non-converged
    # counts are reported separately.
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
        med_steps = (float(np.nanmedian(j[2])) if j is not None
                     and j[2] is not None else None)
        entry = {
            "tol": tol, "err_julia": err_j, "julia_steps": med_steps,
            "nonconv_julia": int((~j[1]).sum()) if j is not None else None,
            "nonconv_default": (int((~dflt[1]).sum())
                                if dflt is not None else None),
            "nonconv_matched": (int((~mtch[1]).sum())
                                if mtch is not None else None),
            "mutual_n": int(mutual.sum()) if mutual is not None else None,
        }
        for tier, data in (("default", dflt), ("matched", mtch)):
            entry["err_" + tier] = (
                ensemble_error_masked(data[0], golden_states, mutual)
                if data is not None else None)
        if mtch is not None and j is not None and mutual is not None \
                and mutual.any():
            with np.errstate(invalid="ignore"):
                d = mtch[0][mutual] - j[0][mutual]
            dn = np.sqrt(np.sum(d ** 2, axis=1))
            entry["rms_diff"] = float(np.sqrt(np.mean(d ** 2)))
            entry["max_diff"] = float(np.nanmax(dn))
            entry["p99_diff"] = float(np.nanpercentile(dn, 99))
        else:
            entry["rms_diff"] = None
            entry["max_diff"] = None
            entry["p99_diff"] = None
        points.append(entry)

    # Verdicts (both tiers, same signed rule): worst-case cubie/julia error
    # ratio over the in-range tolerances. <=1 means cubie is at least as
    # accurate as Julia everywhere; <=ADAPTIVE_TOL means at most that factor
    # worse; above it the tier is LESS ACCURATE.
    matched_verdict = signed_tier_verdict(points, "err_matched", floor,
                                          tiers["matched"] is not None
                                          and julia is not None)
    default_verdict = signed_tier_verdict(points, "err_default", floor,
                                          tiers["default"] is not None
                                          and julia is not None)

    return {
        "row": row, "points": points,
        "matched_verdict": matched_verdict,
        "default_verdict": default_verdict,
    }


def fmt(value, spec="{0:.3e}"):
    return spec.format(value) if value is not None else "-"


def adaptive_report_lines(adaptive_results):
    lines = []
    lines.append("## Adaptive sweeps (error vs tolerance)")
    lines.append("")
    lines.append("Each adaptive algorithm ran at atol = rtol over the "
                 "tolerance grid, in Float32, with pinned initial dt and dt "
                 "bounds. `default` is cubie's own PI controller defaults; "
                 "`matched` mirrors the controller type, gains, safety, gain "
                 "clamps and deadband that DifferentialEquations.jl resolved "
                 "for that algorithm (constants exported by the Julia "
                 "runner).")
    lines.append("")
    lines.append("Verdicts (both tiers, same signed rule): the worst-case "
                 "cubie/julia error ratio (vs golden, over mutually-converged "
                 "trajectories) across the in-range tolerances. MORE ACCURATE "
                 "when cubie is at least as accurate as Julia at every "
                 "tolerance (worst ratio <= 1); TRACKING when cubie is at most "
                 "{0:g}x Julia's error; LESS ACCURATE above that. The `rms "
                 "diff` / `p99 diff` / `max diff` columns are the unsigned "
                 "cubie-vs-Julia trajectory distance — reported for context "
                 "(a float32 accept/reject flip decouples the dt sequences) "
                 "but not used for the verdict, since it penalises cubie even "
                 "when cubie is more accurate.".format(ADAPTIVE_TOL))
    lines.append("")
    lines.append("| algorithm | order | default vs julia | matched vs julia |")
    lines.append("|---|---|---|---|")
    for res in adaptive_results:
        lines.append("| {0} | {1} | {2} | {3} |".format(
            res["row"]["cubie_alias"], res["row"]["order"],
            res["default_verdict"], res["matched_verdict"]))
    lines.append("")
    for res in adaptive_results:
        row = res["row"]
        lines.append("### {0} (order {1}, adaptive)".format(
            row["cubie_alias"], row["order"]))
        lines.append("")
        lines.append("| tol | err cubie (default) | err cubie (matched) | "
                     "err julia | rms diff | p99 diff | max diff | "
                     "julia steps (med) | non-conv def/matched/julia |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for entry in res["points"]:
            nonconv = "{0}/{1}/{2}".format(
                fmt(entry.get("nonconv_default"), "{0:d}"),
                fmt(entry.get("nonconv_matched"), "{0:d}"),
                fmt(entry.get("nonconv_julia"), "{0:d}"))
            lines.append(
                "| {0:.0e} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} |"
                .format(entry["tol"], fmt(entry["err_default"]),
                        fmt(entry["err_matched"]), fmt(entry["err_julia"]),
                        fmt(entry["rms_diff"]), fmt(entry["p99_diff"]),
                        fmt(entry["max_diff"]),
                        fmt(entry["julia_steps"], "{0:.0f}"), nonconv))
        lines.append("")
    return lines


def write_report(key, results, scale, outfile, adaptive_results=None):
    lines = []
    lines.append("# Numerical equivalence: cubie vs DifferentialEquations.jl "
                 "({0})".format(key))
    lines.append("")
    lines.append("Fixed-step Float32 convergence study on the Lorenz ensemble "
                 "(N=1024, rho in [0, 21], t in [0, 1]). Both stacks integrate "
                 "bit-identical Float32 inputs at each dt; errors are ensemble "
                 "l2 norms of the final state against the Float64 golden "
                 "reference (Vern9, tol 1e-13). `rms diff` is the mutual rms "
                 "distance between the two implementations' final states.")
    lines.append("")
    lines.append("Golden-scale rms: {0:.4g}. Order estimates use errors in "
                 "({1:.2e}, {2:.2e}); equivalence verdicts use every point "
                 "above the roundoff floor {1:.2e} where both stacks "
                 "converged — there is no upper cap, so a large converged "
                 "disagreement counts as a mismatch.".format(
                     scale, FLOOR_REL * scale, ORDER_CAP_REL * scale))
    lines.append("")
    lines.append("Verdicts: EQUIVALENT — the mutual rms difference stays below "
                 "{0:.0%} of the truncation error at every in-region dt. Every "
                 "algorithm is judged this way, including different-tableau "
                 "pairs such as sdirk_2_2 (Julia's SDIRK22 aliases to "
                 "Trapezoid) — a genuine tableau difference is a mismatch, not "
                 "a pass. MISMATCH — it does not hold; the offending rows are "
                 "visible in the per-algorithm tables.".format(
                     EQ_FRACTION))
    lines.append("")
    lines.append("Errors and the mutual rms distance use only the "
                 "trajectories both stacks converged on. `worst extra "
                 "non-conv` is the largest per-dt excess of cubie's "
                 "non-converged trajectory count over julia's (positive => "
                 "cubie solved fewer than julia at some dt); the per-dt counts "
                 "are in each algorithm's table.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| algorithm | family | order | obs. order (cubie) | "
                 "obs. order (julia) | verdict | worst extra non-conv "
                 "(cubie-julia) |")
    lines.append("|---|---|---|---|---|---|---|")
    for res in results:
        row = res["row"]
        extras = [nc_c - nc_j for (nc_c, nc_j, _) in res.get(
            "nonconv", {}).values() if nc_c is not None and nc_j is not None]
        worst_extra = max(extras) if extras else None
        lines.append("| {0} | {1} | {2} | {3} | {4} | {5} | {6} |".format(
            row["cubie_alias"], row["family"], row["order"],
            fmt(res["order_cubie"], "{0:.2f}") if np.isfinite(
                res["order_cubie"]) else "-",
            fmt(res["order_julia"], "{0:.2f}") if np.isfinite(
                res["order_julia"]) else "-",
            res["verdict"], fmt(worst_extra, "{0:+d}")))
    lines.append("")

    for res in results:
        row = res["row"]
        lines.append("## {0} (order {1}, {2})".format(
            row["cubie_alias"], row["order"], row["julia_expr"]))
        lines.append("")
        if row["notes"]:
            lines.append("_{0}_".format(row["notes"]))
            lines.append("")
        lines.append("| dt | err cubie | err julia | ratio | rms diff | "
                      "max diff | non-conv cubie | non-conv julia |")
        lines.append("|---|---|---|---|---|---|---|---|")
        nonconv = res.get("nonconv", {})
        for dt, err_c, err_j, rms_diff, max_diff in res["points"]:
            ratio = (err_c / err_j
                     if err_c is not None and err_j not in (None, 0.0)
                     else None)
            nc_c, nc_j, _ = nonconv.get(dt, (None, None, None))
            lines.append(
                "| {0:.10g} | {1} | {2} | {3} | {4} | {5} | {6} | {7} |".format(
                    dt, fmt(err_c), fmt(err_j), fmt(ratio, "{0:.3f}"),
                    fmt(rms_diff), fmt(max_diff),
                    fmt(nc_c, "{0:d}"), fmt(nc_j, "{0:d}")))
        lines.append("")

    if adaptive_results:
        lines.extend(adaptive_report_lines(adaptive_results))

    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_plot(key, results, scale, outfile):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.0 * ncols, 3.2 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for idx, res in enumerate(results):
        ax = axes[idx // ncols][idx % ncols]
        row = res["row"]
        dts_c = [dt for dt, e, _, _, _ in res["points"] if e is not None]
        errs_c = [e for _, e, _, _, _ in res["points"] if e is not None]
        dts_j = [dt for dt, _, e, _, _ in res["points"] if e is not None]
        errs_j = [e for _, _, e, _, _ in res["points"] if e is not None]
        if dts_c:
            ax.loglog(dts_c, errs_c, "o-", color="tab:blue", label="cubie")
        if dts_j:
            ax.loglog(dts_j, errs_j, "x--", color="tab:red",
                      label="DiffEq.jl")
        # Theoretical-order guide anchored to the julia curve mid-region.
        anchor = [(dt, e) for dt, e in zip(dts_j, errs_j)
                  if FLOOR_REL * scale < e < ORDER_CAP_REL * scale]
        if anchor:
            dt0, e0 = anchor[len(anchor) // 2]
            guide_dts = np.array([min(dts_j), max(dts_j)])
            ax.loglog(guide_dts, e0 * (guide_dts / dt0) ** row["order"],
                      ":", color="gray", linewidth=1,
                      label="order {0}".format(row["order"]))
        ax.axhline(FLOOR_REL * scale, color="gray", linewidth=0.5,
                   linestyle="-.")
        ax.set_title("{0} [{1}]".format(row["cubie_alias"], res["verdict"]),
                     fontsize=9)
        ax.grid(True, which="both", alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_axis_off()
    # Clamp the shared y-window: divergent coarse-dt points reach 1e30+ and
    # would squash the meaningful error range into a sliver.
    axes[0][0].set_ylim(FLOOR_REL * scale / 30.0, 100.0 * scale)
    for ax in axes[-1]:
        ax.set_xlabel("dt")
    for r in range(nrows):
        axes[r][0].set_ylabel("ensemble l2 error")
    fig.suptitle("Numerical equivalence, Float32 fixed-step Lorenz ensemble "
                 "({0})".format(key))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def write_adaptive_plot(key, adaptive_results, scale, outfile):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(adaptive_results)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.0 * ncols, 3.2 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
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
        # err = tol fidelity guide.
        tols = [e["tol"] for e in res["points"]]
        ax.loglog(tols, tols, ":", color="gray", linewidth=1,
                  label="err = tol")
        ax.axhline(FLOOR_REL * scale, color="gray", linewidth=0.5,
                   linestyle="-.")
        ax.set_title("{0} [{1}]".format(res["row"]["cubie_alias"],
                                        res["matched_verdict"].split(" ")[0]),
                     fontsize=9)
        ax.grid(True, which="both", alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("tolerance (atol = rtol)")
    for r in range(nrows):
        axes[r][0].set_ylabel("ensemble l2 error")
    fig.suptitle("Adaptive numerical equivalence, Float32 Lorenz ensemble "
                 "({0})".format(key))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def main():
    algorithms = load_algorithms()
    _, golden_states = load_golden_ne()
    scale = float(np.sqrt(np.mean(golden_states ** 2)))

    keys = discover_keys([row["cubie_alias"] for row in algorithms])
    if not keys:
        print("No cubie ne outputs found in {0}; run "
              "GPU_ODE_CUBIE/numerical_equivalence.py first."
              .format(CUBIE_NE_DIR))
        return 1

    status = 0
    for key in sorted(keys):
        results = [analyse_algorithm(row, key, golden_states, scale)
                   for row in algorithms]
        adaptive_results = [
            res for res in (analyse_adaptive(row, key, golden_states, scale)
                            for row in algorithms)
            if res is not None]
        report = "numerical_equivalence_{0}.md".format(key)
        plot = os.path.join("plots", "numerical_equivalence_{0}.png"
                            .format(key))
        write_report(key, results, scale, report,
                     adaptive_results=adaptive_results)
        write_plot(key, results, scale, plot)
        n_bad = sum(res["verdict"] == "MISMATCH" for res in results)
        n_nodata = sum("NO" in res["verdict"] or
                       res["verdict"] == "INSUFFICIENT OVERLAP"
                       for res in results)
        print("[{0}] {1} algorithms: {2} mismatched, {3} without enough "
              "data; report -> {4}, plot -> {5}".format(
                  key, len(results), n_bad, n_nodata, report, plot))
        for res in results:
            print("  {0:22s} {1}".format(res["row"]["cubie_alias"],
                                         res["verdict"]))
        if n_bad:
            status = 2

        if adaptive_results:
            aplot = os.path.join(
                "plots", "numerical_equivalence_adaptive_{0}.png".format(key))
            write_adaptive_plot(key, adaptive_results, scale, aplot)
            n_div = sum("LESS ACCURATE" in res["matched_verdict"]
                        for res in adaptive_results)
            print("[{0}] adaptive: {1} algorithms, {2} less accurate than "
                  "julia under matched controllers; plot -> {3}".format(
                      key, len(adaptive_results), n_div, aplot))
            for res in adaptive_results:
                print("  {0:22s} default={1:20s} matched={2}".format(
                    res["row"]["cubie_alias"], res["default_verdict"],
                    res["matched_verdict"]))
            if n_div:
                status = 2
    return status


if __name__ == "__main__":
    sys.exit(main())
