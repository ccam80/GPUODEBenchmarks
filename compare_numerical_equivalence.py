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
                       ensemble_error, julia_ne_file, cubie_ne_file,
                       julia_ne_adaptive_file, cubie_ne_adaptive_file,
                       read_ne_csv, read_ne_adaptive_csv, CUBIE_NE_DIR)

# Windows consoles default to a legacy codepage (cp1252) that cannot encode
# the glyphs printed below; force UTF-8 where supported.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Error bounds (relative to the golden scale) delimiting where each check
# applies. Below FLOOR_REL the error is float32 roundoff, not truncation.
# Above ORDER_CAP_REL the solve has left the asymptotic regime and the
# observed order is meaningless. The equivalence check tolerates a much
# looser cap: two implementations of the same tableau still track each other
# closely at coarse dt where the error is large but finite, and that
# agreement is strong evidence — only outright divergence (error at the
# scale of the attractor itself) is uninformative.
FLOOR_REL = 4e-6
ORDER_CAP_REL = 3e-2
EQ_CAP_REL = 5e-1

# Equivalence: two implementations of the identical tableau differ only by
# roundoff, so their mutual rms distance must stay well below the truncation
# error (factor EQ_FRACTION) once above the roundoff floor allowance.
EQ_FRACTION = 0.25
EQ_FLOOR_MULT = 3.0

# Non-exact pairs (same method class, different tableau) are only required
# to converge at the same order with errors of comparable magnitude.
ORDER_TOL = 0.6
RATIO_LIM = 4.0

# Adaptive matched-controller tier: even with identical controller type,
# gains and tolerances, a single float32-roundoff flip of one accept/reject
# decision decouples the two stacks' dt sequences, after which their
# per-trajectory difference is bounded below by the local truncation error
# itself — an absolute divergence gate is unattainable here (it IS
# attainable in the fixed-step suite, where the dt sequence is pinned). The
# defensible gate is relative: the mutual rms distance must not exceed
# ADAPTIVE_EQ_FRACTION times the Julia run's own error at that tolerance.
# Healthy algorithms sit at ratio ~1; broken ones sit orders of magnitude
# above.
ADAPTIVE_EQ_FRACTION = 2.0
# Default-controller curves reflect each stack's controller personality;
# they are classified by the median error ratio rather than gated.
ADAPTIVE_RATIO_LIM = 4.0


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
    julia = read_ne_csv(jfile) if os.path.isfile(jfile) else None
    cubie = read_ne_csv(cfile) if os.path.isfile(cfile) else None

    floor = FLOOR_REL * scale
    order_cap = ORDER_CAP_REL * scale
    eq_cap = EQ_CAP_REL * scale

    points = []   # (dt, err_c, err_j, rms_diff, max_diff)
    for dt in DTS_NE:
        fc = cubie.get(dt) if cubie else None
        fj = julia.get(dt) if julia else None
        err_c = ensemble_error(fc, golden_states) if fc is not None else None
        err_j = ensemble_error(fj, golden_states) if fj is not None else None
        rms_diff = max_diff = None
        if fc is not None and fj is not None:
            with np.errstate(invalid="ignore"):
                d = fc - fj
            rms_diff = float(np.sqrt(np.mean(d ** 2)))
            max_diff = float(np.max(np.abs(d)))
        points.append((dt, err_c, err_j, rms_diff, max_diff))

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
            if err_j is not None and np.isfinite(err_j) and \
                    floor < err_j < eq_cap:
                julia_valid += 1
            if not np.isfinite(err_c) or not np.isfinite(err_j):
                continue
            emax = max(err_c, err_j)
            if not (floor < emax < eq_cap):
                continue
            checked += 1
            if row["exact"]:
                if rms_diff > max(EQ_FRACTION * emax, EQ_FLOOR_MULT * floor):
                    ok = False
            else:
                r = err_c / err_j if err_j > 0 else float("inf")
                if not (1.0 / RATIO_LIM < r < RATIO_LIM):
                    ok = False
        if not row["exact"] and np.isfinite(order_c) and np.isfinite(order_j):
            if abs(order_c - order_j) > ORDER_TOL:
                ok = False
        if checked < 2:
            # Julia produced sane in-region errors that cubie never matched
            # anywhere (out-of-region / non-finite): that is divergence, not
            # missing data.
            verdict = ("MISMATCH" if julia_valid >= 2
                       else "INSUFFICIENT OVERLAP")
        elif ok:
            verdict = "EQUIVALENT" if row["exact"] else "CONSISTENT"
        else:
            verdict = "MISMATCH"

    return {
        "row": row, "points": points, "order_cubie": order_c,
        "order_julia": order_j, "verdict": verdict,
        "missing_cubie": cubie is None, "missing_julia": julia is None,
    }


def analyse_adaptive(row, key, golden_states, scale):
    """Per-tolerance metrics and verdict for one algorithm's adaptive runs."""
    alias = row["cubie_alias"]
    jfile = julia_ne_adaptive_file(alias)
    julia = read_ne_adaptive_csv(jfile) if os.path.isfile(jfile) else None
    tiers = {}
    for tier in ("default", "matched"):
        cfile = cubie_ne_adaptive_file(alias, tier, key)
        tiers[tier] = (read_ne_adaptive_csv(cfile)
                       if os.path.isfile(cfile) else None)
    # Algorithms outside the mutual adaptive set (no cubie data) don't
    # belong in the adaptive report.
    if not any(tiers.values()):
        return None

    floor = FLOOR_REL * scale
    eq_cap = EQ_CAP_REL * scale

    points = []
    for tol in TOLS_NE:
        j = julia.get(tol) if julia else None
        err_j = ensemble_error(j[0], golden_states) if j else None
        med_steps = (float(np.nanmedian(j[1])) if j is not None
                     and j[1] is not None else None)
        entry = {"tol": tol, "err_julia": err_j, "julia_steps": med_steps}
        for tier in ("default", "matched"):
            data = tiers[tier].get(tol) if tiers[tier] else None
            err_c = ensemble_error(data[0], golden_states) if data else None
            entry["err_" + tier] = err_c
            if tier == "matched" and data is not None and j is not None:
                with np.errstate(invalid="ignore"):
                    d = data[0] - j[0]
                dn = np.sqrt(np.sum(d ** 2, axis=1))
                entry["rms_diff"] = float(np.sqrt(np.mean(d ** 2)))
                entry["max_diff"] = float(np.nanmax(dn))
                entry["p99_diff"] = float(np.nanpercentile(dn, 99))
            else:
                entry.setdefault("rms_diff", None)
                entry.setdefault("max_diff", None)
                entry.setdefault("p99_diff", None)
        points.append(entry)

    # Verdicts. Matched tier (the CI gate): the mutual rms distance must
    # stay within ADAPTIVE_EQ_FRACTION of the julia run's own error at every
    # tolerance where the julia solve is in a sane range. Default tier
    # (informational): classify the controller personality by the median
    # error ratio.
    if tiers["matched"] is None or julia is None:
        matched_verdict = "NO DATA"
    else:
        checked = 0
        worst = 0.0
        for entry in points:
            err_j = entry["err_julia"]
            if (err_j is None or not np.isfinite(err_j)
                    or not (floor < err_j < eq_cap)):
                continue
            if entry["rms_diff"] is None:
                continue
            checked += 1
            worst = max(worst,
                        entry["rms_diff"] / max(err_j, EQ_FLOOR_MULT * floor))
        if checked == 0:
            matched_verdict = "INSUFFICIENT OVERLAP"
        elif worst <= ADAPTIVE_EQ_FRACTION:
            matched_verdict = "TRACKING (worst ratio {0:.2f})".format(worst)
        else:
            matched_verdict = "DIVERGENT (worst ratio {0:.3g})".format(worst)

    if tiers["default"] is None or julia is None:
        default_verdict = "NO DATA"
    else:
        ratios = []
        for entry in points:
            err_j, err_c = entry["err_julia"], entry["err_default"]
            if (err_j is None or err_c is None
                    or not np.isfinite(err_j) or not np.isfinite(err_c)
                    or not (floor < max(err_j, err_c) < eq_cap)
                    or err_j <= 0):
                continue
            ratios.append(err_c / err_j)
        if not ratios:
            default_verdict = "INSUFFICIENT OVERLAP"
        else:
            med = float(np.median(ratios))
            if 1.0 / ADAPTIVE_RATIO_LIM < med < ADAPTIVE_RATIO_LIM:
                default_verdict = "TRACKING (median ratio {0:.2f})".format(med)
            elif med <= 1.0 / ADAPTIVE_RATIO_LIM:
                default_verdict = ("MORE ACCURATE (median ratio {0:.3g})"
                                   .format(med))
            else:
                default_verdict = ("LESS ACCURATE (median ratio {0:.3g})"
                                   .format(med))

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
    lines.append("Verdicts — matched (the CI gate): even under identical "
                 "controller settings a single float32 accept/reject flip "
                 "decouples the dt sequences, after which the mutual "
                 "distance is bounded below by the local truncation error, "
                 "so the gate is relative: TRACKING when the mutual rms "
                 "distance stays within {0:g}x the Julia run's own error at "
                 "every in-range tolerance (healthy algorithms sit at ratio "
                 "~1; broken ones sit orders of magnitude above). default "
                 "(informational): classified by the median error ratio "
                 "cubie/julia — within a factor of {1:g} counts as "
                 "TRACKING; outside it the tier reports which controller "
                 "personality is more accurate at equal tolerance.".format(
                     ADAPTIVE_EQ_FRACTION, ADAPTIVE_RATIO_LIM))
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
                     "julia steps (med) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for entry in res["points"]:
            lines.append(
                "| {0:.0e} | {1} | {2} | {3} | {4} | {5} | {6} | {7} |"
                .format(entry["tol"], fmt(entry["err_default"]),
                        fmt(entry["err_matched"]), fmt(entry["err_julia"]),
                        fmt(entry["rms_diff"]), fmt(entry["p99_diff"]),
                        fmt(entry["max_diff"]),
                        fmt(entry["julia_steps"], "{0:.0f}")))
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
                 "({1:.2e}, {2:.2e}); equivalence verdicts use errors in "
                 "({1:.2e}, {3:.2e}).".format(
                     scale, FLOOR_REL * scale, ORDER_CAP_REL * scale,
                     EQ_CAP_REL * scale))
    lines.append("")
    lines.append("Verdicts: EQUIVALENT — identical tableau on both sides and "
                 "mutual rms difference stays below {0:.0%} of the truncation "
                 "error at every in-region dt. CONSISTENT — different tableau "
                 "of the same order (see notes); observed orders and error "
                 "magnitudes agree. MISMATCH — neither holds; the offending "
                 "rows are visible in the per-algorithm tables.".format(
                     EQ_FRACTION))
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| algorithm | family | order | obs. order (cubie) | "
                 "obs. order (julia) | verdict |")
    lines.append("|---|---|---|---|---|---|")
    for res in results:
        row = res["row"]
        lines.append("| {0} | {1} | {2} | {3} | {4} | {5} |".format(
            row["cubie_alias"], row["family"], row["order"],
            fmt(res["order_cubie"], "{0:.2f}") if np.isfinite(
                res["order_cubie"]) else "-",
            fmt(res["order_julia"], "{0:.2f}") if np.isfinite(
                res["order_julia"]) else "-",
            res["verdict"]))
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
                      "max diff |")
        lines.append("|---|---|---|---|---|---|")
        for dt, err_c, err_j, rms_diff, max_diff in res["points"]:
            ratio = (err_c / err_j
                     if err_c is not None and err_j not in (None, 0.0)
                     else None)
            lines.append("| {0:.10g} | {1} | {2} | {3} | {4} | {5} |".format(
                dt, fmt(err_c), fmt(err_j), fmt(ratio, "{0:.3f}"),
                fmt(rms_diff), fmt(max_diff)))
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
            n_div = sum("DIVERGENT" in res["matched_verdict"]
                        for res in adaptive_results)
            print("[{0}] adaptive: {1} algorithms, {2} divergent under "
                  "matched controllers; plot -> {3}".format(
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
