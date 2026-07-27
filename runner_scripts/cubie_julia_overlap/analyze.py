#!/usr/bin/env python3
"""Analyze one direct-overlap run and render CSV summaries, plots and report."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import algorithms  # noqa: E402 - suite-local bootstrap above


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fkey(row):
    return (row["algorithm"], row["phase"], row["mode"], int(row["n"]),
            row["setting_kind"], round(float(row["setting"]), 14))


def valid_metric(row):
    return (int(row["failed_trajectories"]) == 0
            and int(row["finite_trajectories"]) == int(row["n"]))


def load_finals(root, relative):
    return np.loadtxt(root / relative, delimiter=",", skiprows=1, usecols=(1, 2, 3), dtype=np.float64)


def timing_summary(rows, metrics=None):
    eligible = None
    if metrics is not None:
        eligible = {
            (r["framework"], r["algorithm"], r["phase"], r["mode"], r["tier"],
             r["n"], r["setting_kind"], round(float(r["setting"]), 14))
            for r in metrics
            if valid_metric(r)
        }
    groups = defaultdict(list)
    for row in rows:
        validity_key = (row["framework"], row["algorithm"], row["phase"],
                        row["mode"], row["tier"], row["n"],
                        row["setting_kind"], round(float(row["setting"]), 14))
        if eligible is not None and validity_key not in eligible:
            continue
        key = (row["framework"], row["algorithm"], row["phase"], row["mode"],
               row["tier"], row["n"], row["setting_kind"], row["setting"])
        groups[key].append(float(row["time_ms"]))
    out = []
    for key, values in sorted(groups.items()):
        a = np.asarray(values)
        out.append(dict(zip(("framework", "algorithm", "phase", "mode", "tier",
                             "n", "setting_kind", "setting"), key),
                        samples=len(values), min_ms=float(np.min(a)),
                        p05_ms=float(np.percentile(a, 5)), median_ms=float(np.median(a)),
                        p95_ms=float(np.percentile(a, 95)), max_ms=float(np.max(a))))
    return out


def observed_orders(metrics):
    groups = defaultdict(list)
    for row in metrics:
        if row["phase"] == "numerical" and row["mode"] == "fixed":
            err = float(row["golden_rmse"])
            if math.isfinite(err) and err > 0:
                groups[(row["framework"], row["algorithm"], row["tier"])].append(
                    (float(row["setting"]), err))
    out = []
    for (framework, algorithm, tier), values in sorted(groups.items()):
        values.sort(reverse=True)
        slopes = []
        for (dt1, e1), (dt2, e2) in zip(values, values[1:]):
            slope = math.log(e1 / e2) / math.log(dt1 / dt2)
            if math.isfinite(slope) and 0 < slope < 20:
                slopes.append(slope)
        out.append({"framework": framework, "algorithm": algorithm, "tier": tier,
                    "observed_order": statistics.median(slopes) if slopes else math.nan,
                    "usable_intervals": len(slopes)})
    return out


def numerical_comparisons(root, metrics):
    julia = {fkey(r): r for r in metrics if r["framework"] == "julia" and r["phase"] == "numerical"}
    out = []
    for cubie in (r for r in metrics if r["framework"] == "cubie" and r["phase"] == "numerical"):
        other = julia.get(fkey(cubie))
        if other is None or not cubie["finals_path"] or not other["finals_path"]:
            continue
        a, b = load_finals(root, cubie["finals_path"]), load_finals(root, other["finals_path"])
        good = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
        delta = np.abs(a[good] - b[good])
        out.append({
            "algorithm": cubie["algorithm"], "mode": cubie["mode"],
            "cubie_tier": cubie["tier"], "n": cubie["n"],
            "setting_kind": cubie["setting_kind"], "setting": cubie["setting"],
            "mutual_rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else math.nan,
            "mutual_p99_abs": float(np.percentile(delta, 99)) if delta.size else math.nan,
            "mutual_max_abs": float(np.max(delta)) if delta.size else math.nan,
            "finite_pairs": int(np.sum(good)), "failed_pairs": int(len(good) - np.sum(good)),
        })
    return out


def speedups(summaries):
    julia = {}
    for row in summaries:
        if row["framework"] == "julia":
            julia[(row["algorithm"], row["phase"], row["mode"], row["n"],
                   row["setting_kind"], round(float(row["setting"]), 14))] = row
    out = []
    for row in summaries:
        if row["framework"] != "cubie":
            continue
        key = (row["algorithm"], row["phase"], row["mode"], row["n"],
               row["setting_kind"], round(float(row["setting"]), 14))
        other = julia.get(key)
        if other:
            out.append({"algorithm": row["algorithm"], "phase": row["phase"],
                        "mode": row["mode"], "cubie_tier": row["tier"], "n": row["n"],
                        "setting_kind": row["setting_kind"], "setting": row["setting"],
                        "cubie_median_ms": row["median_ms"],
                        "julia_median_ms": other["median_ms"],
                        "julia_over_cubie_speedup": float(other["median_ms"]) / float(row["median_ms"])})
    return out


def work_precision_rows(summaries, metrics):
    metric_by_key = {
        (r["framework"], r["algorithm"], r["phase"], r["mode"], r["tier"],
         r["n"], r["setting_kind"], round(float(r["setting"]), 14)): r
        for r in metrics if r["phase"] == "work_precision"
    }
    out = []
    for timing in summaries:
        key = (timing["framework"], timing["algorithm"], timing["phase"],
               timing["mode"], timing["tier"], timing["n"],
               timing["setting_kind"], round(float(timing["setting"]), 14))
        metric = metric_by_key.get(key)
        if metric is None:
            continue
        out.append({
            "framework": timing["framework"], "algorithm": timing["algorithm"],
            "mode": timing["mode"], "tier": timing["tier"], "n": timing["n"],
            "setting_kind": timing["setting_kind"], "setting": timing["setting"],
            "samples": timing["samples"], "min_ms": timing["min_ms"],
            "p05_ms": timing["p05_ms"], "median_ms": timing["median_ms"],
            "p95_ms": timing["p95_ms"], "max_ms": timing["max_ms"],
            "golden_rmse": metric["golden_rmse"],
            "finite_trajectories": metric["finite_trajectories"],
            "failed_trajectories": metric["failed_trajectories"],
        })
    return out


def plots(root, summaries, metrics, work_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [a["cubie_alias"] for a in algorithms()]
    colors = {("julia", "fixed"): "black", ("julia", "julia"): "black",
              ("cubie", "fixed"): "tab:blue", ("cubie", "default"): "tab:orange",
              ("cubie", "pi"): "tab:green"}
    plots_dir = root / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(len(names), 2, figsize=(12, 3.2 * len(names)), squeeze=False)
    for i, name in enumerate(names):
        for j, mode in enumerate(("fixed", "adaptive")):
            ax = axes[i, j]
            for framework, tier in (("julia", "fixed" if mode == "fixed" else "julia"),
                                    ("cubie", "fixed" if mode == "fixed" else "default"),
                                    ("cubie", "pi")):
                vals = [r for r in summaries if r["phase"] == "performance" and
                        r["algorithm"] == name and r["mode"] == mode and
                        r["framework"] == framework and r["tier"] == tier]
                if vals:
                    vals.sort(key=lambda r: int(r["n"]))
                    ax.loglog([int(r["n"]) for r in vals], [float(r["median_ms"]) for r in vals],
                              marker="o", color=colors[(framework, tier)], label="{} {}".format(framework, tier))
            ax.set(title="{} — {}".format(name, mode), xlabel="trajectories", ylabel="median ms")
            ax.grid(True, which="both", alpha=.25)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "performance_scaling.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(names), 2, figsize=(12, 3.2 * len(names)), squeeze=False)
    for i, name in enumerate(names):
        for j, mode in enumerate(("fixed", "adaptive")):
            ax = axes[i, j]
            for framework, tier in (("julia", "fixed" if mode == "fixed" else "julia"),
                                    ("cubie", "fixed" if mode == "fixed" else "default"),
                                    ("cubie", "pi")):
                vals = [r for r in metrics if r["phase"] == "numerical" and r["algorithm"] == name and
                        r["mode"] == mode and r["framework"] == framework and r["tier"] == tier]
                if vals:
                    vals.sort(key=lambda r: float(r["setting"]), reverse=True)
                    ax.loglog([float(r["setting"]) for r in vals], [float(r["golden_rmse"]) for r in vals],
                              marker="o", color=colors[(framework, tier)], label="{} {}".format(framework, tier))
            ax.invert_xaxis()
            ax.set(title="{} — {}".format(name, mode), xlabel="dt" if mode == "fixed" else "tolerance",
                   ylabel="golden RMSE")
            ax.grid(True, which="both", alpha=.25)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "numerical_equivalence.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(names), 2, figsize=(12, 3.2 * len(names)), squeeze=False)
    for i, name in enumerate(names):
        for j, mode in enumerate(("fixed", "adaptive")):
            ax = axes[i, j]
            for framework, tier in (("julia", "fixed" if mode == "fixed" else "julia"),
                                    ("cubie", "fixed" if mode == "fixed" else "default"),
                                    ("cubie", "pi")):
                vals = [r for r in work_rows if r["algorithm"] == name and r["mode"] == mode and
                        r["framework"] == framework and r["tier"] == tier]
                if vals:
                    vals.sort(key=lambda r: float(r["median_ms"]))
                    ax.loglog([float(r["median_ms"]) for r in vals], [float(r["golden_rmse"]) for r in vals],
                              marker="o", color=colors[(framework, tier)], label="{} {}".format(framework, tier))
            ax.set(title="{} — {}".format(name, mode), xlabel="median runtime (ms)", ylabel="golden RMSE")
            ax.grid(True, which="both", alpha=.25)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "work_precision.png", dpi=160)
    plt.close(fig)


def markdown_table(fields, rows, limit=None):
    rows = rows[:limit] if limit else rows
    if not rows:
        return "_No successful rows._\n"
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            vals.append("{:.5g}".format(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    timings = read_rows(root / "cubie_timings.csv") + read_rows(root / "julia_timings.csv")
    metrics = read_rows(root / "cubie_metrics.csv") + read_rows(root / "julia_metrics.csv")
    valid_metrics = [row for row in metrics if valid_metric(row)]
    failures = read_rows(root / "cubie_failures.csv") + read_rows(root / "julia_failures.csv")
    summaries = timing_summary(timings, metrics)
    comparisons = numerical_comparisons(root, valid_metrics)
    orders = observed_orders(valid_metrics)
    boosts = speedups(summaries)
    work_rows = work_precision_rows(summaries, valid_metrics)
    write_rows(root / "timing_summary.csv", ["framework", "algorithm", "phase", "mode", "tier", "n",
               "setting_kind", "setting", "samples", "min_ms", "p05_ms", "median_ms", "p95_ms", "max_ms"], summaries)
    write_rows(root / "numerical_comparisons.csv", ["algorithm", "mode", "cubie_tier", "n", "setting_kind",
               "setting", "mutual_rmse", "mutual_p99_abs", "mutual_max_abs", "finite_pairs", "failed_pairs"], comparisons)
    write_rows(root / "observed_orders.csv", ["framework", "algorithm", "tier", "observed_order", "usable_intervals"], orders)
    write_rows(root / "speedups.csv", ["algorithm", "phase", "mode", "cubie_tier", "n", "setting_kind", "setting",
               "cubie_median_ms", "julia_median_ms", "julia_over_cubie_speedup"], boosts)
    write_rows(root / "work_precision.csv", ["framework", "algorithm", "mode", "tier", "n", "setting_kind",
               "setting", "samples", "min_ms", "p05_ms", "median_ms", "p95_ms", "max_ms", "golden_rmse",
               "finite_trajectories", "failed_trajectories"], work_rows)
    try:
        plots(root, summaries, valid_metrics, work_rows)
        plot_note = "Plots: `plots/performance_scaling.png`, `plots/numerical_equivalence.png`, and `plots/work_precision.png`."
    except Exception as exc:
        (root / "plot_failure.txt").write_text("{}: {}\n".format(type(exc).__name__, exc), encoding="utf-8")
        plot_note = "Plot generation failed; see `plot_failure.txt`."

    perf = [r for r in boosts if r["phase"] == "performance"]
    report = """# Cubie ↔ DiffEqGPU benchmark

Algorithms: `tsit5` ↔ `GPUTsit5()`, `vern7` ↔ `GPUVern7()`, `rosenbrock23_sciml` ↔ `GPURosenbrock23()`, `kvaerno3` ↔ `GPUKvaerno3()`, and `kvaerno5` ↔ `GPUKvaerno5()`.

`diffeqgpu_ode_inventory.csv` contains the eight specialized DiffEqGPU GPU ODE algorithms and their Cubie mappings.

All timed samples are synchronized end-to-end solves, including final host transfer. Each point has an untimed warmup. Fixed performance uses dyadic `dt=2^-10`; full numerical equivalence uses N=1024 with `dt=2^-1..2^-13` and tolerances `1e-2..1e-6`; full work-precision uses N=32768 with `dt=2^-4..2^-13`, tolerances `1e-2..1e-8`, and 20 repeats by default. The raw sample CSVs remain the authoritative timing record.

Every phase, including performance, records finite and failed trajectory counts. A point without a complete all-finite validity metric is excluded from timing summaries, speedups, and plots. Workers continue after point failures to preserve successful evidence, then return nonzero so the launcher marks the suite incomplete.

Analytic Lorenz Jacobian and time-gradient functions are supplied to every Julia problem, including explicit runs, and implicit Julia constructors disable autodiff. A failed point is appended to the failure ledger and later points continue.

## Performance speedups

`julia_over_cubie_speedup > 1` means Cubie was faster. Percentile timing statistics for both frameworks are in `timing_summary.csv`.

"""
    report += markdown_table(["algorithm", "mode", "cubie_tier", "n", "cubie_median_ms", "julia_median_ms", "julia_over_cubie_speedup"], perf)
    report += "\n## Numerical equivalence\n\nPer-trajectory Float32 finals are retained beneath `finals/`. Mutual metrics are elementwise at t=1.\n\n"
    report += markdown_table(["algorithm", "mode", "cubie_tier", "setting", "mutual_rmse", "mutual_p99_abs", "mutual_max_abs", "failed_pairs"], comparisons)
    report += "\n## Observed fixed-step convergence order\n\n"
    report += markdown_table(["framework", "algorithm", "tier", "observed_order", "usable_intervals"], orders)
    report += "\n## Work-precision\n\n`work_precision.csv` joins every work-point timing distribution (min/p05/median/p95/max) to golden RMSE. `plots/work_precision.png` plots median runtime on the x-axis against golden RMSE on the y-axis; it is an error-work plot, not another setting sweep.\n"
    report += "\n## Failures and non-finite results\n\n{} point failures were recorded. See the framework failure CSVs for full messages. Non-finite trajectory counts are retained per successful point in the metric CSVs.\n\n".format(len(failures))
    report += markdown_table(["framework", "algorithm", "phase", "mode", "tier", "setting_kind", "setting", "error_type", "message"], failures, limit=100)
    report += "\n## Artifacts\n\n" + plot_note + " Raw and derived CSVs in this directory are algorithm-, mode-, tier-, N-, and setting-keyed.\n"
    (root / "report.md").write_text(report, encoding="utf-8")
    print("Wrote {}".format(root / "report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
