"""Rewrite a store's cubie adaptive rows to the controller the sets declare for each algorithm: a row that ran that controller takes its name and gains and is rehashed, its finals file renamed; a row that ran another controller is dropped, as are its finals and optimize records and every cubie finals file no row names; two rows of one run_id keep the most complete; the julia_cpu controllers/ directories are deleted. CLI: rehash_controllers.py [--root DIR] [--dry-run]."""

import argparse
import json
import math
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cubie_adapter  # noqa: E402
import sets  # noqa: E402
import store as store_mod  # noqa: E402
from algorithms import algorithm_facts  # noqa: E402

# Algorithms whose cubie default controller is Julia's: a stored default row ran the declared controller.
DEFAULT_IS_JULIA = ("kvaerno3", "kvaerno5", "l_stable_sdirk_4", "radau_iia_3", "radau_iia_5", "radau_iia_9")
GAIN_REL_TOL = 1e-6


def declared(algorithm):
    """(controller, gains dict) the sets declare for a cubie algorithm; None when Julia has no controller that maps."""
    facts = algorithm_facts(algorithm)
    settings = cubie_adapter.julia_controller(sets.julia_controllers().get(algorithm), facts["order"])
    if settings is None:
        return None
    settings = dict(settings)
    return settings.pop("step_controller"), settings


def ran_declared(algorithm, controller, gains, target):
    """True when a stored (controller, gains) ran the declared controller: the same name with every given gain within GAIN_REL_TOL, or a default row of an algorithm in DEFAULT_IS_JULIA."""
    name, wanted = target
    if controller == "default":
        return not gains and algorithm in DEFAULT_IS_JULIA
    if controller != name or not set(gains) <= set(wanted):
        return False
    return all(math.isclose(float(gains[k]), float(wanted[k]), rel_tol=GAIN_REL_TOL, abs_tol=1e-12)
               for k in gains)


def _completeness(row):
    return (math.isfinite(store_mod._float(row.get("min_ms"))), bool(row.get("finals")),
            math.isfinite(store_mod._float(row.get("build_s"))), row.get("recorded_utc") or 0)


def rewrite_rows(rows):
    """(kept rows, dropped rows, {old finals path: new finals path}) of one results file's rows."""
    kept, dropped, renames = {}, [], {}
    for row in rows:
        if row["package"] not in cubie_adapter.PACKAGES or row["controller"] == "fixed":
            kept[row["run_id"]] = row
            continue
        target = declared(row["algorithm"])
        gains = json.loads(row["gains"] or "{}")
        if target is None or not ran_declared(row["algorithm"], row["controller"], gains, target):
            dropped.append(row)
            continue
        new = dict(row, controller=target[0], gains=store_mod.canonical_json(target[1]))
        new.update(store_mod.ids(new))
        if new["finals"]:
            renamed = store_mod.finals_name(new)
            renames[new["finals"]] = renamed
            new["finals"] = renamed
        held = kept.get(new["run_id"])
        if held is None or _completeness(new) > _completeness(held):
            if held is not None:
                dropped.append(held)
            kept[new["run_id"]] = new
        else:
            dropped.append(row)
    return list(kept.values()), dropped, renames


def rewrite_optimize(rows):
    """(kept, dropped) optimize records: adaptive cubie records take the declared controller and gains, one per kernel identity (the last stands)."""
    kept, dropped = {}, []
    for row in rows:
        if row["controller"] != "fixed":
            target = declared(row["algorithm"])
            gains = json.loads(row["gains"] or "{}")
            if target is None or not ran_declared(row["algorithm"], row["controller"], gains, target):
                dropped.append(row)
                continue
            row = dict(row, controller=target[0], gains=store_mod.canonical_json(target[1]))
        ident = tuple(row.get(f, "") for f in cubie_adapter.OPTIMIZE_FIELDS[:9])
        if ident in kept:
            dropped.append(kept[ident])
        kept[ident] = row
    return list(kept.values()), dropped


def _package_dir(path):
    return os.path.dirname(os.path.dirname(path))


def _move_finals(package_dir, renames, dry_run, out):
    for old, new in renames.items():
        src = os.path.join(package_dir, *old.split("/"))
        dst = os.path.join(package_dir, *new.split("/"))
        if src == dst or not os.path.isfile(src):
            continue
        out.write("  finals {0} -> {1}\n".format(old, new))
        if dry_run:
            continue
        if os.path.isfile(dst):
            os.remove(src)
        else:
            os.replace(src, dst)


def _drop_finals(package_dir, rows, live, dry_run, out):
    for row in rows:
        relative = row.get("finals")
        if not relative or relative in live:
            continue
        path = os.path.join(package_dir, *relative.split("/"))
        if os.path.isfile(path):
            out.write("  finals {0} deleted\n".format(relative))
            if not dry_run:
                os.remove(path)


def rehash(root, dry_run=False, out=sys.stdout):
    """Rewrite every results file, optimize.csv and finals directory under root; returns (rows rewritten, rows dropped)."""
    data = store_mod.Store(root)
    rewritten = dropped_total = 0
    for path in data.results_files():
        rows = data._read_results(path)
        kept, dropped, renames = rewrite_rows(rows)
        changed = [r for r in kept if r["run_id"] not in {row["run_id"] for row in rows}]
        if not dropped and not changed:
            continue
        out.write("{0}: {1} rewritten, {2} dropped\n".format(os.path.relpath(path, root), len(changed), len(dropped)))
        rewritten += len(changed)
        dropped_total += len(dropped)
        package_dir = _package_dir(path)
        live = {r["finals"] for r in kept if r["finals"]} | set(renames)
        _drop_finals(package_dir, dropped, live, dry_run, out)
        _move_finals(package_dir, renames, dry_run, out)
        if not dry_run:
            with store_mod._Lock(path):
                data._write_results(path, kept)
    named = {(r["key"], r["package"], r["finals"]) for r in data.rows() if r["finals"]}
    for key_dir in sorted(os.listdir(root)):
        if not key_dir.startswith("key="):
            continue
        for package in cubie_adapter.PACKAGES:
            finals_dir = os.path.join(root, key_dir, "package=" + package, "finals")
            for name in sorted(os.listdir(finals_dir)) if os.path.isdir(finals_dir) else ():
                if (key_dir[4:], package, "finals/" + name) not in named:
                    out.write("{0}: unnamed by any row, deleted\n".format(
                        os.path.relpath(os.path.join(finals_dir, name), root)))
                    if not dry_run:
                        os.remove(os.path.join(finals_dir, name))
            path = os.path.join(root, key_dir, "package=" + package, "optimize.csv")
            if not os.path.isfile(path):
                continue
            rows = cubie_adapter._load(path)
            kept, dropped = rewrite_optimize(rows)
            if dropped or kept != rows:
                out.write("{0}: {1} records kept, {2} dropped\n".format(os.path.relpath(path, root), len(kept), len(dropped)))
                if not dry_run:
                    with store_mod._Lock(path):
                        cubie_adapter._save(path, kept)
        exports = os.path.join(root, key_dir, "package=julia_cpu", "controllers")
        if os.path.isdir(exports):
            out.write("{0}: deleted\n".format(os.path.relpath(exports, root)))
            if not dry_run:
                shutil.rmtree(exports)
    return rewritten, dropped_total


def _cli(argv):
    parser = argparse.ArgumentParser(prog="rehash_controllers.py", description=__doc__)
    parser.add_argument("--root", default="data")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    rewritten, dropped = rehash(args.root, dry_run=args.dry_run)
    print("{0} rows rewritten, {1} dropped{2}".format(rewritten, dropped, " (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
