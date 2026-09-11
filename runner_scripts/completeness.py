"""What the store lacks of each trial under one key: transfers rows, the cold build time of a cold line, a readable finals file, a valid record for the line's optimize."""

import math

import cubie_adapter
from abandon import run_ids

MODES = (None, "resume", "no_overwrite")


class Missing:
    """What one trial lacks: `rows` (transfers without a row, or with a NaN time under no_overwrite), `build` (transfers whose row has no cold build time), `finals` (no readable finals file while finals are wanted; rows all NaN want none), `optimize` (the line's optimize record is absent or stale)."""

    def __init__(self, trial, wants_finals):
        self.trial = trial
        self.rows = []
        self.build = []
        self.finals = False
        self.optimize = None
        self.wants_finals = wants_finals

    def complete(self):
        return not (self.rows or self.build or self.finals or self.optimize)

    def transfers(self):
        """The transfers to run again: every one when the optimize record is invalid, else those without a complete row, else the last one alone for the finals."""
        if self.optimize:
            return list(self.trial["transfers"])
        wanted = set(self.rows) | set(self.build)
        if not wanted and self.finals:
            return list(self.trial["transfers"][-1:])
        return [t for t in self.trial["transfers"] if t in wanted]

    def reasons(self):
        """One text per lack: 'row:both', 'build:none', 'finals', 'optimize:<why>'."""
        out = ["row:" + t for t in self.rows] + ["build:" + t for t in self.build]
        if self.finals:
            out.append("finals")
        if self.optimize:
            out.append("optimize:" + self.optimize)
        return out


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def optimize_systems(trial_list):
    """{package: [(problem, system_params, precision)]} of the cubie lines that optimize, the systems whose source hash validates their records."""
    systems = {}
    for trial in trial_list:
        if trial["optimize"] and trial["package"] in cubie_adapter.PACKAGES:
            systems.setdefault(trial["package"], set()).add(cubie_adapter.system_key(trial))
    return {package: sorted(keys) for package, keys in systems.items()}


def optimize_status(trial, key, rows, mode, source):
    """None when the line's optimize.csv record stands, else 'absent', 'timeout' (a lack under no_overwrite alone) or 'source <recorded>' when recorded from another source than `source`."""
    row = cubie_adapter.find_optimized(rows, cubie_adapter.optimize_ident(trial, key))
    if row is None:
        return "absent"
    if row.get("label") == "timeout":
        return "timeout" if mode == "no_overwrite" else None
    if source is not None and row.get("source", "") != source:
        return "source " + (row.get("source", "") or "none")
    return None


def audit(trial_list, key, store, mode=None, sources=None):
    """{trial_id: Missing} of the lines with transfers under a key; 'no_overwrite' wants a finite time, None and 'resume' any row; `sources(package, systems)` gives the current source hash per cubie system, None accepts any recorded source."""
    if mode not in MODES:
        raise ValueError("mode is one of None, resume, no_overwrite")
    recorded = {row["run_id"]: row for row in store.rows(key=key)}
    hashes = {}
    optimize_rows = {}
    if sources is not None:
        for package, systems in optimize_systems(trial_list).items():
            hashes[package] = sources(package, systems)
    out = {}
    for trial in trial_list:
        if not trial["transfers"]:
            continue
        rows = {t: recorded.get(run) for t, run in run_ids(trial, key).items()}
        carried = [r["finals"] for r in rows.values() if r is not None and r["finals"]]
        nan_only = all(r is not None and not _finite(r["min_ms"]) for r in rows.values())
        missing = Missing(trial, bool(carried) or (bool(trial["finals"]) and not nan_only))
        for transfers, row in rows.items():
            if row is None or (mode == "no_overwrite" and not _finite(row["min_ms"])):
                missing.rows.append(transfers)
            elif trial["cold"] and not _finite(row["build_s"]):
                missing.build.append(transfers)
        if missing.wants_finals and not any(store.finals_readable(trial["package"], key, relative)
                                            for relative in carried):
            missing.finals = True
        if trial["optimize"] and trial["package"] in cubie_adapter.PACKAGES:
            package = trial["package"]
            if package not in optimize_rows:
                optimize_rows[package] = cubie_adapter.optimize_rows(package, key, store.root)
            source = hashes.get(package, {}).get(cubie_adapter.system_key(trial))
            missing.optimize = optimize_status(trial, key, optimize_rows[package], mode, source)
        out[trial["trial_id"]] = missing
    return out


def summary(audits):
    """{reason: count} over the incomplete trials of an audit, the optimize reasons by their first word."""
    counts = {}
    for missing in audits.values():
        for reason in missing.reasons():
            word = reason.split(" ")[0]
            counts[word] = counts.get(word, 0) + 1
    return dict(sorted(counts.items()))
