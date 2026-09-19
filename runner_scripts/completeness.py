"""What the store lacks of each trial under one key: transfers rows, the cold build time of a cold line, a readable finals file, a record for the line's optimize. A row abandon_compile wrote for a line before it ran (compile_timeout, NaN, its reason) is no row. A recorded row is never dated or checked against a source: an older timing stands whatever was optimized or changed after it."""

import math

import cubie_adapter
import store as store_mod
from abandon import run_ids

MODES = (None, "resume", "no_overwrite")


class Missing:
    """What one trial lacks: `rows` (transfers without a row, or with a NaN time under no_overwrite), `build` (transfers whose finite row has no cold build time), `finals` (no readable finals file while finals are wanted); rows all NaN want neither, `optimize` (the line's optimize record is absent, or timed out under no_overwrite), `optimize_timed_out` (a timed-out record that stands under resume)."""

    def __init__(self, trial, wants_finals):
        self.trial = trial
        self.rows = []
        self.build = []
        self.finals = False
        self.optimize = None
        self.optimize_timed_out = False
        self.wants_finals = wants_finals

    def complete(self):
        return not (self.rows or self.build or self.finals or self.optimize)

    def transfers(self):
        """The transfers to run again: every one when the optimize record is lacking or a requested output (the build time, the finals) is, so the timing, build time and finals of a trial come from one execution; else those without a complete row."""
        if self.optimize or self.build or self.finals:
            return list(self.trial["transfers"])
        return [t for t in self.trial["transfers"] if t in set(self.rows)]

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


def compile_placeholder(row):
    """True for a row abandon_compile wrote before its line ran: compile_timeout, no time, the compile timeout reason."""
    return row["compile"] == store_mod.COMPILE_TIMEOUT and not _finite(row["min_ms"]) \
        and (row["reason"] or "").startswith(store_mod.COMPILE_TIMEOUT_REASON)


def optimize_status(record, mode):
    """None when a kernel's optimize.csv record stands, else 'absent' or 'timeout' (a lack under no_overwrite alone)."""
    if record is None:
        return "absent"
    if record.get("label") == "timeout":
        return "timeout" if mode == "no_overwrite" else None
    return None


def audit(trial_list, key, store, mode=None):
    """{trial_id: Missing} of the lines with transfers under a key; 'no_overwrite' wants a finite time, None and 'resume' any row."""
    if mode not in MODES:
        raise ValueError("mode is one of None, resume, no_overwrite")
    recorded = {row["run_id"]: row for row in store.rows(key=key)}
    optimize_rows = {}
    out = {}
    for trial in trial_list:
        if not trial["transfers"]:
            continue
        rows = {t: recorded.get(run) for t, run in run_ids(trial, key).items()}
        rows = {t: None if r is not None and compile_placeholder(r) else r for t, r in rows.items()}
        carried = [r["finals"] for r in rows.values() if r is not None and r["finals"]]
        nan_only = all(r is not None and not _finite(r["min_ms"]) for r in rows.values())
        missing = Missing(trial, bool(carried) or (bool(trial["finals"]) and not nan_only))
        for transfers, row in rows.items():
            if row is None or (mode == "no_overwrite" and not _finite(row["min_ms"])):
                missing.rows.append(transfers)
            elif trial["cold"] and _finite(row["min_ms"]) and not _finite(row["build_s"]):
                missing.build.append(transfers)
        if missing.wants_finals and not any(store.finals_readable(trial["package"], key, relative)
                                            for relative in carried):
            missing.finals = True
        if trial["optimize"] and trial["package"] in cubie_adapter.PACKAGES:
            package = trial["package"]
            if package not in optimize_rows:
                optimize_rows[package] = cubie_adapter.optimize_rows(package, key, store.root)
            record = cubie_adapter.find_optimized(optimize_rows[package], cubie_adapter.kernel_ident(trial, key))
            missing.optimize = optimize_status(record, mode)
            missing.optimize_timed_out = record is not None and record.get("label") == "timeout" \
                and missing.optimize is None
        out[trial["trial_id"]] = missing
    return out


def summary(audits):
    """{reason: count} over the incomplete trials of an audit."""
    counts = {}
    for missing in audits.values():
        for reason in missing.reasons():
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))
