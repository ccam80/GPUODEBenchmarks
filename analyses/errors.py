"""The one comparison of a run with its golden: compare() rebuilds both grids, pairs trajectories by exact float32 parameter value and takes the RMS difference over every state of the pairs neither side flags as errored, over the finals, or over every sample of the traces when both rows carry them (a trajectory with a non-finite sample is errored); golden_of() finds the julia_cpu float64 row with finals or traces of the same system under any key, the one running the catalogue's golden algorithm when rows of other algorithms stand beside it; error() compares a row with it; interval_error() is the mean absolute inter-beat interval error of the traced first state against the golden's in ms, trace_nan_pct() the percent of traced trajectories with a NaN."""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER_SCRIPTS = os.path.join(os.path.dirname(HERE), "runner_scripts")
if RUNNER_SCRIPTS not in sys.path:
    sys.path.insert(0, RUNNER_SCRIPTS)

import grid  # noqa: E402
import store as store_mod  # noqa: E402
from problems import load_problems  # noqa: E402
from protocol import TRACE_SPAN_S  # noqa: E402

NAN = float("nan")
GOLDEN_PACKAGE = "julia_cpu"
GOLDEN_PRECISION = "float64"
SYSTEM_FIELDS = ("problem", "system_params", "duration")
# A peak of the traced first state must rise above this (0 mV for a membrane voltage).
PEAK_THRESHOLD = 0.0
GOLDEN_WHERE = "package = '{0}' AND precision = '{1}' AND (finals <> '' OR traces <> '')".format(
    GOLDEN_PACKAGE, GOLDEN_PRECISION)


def _has_finals(row):
    return bool(row.get("finals"))


def _has_traces(row):
    return bool(row.get("traces"))


def _same_system(a, b):
    return all(a[f] == b[f] for f in SYSTEM_FIELDS)


def _name(row):
    return "{0}/{1} run {2} (n = {3})".format(row["key"], row["package"], row["run_id"], row["n"])


class Errors:
    """Comparisons over one store; finals files and the golden rows are loaded once."""

    def __init__(self, store):
        self.store = store
        self._finals = {}
        self._traces = {}
        self._goldens = None

    def finals(self, row):
        """(grid values float32[m], states float64[m, k], errored bool[m]) of a row's finals, indexed by trajectory."""
        relative = row.get("finals") or ""
        if not relative:
            raise ValueError("no finals on " + _name(row))
        ident = (row["package"], row["key"], relative)
        if ident not in self._finals:
            traj, states, t_final, retcode = self.store.load_finals(*ident)
            values = grid.grid_values(row["grid_scale"], row["grid_min"], row["grid_max"],
                                      row["n"], row.get("grid_dtype", "float32"))
            errored = store_mod.errored_mask(states, t_final, retcode, row["duration"])
            self._finals[ident] = (values[traj], states.astype(np.float64), errored)
        return self._finals[ident]

    def traces(self, row):
        """(grid values float32[m], states float64[m, samples, k], errored bool[m]) of a row's traces, indexed by trajectory; a trajectory with a non-finite sample is errored."""
        relative = row.get("traces") or ""
        if not relative:
            raise ValueError("no traces on " + _name(row))
        ident = (row["package"], row["key"], relative)
        if ident not in self._traces:
            traj, _, states = self.store.load_traces(*ident)
            values = grid.grid_values(row["grid_scale"], row["grid_min"], row["grid_max"],
                                      row["n"], row.get("grid_dtype", "float32"))
            errored = ~np.isfinite(states).all(axis=(1, 2))
            self._traces[ident] = (values[traj], states.astype(np.float64), errored)
        return self._traces[ident]

    def _kept(self, row, traced):
        return self.traces(row) if traced else self.finals(row)

    def paired(self, a, b, traced=False):
        """(index_a, index_b) into the two finals (or traces) of the trajectories with the same float32 grid value that neither row flags."""
        values_a, _, errored_a = self._kept(a, traced)
        values_b, _, errored_b = self._kept(b, traced)
        _, index_a, index_b = np.intersect1d(values_a, values_b, return_indices=True)
        keep = ~(errored_a[index_a] | errored_b[index_b])
        return index_a[keep], index_b[keep]

    def compare(self, a, b):
        """RMS over every state, in float64, of the difference between the finals of two rows (every sample of their traces when both carry them) over the trajectories paired by exact float32 grid value that neither row flags; NaN when none pair."""
        traced = _has_traces(a) and _has_traces(b)
        _, states_a, _ = self._kept(a, traced)
        _, states_b, _ = self._kept(b, traced)
        if states_a.shape[1:] != states_b.shape[1:]:
            raise ValueError("{0} has states {1}, {2} has {3}".format(
                _name(a), states_a.shape[1:], _name(b), states_b.shape[1:]))
        index_a, index_b = self.paired(a, b, traced)
        if index_a.shape[0] == 0:
            return NAN
        diff = states_a[index_a] - states_b[index_b]
        return float(np.sqrt(np.mean(diff * diff)))

    def goldens(self):
        """Every julia_cpu float64 row with finals or traces, under any key."""
        if self._goldens is None:
            self._goldens = self.store.rows(sql_where=GOLDEN_WHERE)
        return self._goldens

    def golden_of(self, row):
        """The golden row of the same problem, system_params and duration; None when absent; among rows of several algorithms those of the catalogue's golden algorithm stand; raises naming the rows when more than one remains."""
        matches = [g for g in self.goldens() if _same_system(g, row)]
        if not matches:
            return None
        preferred = [g for g in matches if g["algorithm"] == golden_algorithm(row["problem"])]
        if preferred:
            matches = preferred
        if len(matches) > 1:
            raise ValueError("more than one golden for {0} {1} duration {2}: {3}".format(
                row["problem"], row["system_params"], row["duration"],
                "; ".join(_name(g) for g in matches)))
        return matches[0]

    def peak_times(self, row):
        """(grid values float32[m], [peak times float64 per trajectory], errored bool[m]) of a row's traces: the strict local maxima of the first state above PEAK_THRESHOLD, each refined by a parabola through its three samples; a sample at either end of a trace is never a peak; an errored trajectory has none."""
        values, states, errored = self.traces(row)
        times = store_mod.trace_times()
        peaks = [_peaks(times, states[traj, :, 0]) if not errored[traj] else np.zeros(0)
                 for traj in range(values.shape[0])]
        return values, peaks, errored

    def trace_nan_pct(self, row):
        """Percent of a row's traced trajectories with a non-finite sample; NaN without traces."""
        if not _has_traces(row):
            return NAN
        _, _, errored = self.traces(row)
        return 100.0 * float(errored.mean()) if errored.shape[0] else NAN

    def interval_error(self, row):
        """Mean absolute inter-beat interval error in ms: over the trajectories paired by grid value that neither the row nor its golden flags, the intervals between the trace start, each peak and the trace end are paired in order and their differences summed, an interval without a partner counting in full; the total is divided by the golden's interval count. NaN without traces on the row or with no pair; raises when the row has traces and its golden has none."""
        if not _has_traces(row):
            return NAN
        golden = self.golden_of(row)
        if golden is None or not _has_traces(golden):
            raise ValueError("no traced golden for " + _name(row))
        index_a, index_b = self.paired(row, golden, traced=True)
        _, peaks_a, _ = self.peak_times(row)
        _, peaks_b, _ = self.peak_times(golden)
        total, count = 0.0, 0
        for a, b in zip(index_a, index_b):
            run, wanted = _intervals(peaks_a[a]), _intervals(peaks_b[b])
            shared = min(run.shape[0], wanted.shape[0])
            total += float(np.abs(run[:shared] - wanted[:shared]).sum() + run[shared:].sum() + wanted[shared:].sum())
            count += wanted.shape[0]
        return 1000.0 * total / count if count else NAN

    def error(self, row):
        """compare(row, golden_of(row)); NaN when the row has neither finals nor traces, or no golden exists, or the two hold nothing in common."""
        if not (_has_finals(row) or _has_traces(row)):
            return NAN
        golden = self.golden_of(row)
        if golden is None:
            return NAN
        traced = _has_traces(row) and _has_traces(golden)
        if not traced and not (_has_finals(row) and _has_finals(golden)):
            return NAN
        return self.compare(row, golden)


def _peaks(times, signal):
    """Times of the strict local maxima of a sampled signal above PEAK_THRESHOLD, each refined by the parabola through its three samples."""
    inner = signal[1:-1]
    where = np.where((inner > signal[:-2]) & (inner >= signal[2:]) & (inner > PEAK_THRESHOLD))[0] + 1
    out = []
    for i in where:
        left, mid, right = float(signal[i - 1]), float(signal[i]), float(signal[i + 1])
        curve = left - 2.0 * mid + right
        shift = 0.5 * (left - right) / curve if curve != 0.0 else 0.0
        out.append(times[i] + shift * (times[1] - times[0]))
    return np.asarray(out, dtype=np.float64)


def _intervals(peaks):
    """The intervals between the trace start, each peak and the trace end, in seconds."""
    return np.diff(np.concatenate(([0.0], peaks, [TRACE_SPAN_S])))


def golden_algorithm(problem):
    """The catalogue's golden algorithm of a problem; None for a problem it does not list."""
    for entry in load_problems():
        if entry["problem"] == problem:
            return entry["golden_algorithm"]
    return None


def compare(a, b, store):
    return Errors(store).compare(a, b)


def golden_of(row, store):
    return Errors(store).golden_of(row)


def error(row, store):
    return Errors(store).error(row)


def is_finite_positive(value):
    return isinstance(value, float) and math.isfinite(value) and value > 0.0
