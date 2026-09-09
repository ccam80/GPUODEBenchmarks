"""The one comparison of finals: compare() rebuilds both grids, pairs trajectories by exact float32 parameter value and takes the RMS difference over every state of the pairs neither side flags as errored; golden_of() finds the julia_cpu float64 finals row of the same system under any key; error() compares a row with it."""

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

NAN = float("nan")
GOLDEN_PACKAGE = "julia_cpu"
GOLDEN_PRECISION = "float64"
SYSTEM_FIELDS = ("problem", "system_params", "duration")
GOLDEN_WHERE = "package = '{0}' AND precision = '{1}' AND finals <> ''".format(
    GOLDEN_PACKAGE, GOLDEN_PRECISION)


def _has_finals(row):
    return bool(row.get("finals"))


def _same_system(a, b):
    return all(a[f] == b[f] for f in SYSTEM_FIELDS)


def _name(row):
    return "{0}/{1} run {2} (n = {3})".format(row["key"], row["package"], row["run_id"], row["n"])


class Errors:
    """Comparisons over one store; finals files and the golden rows are loaded once."""

    def __init__(self, store):
        self.store = store
        self._finals = {}
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

    def paired(self, a, b):
        """(index_a, index_b) into the two finals of the trajectories with the same float32 grid value that neither row flags."""
        values_a, _, errored_a = self.finals(a)
        values_b, _, errored_b = self.finals(b)
        _, index_a, index_b = np.intersect1d(values_a, values_b, return_indices=True)
        keep = ~(errored_a[index_a] | errored_b[index_b])
        return index_a[keep], index_b[keep]

    def compare(self, a, b):
        """RMS over every state, in float64, of the difference between the finals of two rows over the trajectories paired by exact float32 grid value that neither row flags; NaN when none pair."""
        _, states_a, _ = self.finals(a)
        _, states_b, _ = self.finals(b)
        if states_a.shape[1] != states_b.shape[1]:
            raise ValueError("{0} has {1} states, {2} has {3}".format(
                _name(a), states_a.shape[1], _name(b), states_b.shape[1]))
        index_a, index_b = self.paired(a, b)
        if index_a.shape[0] == 0:
            return NAN
        diff = states_a[index_a] - states_b[index_b]
        return float(np.sqrt(np.mean(diff * diff)))

    def goldens(self):
        """Every julia_cpu float64 row with finals, under any key."""
        if self._goldens is None:
            self._goldens = self.store.rows(sql_where=GOLDEN_WHERE)
        return self._goldens

    def golden_of(self, row):
        """The golden row of the same problem, system_params and duration; None when absent; raises naming the rows when more than one exists."""
        matches = [g for g in self.goldens() if _same_system(g, row)]
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError("more than one golden for {0} {1} duration {2}: {3}".format(
                row["problem"], row["system_params"], row["duration"],
                "; ".join(_name(g) for g in matches)))
        return matches[0]

    def error(self, row):
        """compare(row, golden_of(row)); NaN when the row has no finals or no golden exists."""
        if not _has_finals(row):
            return NAN
        golden = self.golden_of(row)
        if golden is None:
            return NAN
        return self.compare(row, golden)


def compare(a, b, store):
    return Errors(store).compare(a, b)


def golden_of(row, store):
    return Errors(store).golden_of(row)


def error(row, store):
    return Errors(store).error(row)


def is_finite_positive(value):
    return isinstance(value, float) and math.isfinite(value) and value > 0.0
