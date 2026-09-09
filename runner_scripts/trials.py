"""Trials: the unit of work a runner executes (docs/unification-plan.md 1.3), the --for view expansion of 1.5, the identity filters, and the JSONL a runner reads."""

import csv
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field, replace

import store
from algorithms import MODES, get_algorithm, load_algorithms, ne_member
from cubie_adapter import PACKAGES as CUBIE_PACKAGES
from cubie_adapter import controllers_equal
from cubie_adapter import matched_controller as _matched_controller
from problems import STATES_PROBLEM, get_problem, load_problems, states_row
from protocol import (N_NE, N_WP, NMAX_DEFAULT, OPTIMIZE_N,
                      OPTIMIZE_PER_POINT_FAMILIES, OVERLAP_TOL, STATES_GRID,
                      STATES_N, TIMING_TOL, TOLS, parse_ns)

KINDS = ("solve", "warm", "optimize")
GRIDS = ("sweep", "prefix")
AXES = ("n", "setting", "states")
VIEWS = ("perf", "wp", "ne", "states", "overlap")
TRANSFERS = ("both", "none")
TIERS = ("default", "matched", "pi")

NE_PACKAGES = ("cubie", "cubie_mlir", "julia_cpu")
OVERLAP_PACKAGES = ("cubie", "cubie_mlir", "julia_gpu")
# Packages whose legs compile something worth warming; states legs are never warmed.
COMPILING_PACKAGES = ("cubie", "cubie_mlir", "jax", "cpp", "myokit_cuda", "julia_gpu")
FINALS_N = 32768
FINALS_NE = N_NE

NAN = float("nan")


@dataclass
class Trial:
    """One line of a trials file; `leg` and `id` derive from the fields."""

    package: str
    problem: str
    algorithm: str
    mode: str
    setting_kind: str
    setting: float
    n: int
    states: int
    tier: str = "default"
    kind: str = "solve"
    transfers: list = field(default_factory=list)
    grid: str = "sweep"
    finals: int = 0
    controller: dict = None
    axis: str = "n"
    ordinal: int = 0

    @property
    def leg(self):
        return "/".join((self.problem, self.algorithm, self.mode, self.axis))

    @property
    def id(self):
        base = "{0}/{1}/{2}/{3}/{4}={5}/n={6}/s={7}/{8}".format(
            self.package, self.problem, self.algorithm, self.mode, self.setting_kind,
            store.format_setting(self.setting), self.n, self.states, self.tier)
        return base if self.kind == "solve" else self.kind + "/" + base

    @property
    def identity_key(self):
        """The store identity without transfers, plus the kind; two trials with the same key are one trial."""
        return (self.kind, self.package, self.problem, self.algorithm, self.mode,
                self.setting_kind, store.format_setting(self.setting), self.n,
                self.states, self.tier)

    @property
    def leg_key(self):
        return (self.package, self.leg)

    def identity(self, key, transfers=None):
        """The store identity columns of this trial under a dataset key; transfers included when given."""
        ident = {"package": self.package, "key": key, "problem": self.problem,
                 "algorithm": self.algorithm, "mode": self.mode,
                 "setting_kind": self.setting_kind, "setting": float(self.setting),
                 "n": int(self.n), "states": int(self.states), "tier": self.tier}
        if transfers is not None:
            ident["transfers"] = transfers
        return ident

    def to_json(self):
        return OrderedDict([
            ("id", self.id), ("kind", self.kind), ("package", self.package),
            ("problem", self.problem), ("algorithm", self.algorithm), ("mode", self.mode),
            ("setting_kind", self.setting_kind), ("setting", float(self.setting)),
            ("n", int(self.n)), ("states", int(self.states)), ("tier", self.tier),
            ("transfers", list(self.transfers)), ("grid", self.grid),
            ("finals", int(self.finals)), ("controller", self.controller),
            ("leg", self.leg), ("ordinal", int(self.ordinal))])

    @classmethod
    def from_json(cls, record):
        axis = record["leg"].split("/")[-1]
        if axis not in AXES:
            raise ValueError("leg '{0}' names no axis".format(record["leg"]))
        trial = cls(package=record["package"], problem=record["problem"],
                    algorithm=record["algorithm"], mode=record["mode"],
                    setting_kind=record["setting_kind"], setting=float(record["setting"]),
                    n=int(record["n"]), states=int(record["states"]), tier=record["tier"],
                    kind=record["kind"], transfers=list(record["transfers"]),
                    grid=record["grid"], finals=int(record["finals"]),
                    controller=record.get("controller"), axis=axis,
                    ordinal=int(record["ordinal"]))
        _validate(trial)
        return trial


def _validate(trial):
    checks = (("kind", KINDS), ("grid", GRIDS), ("axis", AXES), ("tier", TIERS),
              ("mode", MODES), ("package", store.PACKAGES))
    for name, allowed in checks:
        if getattr(trial, name) not in allowed:
            raise ValueError("{0} '{1}' is not one of {2}".format(
                name, getattr(trial, name), ", ".join(allowed)))
    for transfers in trial.transfers:
        if transfers not in TRANSFERS:
            raise ValueError("transfers '{0}' is not one of {1}".format(
                transfers, ", ".join(TRANSFERS)))


# --------------------------------------------------------------------- JSONL

def write_jsonl(path, trials):
    """One trial per line, warm then optimize then solve within each leg, legs in order of first appearance."""
    order = {"warm": 0, "optimize": 1, "solve": 2}
    legs = list(OrderedDict.fromkeys(t.leg_key for t in trials))
    ranked = sorted(trials, key=lambda t: (legs.index(t.leg_key), order[t.kind], t.ordinal, t.id))
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for trial in ranked:
            handle.write(json.dumps(trial.to_json()) + "\n")
    return ranked


def read_jsonl(path):
    trials = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                trials.append(Trial.from_json(json.loads(line)))
    return trials


# ------------------------------------------------------------------ ordering

def cost_key(trial):
    """Ascending cost within a leg: N ascending, dt descending, tol descending, states ascending; tiers after each other at equal cost."""
    setting_cost = -float(trial.setting)
    tier = TIERS.index(trial.tier)
    if trial.axis == "n":
        return (trial.n, setting_cost, trial.states, tier)
    if trial.axis == "setting":
        return (setting_cost, trial.n, trial.states, tier)
    return (trial.states, trial.n, setting_cost, tier)


def assign_ordinals(trials):
    """Number the solve trials of every (package, leg) in cost order; warm and optimize trials take the ordinal of the solve trial they serve."""
    legs = {}
    for trial in trials:
        if trial.kind == "solve":
            legs.setdefault(trial.leg_key, []).append(trial)
    tuned = {}
    for members in legs.values():
        members.sort(key=cost_key)
        for ordinal, trial in enumerate(members):
            trial.ordinal = ordinal
            point = (trial.package, trial.leg, trial.setting_kind,
                     store.format_setting(trial.setting), trial.states)
            tuned.setdefault(point, ordinal)
    for trial in trials:
        if trial.kind == "warm":
            trial.ordinal = 0
        elif trial.kind == "optimize":
            trial.ordinal = tuned.get((trial.package, trial.leg, trial.setting_kind,
                                       store.format_setting(trial.setting), trial.states), 0)
    return trials


# ------------------------------------------------------------------- merging

def merge(trials):
    """Deduplicate by identity: transfers union, the larger finals, prefix over sweep; the first trial keeps its leg and controller."""
    merged = OrderedDict()
    for trial in trials:
        key = trial.identity_key
        standing = merged.get(key)
        if standing is None:
            merged[key] = replace(trial, transfers=list(trial.transfers))
            continue
        standing.transfers = [t for t in TRANSFERS if t in standing.transfers or t in trial.transfers]
        standing.finals = max(standing.finals, trial.finals)
        if trial.grid == "prefix":
            standing.grid = "prefix"
    return list(merged.values())


# ----------------------------------------------------------------- controllers

def shipped_controller(row):
    """Cubie's shipped controller for an algorithm row; imports cubie."""
    import cubie_adapter
    return cubie_adapter.default_controller(row.name, row["family"], row["order"])


def pi_controller(row):
    """The overlap suite's pi tier for an algorithm row; imports cubie."""
    import cubie_adapter
    return cubie_adapter.pi_tier_controller(row["order"])


def read_controller_constants(store_root, key, problem):
    """{cubie_alias: constants} from controllers/<problem>.csv of julia_cpu under the key; {} when absent."""
    path = os.path.join(store.Store(store_root).package_dir("julia_cpu", key),
                        "controllers", problem + ".csv")
    if not os.path.isfile(path):
        return {}
    constants = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            entry = {"controller": record["controller"]}
            for name in ("beta1", "beta2", "qmin", "qmax", "gamma"):
                text = (record.get(name) or "").strip()
                entry[name] = float(text) if text else None
            entry["order"] = int(float(record["order"])) if record.get("order") else None
            constants[record["cubie_alias"]] = entry
    return constants


def adaptive_tiers(package, row, view, constants):
    """[(tier, controller)] of an adaptive trial: default, plus matched (ne) or pi (overlap) for cubie packages when the controller differs from the shipped one."""
    tiers = [("default", None)]
    if package not in CUBIE_PACKAGES:
        return tiers
    if view == "ne" and row.name in constants:
        matched = _matched_controller(constants[row.name], row["order"])[0]
        if matched is not None and not controllers_equal(matched, shipped_controller(row)):
            tiers.append(("matched", matched))
    if view == "overlap":
        pi = pi_controller(row)
        if not controllers_equal(pi, shipped_controller(row)):
            tiers.append(("pi", pi))
    return tiers


# ----------------------------------------------------------------- expansion

def timing_setting(problem, mode):
    """(setting_kind, setting) of the N and states sweeps."""
    if mode == "fixed":
        return "dt", problem.timing_dt
    return "tol", TIMING_TOL


def _selected(rows, names, label):
    if names in (None, "all"):
        return rows
    wanted = [name for name in names.split(",") if name] if isinstance(names, str) else list(names)
    known = {row.name: row for row in rows}
    for name in wanted:
        if name not in known:
            raise SystemExit("unknown {0} '{1}'".format(label, name))
    return [known[name] for name in wanted]


class Request:
    """What to expand: the axes, the views, the N list and the store the tiers read."""

    def __init__(self, packages, problems="all", algorithms="all", modes=MODES,
                 views=VIEWS, nlist=None, states_grid=STATES_GRID, key="",
                 store_root="data"):
        for package in packages:
            if package not in store.PACKAGES:
                raise SystemExit("unknown package '{0}' (expected one of {1})".format(
                    package, ", ".join(store.PACKAGES)))
        self.packages = list(packages)
        self.problems = _selected(load_problems(), problems, "problem")
        self.algorithms = _selected(load_algorithms(), algorithms, "algorithm")
        self.modes = tuple(modes)
        for view in views:
            if view not in VIEWS:
                raise SystemExit("unknown view '{0}' (expected one of {1})".format(
                    view, ", ".join(VIEWS)))
        self.views = tuple(v for v in VIEWS if v in views)
        self.nlist = sorted(nlist) if nlist is not None else parse_ns(str(NMAX_DEFAULT))
        self.states_grid = tuple(sorted(states_grid))
        self.key = key
        self.store_root = store_root
        self._constants = {}

    def constants(self, problem):
        if problem not in self._constants:
            self._constants[problem] = read_controller_constants(self.store_root, self.key, problem)
        return self._constants[problem]


def _solve(package, problem, row, mode, kind, setting, n, states, axis, grid,
           transfers, finals, tier="default", controller=None):
    return Trial(package=package, problem=problem.name, algorithm=row.name, mode=mode,
                 setting_kind=kind, setting=float(setting), n=int(n), states=int(states),
                 tier=tier, kind="solve", transfers=list(transfers), grid=grid,
                 finals=int(finals), controller=controller, axis=axis)


def _members(request, view):
    """(package, problem, algorithm row, mode) the view runs."""
    for package in request.packages:
        if view == "ne" and package not in NE_PACKAGES:
            continue
        if view == "overlap" and package not in OVERLAP_PACKAGES:
            continue
        problems = request.problems
        if view == "states":
            problems = [p for p in problems if p.name == STATES_PROBLEM]
        for problem in problems:
            if not problem.supports(package):
                continue
            for row in request.algorithms:
                for mode in request.modes:
                    if view == "ne":
                        member = ne_member(row, mode)
                    elif view == "overlap":
                        member = bool(row["julia_gpu"])
                    else:
                        member = row.supports(package, mode)
                    if member:
                        yield package, problem, row, mode


def _wp_settings(problem, row, mode):
    return list(TOLS) if mode == "adaptive" else problem.dts(row.name)


def _ne_settings(problem, mode):
    return list(TOLS) if mode == "adaptive" else problem.ne_dts()


def expand_view(request, view):
    """The solve trials of one view, before merging."""
    trials = []
    for package, problem, row, mode in _members(request, view):
        if view == "perf":
            kind, setting = timing_setting(problem, mode)
            for n in request.nlist:
                trials.append(_solve(package, problem, row, mode, kind, setting, n,
                                     problem["states"], "n", "sweep", TRANSFERS,
                                     FINALS_N if n == FINALS_N else 0))
        elif view == "wp":
            kind = "tol" if mode == "adaptive" else "dt"
            finals = FINALS_NE if package in CUBIE_PACKAGES and ne_member(row, mode) else 0
            for setting in _wp_settings(problem, row, mode):
                trials.append(_solve(package, problem, row, mode, kind, setting, N_WP,
                                     problem["states"], "setting", "prefix", ("none",),
                                     finals))
        elif view == "ne":
            kind = "tol" if mode == "adaptive" else "dt"
            n = N_NE if package == "julia_cpu" else N_WP
            tiers = ([("default", None)] if mode == "fixed"
                     else adaptive_tiers(package, row, view, request.constants(problem.name)))
            for tier, controller in tiers:
                for setting in _ne_settings(problem, mode):
                    trials.append(_solve(package, problem, row, mode, kind, setting, n,
                                         problem["states"], "setting", "prefix", ("none",),
                                         FINALS_NE, tier, controller))
        elif view == "states":
            kind, setting = timing_setting(problem, mode)
            for states in request.states_grid:
                trials.append(_solve(package, states_row(states), row, mode, kind, setting,
                                     STATES_N, states, "states", "sweep", TRANSFERS, 0))
        elif view == "overlap":
            kind = "tol" if mode == "adaptive" else "dt"
            tiers = ([("default", None)] if mode == "fixed"
                     else adaptive_tiers(package, row, view, request.constants(problem.name)))
            sweep_setting = OVERLAP_TOL if mode == "adaptive" else problem.timing_dt
            for tier, controller in tiers:
                for n in request.nlist:
                    trials.append(_solve(package, problem, row, mode, kind, sweep_setting, n,
                                         problem["states"], "n", "sweep", TRANSFERS, 0,
                                         tier, controller))
                for setting in _wp_settings(problem, row, mode):
                    trials.append(_solve(package, problem, row, mode, kind, setting, N_WP,
                                         problem["states"], "setting", "prefix", TRANSFERS,
                                         0, tier, controller))
                if package == "julia_gpu" and ne_member(row, mode):
                    for setting in _ne_settings(problem, mode):
                        trials.append(_solve(package, problem, row, mode, kind, setting,
                                             N_NE, problem["states"], "setting", "prefix",
                                             TRANSFERS, FINALS_NE, tier, controller))
    return trials


def per_point(row):
    return row["family"] in OPTIMIZE_PER_POINT_FAMILIES


def service_trials(solves):
    """The warm and optimize trials the solve trials need: one warm per compiling leg off the states axis, cubie optimize per [optimize]; an optimize point shared by legs is one trial."""
    extra = []
    by_leg = OrderedDict()
    for trial in solves:
        by_leg.setdefault(trial.leg_key, []).append(trial)
    optimized = set()
    for (package, _), members in by_leg.items():
        first = min(members, key=cost_key)
        if package in COMPILING_PACKAGES and first.axis != "states":
            extra.append(replace(first, kind="warm", transfers=[], finals=0, controller=None))
        if package in CUBIE_PACKAGES:
            row = get_algorithm(first.algorithm)
            problem = get_problem(first.problem)
            for trial in sorted(members, key=cost_key):
                if per_point(row):
                    kind, setting = trial.setting_kind, trial.setting
                else:
                    kind, setting = timing_setting(problem, trial.mode)
                optimize = replace(trial, kind="optimize", setting_kind=kind,
                                   setting=float(setting), n=OPTIMIZE_N, tier="default",
                                   transfers=[], finals=0, controller=None)
                if optimize.identity_key in optimized:
                    continue
                optimized.add(optimize.identity_key)
                extra.append(optimize)
    return extra


def expand(request):
    """Every trial of the request's views: merged solve trials with ordinals, plus their warm and optimize trials."""
    solves = []
    for view in request.views:
        solves += expand_view(request, view)
    solves = merge(solves)
    trials = solves + service_trials(solves)
    return assign_ordinals(trials)


# ------------------------------------------------------------------- filters

class StoreIndex:
    """Row status per trial and transfers from one DuckDB read per package."""

    def __init__(self, store_root, key):
        self.store = store.Store(store_root)
        self.key = key
        self._rows = {}

    def _package_rows(self, package):
        if package not in self._rows:
            index = {}
            for row in self.store.rows(package=package, key=self.key):
                point = (row["problem"], row["algorithm"], row["mode"], row["setting_kind"],
                         int(row["n"]), int(row["states"]), row["tier"], row["transfers"])
                index.setdefault(point, []).append((float(row["setting"]), row["min_ms"]))
            self._rows[package] = index
        return self._rows[package]

    def status(self, trial, transfers):
        """'absent', 'nan' or 'finite' for the row of a trial's transfers leg."""
        point = (trial.problem, trial.algorithm, trial.mode, trial.setting_kind,
                 int(trial.n), int(trial.states), trial.tier, transfers)
        matched = [ms for setting, ms in self._package_rows(trial.package).get(point, [])
                   if store.setting_matches(setting, trial.setting)]
        if not matched:
            return "absent"
        import math
        if any(ms is not None and math.isfinite(ms) for ms in matched):
            return "finite"
        return "nan"


def point_matches(trial, point):
    """True when a --point names the trial: its id, or a leading path of it."""
    return trial.id == point or trial.id.startswith(point.rstrip("/") + "/")


def keep_services(trials):
    """Warm and optimize trials only for legs that still have solve trials."""
    live = {t.leg_key for t in trials if t.kind == "solve"}
    return [t for t in trials if t.kind == "solve" or t.leg_key in live]


def apply_filters(trials, tiers=None, transfers=None, settings=None, points=(),
                  resume=False, no_overwrite=False, index=None):
    """Narrow the solve trials by tier, transfers, setting and --point, then by the store under --resume or --no-overwrite; services follow their legs."""
    kept = []
    for trial in trials:
        if trial.kind != "solve":
            kept.append(trial)
            continue
        if tiers is not None and trial.tier not in tiers:
            continue
        if transfers is not None:
            trial.transfers = [t for t in trial.transfers if t in transfers]
            if not trial.transfers:
                continue
        if settings is not None and not any(store.setting_matches(trial.setting, s) for s in settings):
            continue
        if points and not any(point_matches(trial, p) for p in points):
            continue
        if resume or no_overwrite:
            statuses = [index.status(trial, t) for t in trial.transfers]
            if resume and all(s != "absent" for s in statuses):
                continue
            if no_overwrite and all(s == "finite" for s in statuses):
                continue
        kept.append(trial)
    return keep_services(kept)


def counts(trials):
    """{package: {"warm": w, "optimize": o, "solve": s, "legs": {leg: solve count}}}."""
    summary = OrderedDict()
    for trial in trials:
        entry = summary.setdefault(trial.package, {"warm": 0, "optimize": 0, "solve": 0,
                                                   "legs": OrderedDict()})
        entry[trial.kind] += 1
        if trial.kind == "solve":
            entry["legs"][trial.leg] = entry["legs"].get(trial.leg, 0) + 1
    return summary
