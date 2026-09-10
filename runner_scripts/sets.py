"""Set expansion: a TOML file under sets/ names packages, problems, algorithms, grids, steppings and the optimize step; expand() turns the named sets into run specs, each with its transfers, finals flag, axis, build mode and optimize choice; declarations() expands every set file so a point's specs from all of them can merge. `python sets.py <name>` prints the spec count per package."""

import csv
import math
import os
import sys
import tomllib

from algorithms import algorithm_names, load_algorithms
from problems import load_problems
from protocol import WATCHDOG_SECONDS
from store import PACKAGES, canonical_json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS_DIR = os.path.join(REPO_ROOT, "sets")

NAN = float("nan")
CUBIE_PACKAGES = ("cubie", "cubie_mlir")
GRID_FIELDS = ("parameter", "scale", "min", "max")
SET_KEYS = ("packages", "problems", "algorithms", "precision", "finals", "transfers", "build",
            "optimize", "watchdog")
OPTIMIZE_KEYS = ("packages", "n", "per")
GRID_KEYS = ("packages", "parameter", "scale", "min", "max", "problems", "n", "system_params")
STEPPING_KEYS = ("packages", "algorithms", "controller", "dt", "newton", "tol", "dt0",
                 "dt_min", "dt_max", "gains")
GOLDEN_ALGORITHM = "golden_algorithm"
GOLDEN_TOL = "golden_tol"
# The spec columns a trial carries, in table order, followed by the expansion's own fields.
SPEC_KEYS = ("problem", "system_params", "duration", "precision", "parameter", "grid_scale",
             "grid_min", "grid_max", "n", "grid_dtype", "algorithm", "controller", "dt",
             "dt_min", "dt_max", "atol", "rtol", "gains", "newton_atol", "newton_rtol",
             "package")
EXTRA_KEYS = ("transfers", "finals", "axis", "build", "optimize", "watchdog_s", "set", "stepping")


class SetError(ValueError):
    """A set file that does not follow the schema."""


# ----------------------------------------------------------------- loading

def set_names(sets_dir=SETS_DIR):
    """The shipped set names, sorted."""
    return sorted(os.path.splitext(f)[0] for f in os.listdir(sets_dir) if f.endswith(".toml"))


def set_path(name, sets_dir=SETS_DIR):
    """sets/<name>.toml, or the path itself when it names a .toml file."""
    if name.endswith(".toml") and os.path.isfile(name):
        return name
    path = os.path.join(sets_dir, name + ".toml")
    if not os.path.isfile(path):
        raise SetError("unknown set '{0}' (expected one of: {1})".format(
            name, ", ".join(set_names(sets_dir))))
    return path


def _check_keys(table, allowed, where):
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise SetError("{0}: unknown key(s) {1}".format(where, ", ".join(unknown)))


def _name_list(value, allowed, where, tokens=()):
    """"all", a listed token, or a list of names each in allowed."""
    if value == "all" or value in tokens:
        return value
    if not isinstance(value, list) or not value:
        raise SetError("{0} must be \"all\" or a non-empty list".format(where))
    for name in value:
        if name not in allowed:
            raise SetError("{0}: unknown name '{1}'".format(where, name))
    return list(value)


def load_set(name, sets_dir=SETS_DIR):
    """The validated set table with defaults applied; `name` is a shipped set or a .toml path."""
    path = set_path(name, sets_dir)
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    _check_keys(data, ("set", "grid", "stepping"), path)
    problems = [row.name for row in load_problems()]
    algorithms = algorithm_names()
    head = dict(data.get("set", {}))
    _check_keys(head, SET_KEYS, path + " [set]")
    head.setdefault("packages", list(PACKAGES))
    head.setdefault("problems", "all")
    head.setdefault("algorithms", "all")
    head.setdefault("precision", "float32")
    head.setdefault("finals", False)
    head.setdefault("transfers", ["both", "none"])
    head.setdefault("build", "warm")
    head["packages"] = _name_list(head["packages"], PACKAGES, path + " packages")
    if head["packages"] == "all":
        head["packages"] = list(PACKAGES)
    head["problems"] = _name_list(head["problems"], problems, path + " problems")
    head["algorithms"] = _name_list(head["algorithms"], algorithms, path + " algorithms",
                                    tokens=(GOLDEN_ALGORITHM,))
    if head["precision"] not in ("float32", "float64"):
        raise SetError(path + ": precision must be float32 or float64")
    if not isinstance(head["finals"], bool):
        raise SetError(path + ": finals must be true or false")
    if head["transfers"] not in (["both"], ["none"], ["both", "none"], ["none", "both"]):
        raise SetError(path + ": transfers must list both and/or none")
    if head["build"] not in ("warm", "cold"):
        raise SetError(path + ": build must be warm or cold")
    head.setdefault("watchdog", WATCHDOG_SECONDS)
    watchdog = head["watchdog"]
    if isinstance(watchdog, bool) or not isinstance(watchdog, (int, float)) or not watchdog > 0:
        raise SetError(path + ": watchdog must be a positive number of seconds")
    head["watchdog"] = float(watchdog)
    optimize = head.get("optimize")
    if optimize is not None:
        where = path + " [set.optimize]"
        if not isinstance(optimize, dict):
            raise SetError(where + " must be a table")
        _check_keys(optimize, OPTIMIZE_KEYS, where)
        optimize.setdefault("packages", "all")
        optimize.setdefault("per", "leg")
        optimize["packages"] = _name_list(optimize["packages"], PACKAGES, where + " packages")
        n = optimize.get("n")
        if not (n == "solve" or (isinstance(n, int) and not isinstance(n, bool) and n >= 2)):
            raise SetError(where + ": n must be an integer >= 2 or \"solve\"")
        if optimize["per"] not in ("leg", "solve"):
            raise SetError(where + ": per must be leg or solve")
    head["optimize"] = optimize
    grids = data.get("grid", [])
    steppings = data.get("stepping", [])
    if not grids or not steppings:
        raise SetError(path + ": a set needs at least one [[grid]] and one [[stepping]]")
    for index, grid in enumerate(grids):
        where = "{0} [[grid]] {1}".format(path, index + 1)
        _check_keys(grid, GRID_KEYS, where)
        grid.setdefault("packages", "all")
        grid["packages"] = _name_list(grid["packages"], PACKAGES, where + " packages")
        for field in GRID_FIELDS:
            grid.setdefault(field, "default")
        grid.setdefault("problems", {})
        grid.setdefault("system_params", {})
        if not isinstance(grid.get("n"), list) or not all(isinstance(v, int) and v >= 2 for v in grid["n"]):
            raise SetError(where + ": n must be a list of integers >= 2")
        for problem, overrides in grid["problems"].items():
            if problem not in problems:
                raise SetError("{0}: unknown problem '{1}' in overrides".format(where, problem))
            _check_keys(overrides, GRID_FIELDS, where + " problems." + problem)
    for index, stepping in enumerate(steppings):
        where = "{0} [[stepping]] {1}".format(path, index + 1)
        _check_keys(stepping, STEPPING_KEYS, where)
        stepping.setdefault("packages", "all")
        stepping.setdefault("algorithms", "all")
        stepping["packages"] = _name_list(stepping["packages"], PACKAGES, where + " packages")
        stepping["algorithms"] = _name_list(stepping["algorithms"], algorithms,
                                            where + " algorithms", tokens=(GOLDEN_ALGORITHM,))
        if "controller" not in stepping:
            raise SetError(where + ": controller is required")
        if stepping["controller"] == "fixed":
            if "dt" not in stepping:
                raise SetError(where + ": a fixed stepping needs dt")
            for key in ("tol", "dt0", "dt_min", "dt_max", "gains"):
                if key in stepping:
                    raise SetError("{0}: {1} does not apply to a fixed stepping".format(where, key))
        else:
            if "tol" not in stepping:
                raise SetError(where + ": an adaptive stepping needs tol")
            if "dt" in stepping:
                raise SetError(where + ": dt does not apply to an adaptive stepping; use dt0")
            stepping.setdefault("dt0", "none")
            stepping.setdefault("dt_min", "none")
            stepping.setdefault("dt_max", "none")
            stepping.setdefault("gains", {})
        stepping.setdefault("newton", "none")
    return {"name": os.path.splitext(os.path.basename(path))[0], "path": path,
            "set": head, "grid": grids, "stepping": steppings}


# -------------------------------------------------------------- resolution

def _scaled(value, duration, where):
    """A float, or {duration_times_2_pow = k} / {duration_times = f} scaled by the duration."""
    if isinstance(value, bool):
        raise SetError(where + ": a number is required")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict) and len(value) == 1:
        (key, inner), = value.items()
        if key == "duration_times_2_pow":
            return duration * 2.0 ** float(inner)
        if key == "duration_times":
            return duration * float(inner)
    raise SetError(where + ": expected a number, {duration_times_2_pow = k} or {duration_times = f}")


def _scaled_list(value, duration, where):
    """A list of floats; a scaled form with a list inside yields one value per entry."""
    if isinstance(value, dict) and len(value) == 1:
        (key, inner), = value.items()
        if isinstance(inner, list):
            return [_scaled({key: item}, duration, where) for item in inner]
        return [_scaled(value, duration, where)]
    if isinstance(value, list):
        return [_scaled(item, duration, where) for item in value]
    return [_scaled(value, duration, where)]


def _pin(value, duration, where):
    """A pin: "none" is NaN, anything else a scaled float."""
    if value == "none":
        return NAN
    return _scaled(value, duration, where)


def _grid_fields(grid, problem):
    """(parameter, scale, min, max) of a grid for a problem: defaults from the catalogue, then the per-problem overrides."""
    fields = {"parameter": problem["sweep_parameter"], "scale": problem["sweep_scale"],
              "min": problem["sweep_min"], "max": problem["sweep_max"]}
    for key in GRID_FIELDS:
        if grid[key] != "default":
            fields[key] = grid[key]
    fields.update(grid["problems"].get(problem.name, {}))
    return (str(fields["parameter"]), str(fields["scale"]), float(fields["min"]),
            float(fields["max"]))


def _system_params(grid, problem, where):
    """The construction-parameter objects a grid yields for a problem."""
    params = grid["system_params"]
    if not params:
        return [problem.system_params()]
    if list(params) != ["states"]:
        raise SetError(where + ": system_params takes states only")
    states = params["states"]
    if not isinstance(states, list):
        states = [states]
    try:
        return [problem.system_params(s) for s in states]
    except ValueError as exc:
        raise SetError("{0}: {1}".format(where, exc))


def _axis(grid, stepping):
    """The leg axis a grid and stepping declare: states, then the swept stepping value, else n."""
    if grid["system_params"]:
        return "states"
    if stepping["controller"] == "fixed":
        dt = stepping["dt"]
        count = len(dt.get("duration_times_2_pow", dt.get("duration_times", [None]))) \
            if isinstance(dt, dict) else len(dt) if isinstance(dt, list) else 1
        if count > 1:
            return "dt"
    elif isinstance(stepping["tol"], list) and len(stepping["tol"]) > 1:
        return "tol"
    return "n"


def _narrow(names, requested):
    """names kept by a narrowing list ("all" keeps every one)."""
    if requested == "all":
        return list(names)
    return [name for name in names if name in requested]


class _Algorithm(dict):
    """A golden-algorithm pseudo row: a name with no catalogue family, order or newton capability."""

    @property
    def name(self):
        return self["algorithm"]


def _algorithms(package, kind, head, stepping, problem, catalogue):
    """The package's (package, algorithm) rows with the stepping kind true, narrowed by the set and the stepping."""
    for level in (head["algorithms"], stepping["algorithms"]):
        if level == GOLDEN_ALGORITHM:
            return [_Algorithm(algorithm=problem["golden_algorithm"], family=None, order=None,
                               newton=None)]
    rows = [row for row in catalogue if row.supports(package, kind)]
    for level in (head["algorithms"], stepping["algorithms"]):
        if level != "all":
            rows = [row for row in rows if row.name in level]
    return rows


def controllers_table(key, problem, root="data"):
    """julia_cpu's resolved controller constants for a problem under a key, keyed by algorithm; {} when the file is absent."""
    path = os.path.join(root, "key=" + key, "package=julia_cpu", "controllers", problem + ".csv")
    if not os.path.isfile(path):
        return {}
    table = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row.get("algorithm") or row.get("cubie_alias")
            entry = {"controller": row["controller"]}
            for field in ("beta1", "beta2", "qmin", "qmax", "gamma", "order"):
                raw = row.get(field, "")
                entry[field] = float(raw) if raw not in ("", None) else None
            table[name] = entry
    return table


def _controller(stepping, package, algorithm, problem, key, root, where):
    """(controller, gains) of an adaptive stepping for one algorithm, or None when the entry is skipped."""
    import cubie_adapter
    token = stepping["controller"]
    gains = stepping["gains"]
    if token == "default":
        if gains == "dirk_defaults":
            raise SetError(where + ": dirk_defaults needs controller = \"pi\"")
        return "default", dict(gains)
    if token == "matched" or gains == "dirk_defaults":
        if package not in CUBIE_PACKAGES:
            raise SetError("{0}: {1} applies to the cubie packages only".format(
                where, "matched" if token == "matched" else "dirk_defaults"))
        if algorithm["order"] is None:
            raise SetError(where + ": matched and dirk_defaults need a catalogue algorithm")
        if token == "matched":
            constants = controllers_table(key, problem.name, root).get(algorithm.name)
            settings, _ = cubie_adapter.matched_controller(constants, algorithm["order"])
            if settings is None:
                return None
        else:
            settings = cubie_adapter.pi_tier_controller(algorithm["order"])
            if settings["step_controller"] != token:
                raise SetError(where + ": dirk_defaults needs controller = \"pi\"")
        shipped = cubie_adapter.default_controller(algorithm.name, algorithm["family"],
                                                   algorithm["order"])
        if cubie_adapter.controllers_equal(settings, shipped):
            return None
        settings = dict(settings)
        return settings.pop("step_controller"), settings
    if not isinstance(gains, dict):
        raise SetError(where + ": gains must be a table or \"dirk_defaults\"")
    return token, dict(gains)


def _newton(stepping, algorithm, tol, where):
    """(newton_atol, newton_rtol): NaN unless the (package, algorithm) row has newton = true."""
    newton = stepping["newton"]
    if newton == "none":
        return NAN, NAN
    if algorithm["newton"] is None:
        raise SetError(where + ": newton needs a catalogue algorithm")
    if not algorithm["newton"]:
        return NAN, NAN
    if newton == "tol":
        if tol is None:
            raise SetError(where + ": newton = \"tol\" needs an adaptive stepping")
        return tol, tol
    if isinstance(newton, dict) and set(newton) == {"atol", "rtol"}:
        return float(newton["atol"]), float(newton["rtol"])
    raise SetError(where + ": newton must be \"tol\", \"none\" or {atol, rtol}")


def _steppings(stepping, package, algorithm, problem, key, root, where):
    """The stepping column values, one dict per dt or tolerance."""
    duration = problem["duration"]
    out = []
    if stepping["controller"] == "fixed":
        for dt in _scaled_list(stepping["dt"], duration, where + " dt"):
            atol, rtol = _newton(stepping, algorithm, None, where)
            out.append({"controller": "fixed", "dt": dt, "dt_min": NAN, "dt_max": NAN,
                        "atol": NAN, "rtol": NAN, "gains": canonical_json({}),
                        "newton_atol": atol, "newton_rtol": rtol})
        return out
    resolved = _controller(stepping, package, algorithm, problem, key, root, where)
    if resolved is None:
        return out
    controller, gains = resolved
    tols = stepping["tol"]
    if tols == GOLDEN_TOL:
        tols = [problem["golden_tol"]]
    if not isinstance(tols, list):
        tols = [tols]
    dt0 = _pin(stepping["dt0"], duration, where + " dt0")
    dt_min = _pin(stepping["dt_min"], duration, where + " dt_min")
    dt_max = _pin(stepping["dt_max"], duration, where + " dt_max")
    for tol in tols:
        tol = float(tol)
        atol, rtol = _newton(stepping, algorithm, tol, where)
        out.append({"controller": controller, "dt": dt0, "dt_min": dt_min, "dt_max": dt_max,
                    "atol": tol, "rtol": tol, "gains": canonical_json(gains),
                    "newton_atol": atol, "newton_rtol": rtol})
    return out


# --------------------------------------------------------------- expansion

def _optimize_for(table, package):
    """{n, per} of the set's optimize table when it names the package, else None."""
    if table is None or (table["packages"] != "all" and package not in table["packages"]):
        return None
    return {"n": table["n"], "per": table["per"]}


def expand(names, key, root="data", packages=None, problems=None, algorithms=None, n=None,
           sets_dir=SETS_DIR):
    """Run specs of the named sets in order: the cartesian product of packages, the problems each implements, the algorithms it runs under each stepping kind, grids and steppings. `packages`, `problems`, `algorithms` and `n` narrow; a grid keeps the counts of its n list that `n` names."""
    catalogue = load_algorithms()
    problem_rows = load_problems()
    specs = []
    for name in names:
        loaded = load_set(name, sets_dir)
        head = loaded["set"]
        for gi, grid in enumerate(loaded["grid"]):
            gwhere = "{0} [[grid]] {1}".format(loaded["path"], gi + 1)
            n_list = [count for count in grid["n"] if n is None or count in n]
            for si, stepping in enumerate(loaded["stepping"]):
                swhere = "{0} [[stepping]] {1}".format(loaded["path"], si + 1)
                kind = "fixed" if stepping["controller"] == "fixed" else "adaptive"
                axis = _axis(grid, stepping)
                chosen = _narrow(_narrow(head["packages"], grid["packages"]), stepping["packages"])
                if packages is not None:
                    chosen = [p for p in chosen if p in packages]
                for package in chosen:
                    for problem in problem_rows:
                        if not problem.supports(package):
                            continue
                        if head["problems"] != "all" and problem.name not in head["problems"]:
                            continue
                        if problems is not None and problem.name not in problems:
                            continue
                        parameter, scale, lo, hi = _grid_fields(grid, problem)
                        rows = _algorithms(package, kind, head, stepping, problem, catalogue)
                        if algorithms is not None:
                            rows = [row for row in rows if row.name in algorithms]
                        for params in _system_params(grid, problem, gwhere):
                            for algorithm in rows:
                                for values in _steppings(stepping, package, algorithm, problem,
                                                         key, root, swhere):
                                    for count in n_list:
                                        spec = {
                                            "problem": problem.name,
                                            "system_params": canonical_json(params),
                                            "duration": float(problem["duration"]),
                                            "precision": head["precision"],
                                            "parameter": parameter, "grid_scale": scale,
                                            "grid_min": lo, "grid_max": hi, "n": int(count),
                                            "grid_dtype": "float32",
                                            "algorithm": algorithm.name}
                                        spec.update(values)
                                        spec["package"] = package
                                        spec["transfers"] = list(head["transfers"])
                                        spec["finals"] = bool(head["finals"])
                                        spec["axis"] = axis
                                        spec["build"] = head["build"]
                                        spec["optimize"] = _optimize_for(head["optimize"], package)
                                        spec["watchdog_s"] = head["watchdog"]
                                        spec["set"] = loaded["name"]
                                        spec["stepping"] = stepping["controller"]
                                        specs.append(spec)
    return specs


def declarations(key, root="data", packages=None, problems=None, algorithms=None, n=None,
                 sets_dir=SETS_DIR):
    """The specs of every set file under sets_dir, narrowed like expand(): the declarations a requested point merges with."""
    return expand(set_names(sets_dir), key, root, packages=packages, problems=problems,
                  algorithms=algorithms, n=n, sets_dir=sets_dir)


def declared_counts(names, sets_dir=SETS_DIR):
    """The trajectory counts the grids of the named sets list, sorted."""
    counts = set()
    for name in names:
        for grid in load_set(name, sets_dir)["grid"]:
            counts.update(grid["n"])
    return sorted(counts)


def _close(value, wanted):
    return any(math.isclose(value, w, rel_tol=1e-9, abs_tol=0.0) for w in wanted)


def narrow(specs, mode=None, controllers=None, tols=None, dts=None):
    """Specs kept by the stepping filters: `mode` fixed or adaptive; `controllers` names a spec's controller or the stepping token that produced it; `tols` keeps adaptive specs at those tolerances and `dts` fixed specs at those steps, either one alone dropping the other kind."""
    kept = []
    for spec in specs:
        fixed = spec["controller"] == "fixed"
        if mode == "fixed" and not fixed:
            continue
        if mode == "adaptive" and fixed:
            continue
        if controllers is not None and spec["controller"] not in controllers \
                and spec["stepping"] not in controllers:
            continue
        if tols is not None or dts is not None:
            if fixed:
                if dts is None or not _close(spec["dt"], dts):
                    continue
            elif tols is None or not _close(spec["atol"], tols):
                continue
        kept.append(spec)
    return kept


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: sets.py <set>[,<set>] [key]")
    given = sys.argv[2] if len(sys.argv) > 2 else "plan"
    counts = {}
    for spec in expand(sys.argv[1].split(","), given):
        counts[spec["package"]] = counts.get(spec["package"], 0) + 1
    for package, count in counts.items():
        print("{0} {1}".format(package, count))
