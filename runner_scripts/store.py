"""The result store: data/key=<os>_<gpu>/package=<pkg>/results/<problem>__<algorithm>.parquet per (problem, algorithm), finals/<trial_id>.parquet beside it, DuckDB over the tree; a row is its run spec, hashed to run_id (the replace key), trial_id and group_id. CLI: store.py [--root DIR] record <rows.json|-> [--floor] | finals <spec.json> <finals.csv> | status <run_id> | query "<sql over results>" | clear <filter.json> | hash <spec.json>."""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PACKAGES = ("cubie", "cubie_mlir", "jax", "pytorch", "myokit_cuda", "cpp",
            "julia_gpu", "julia_cpu")
PRECISIONS = ("float32", "float64")
GRID_SCALES = ("linear", "log")
GRID_DTYPES = ("float32",)
TRANSFERS = ("both", "none")

# The run spec in table order; the type drives validation and the hash text.
SPEC_TYPES = (
    ("problem", "name"), ("system_params", "json"), ("duration", "float"),
    ("precision", "str"),
    ("parameter", "str"), ("grid_scale", "str"), ("grid_min", "float"),
    ("grid_max", "float"), ("n", "int"), ("grid_dtype", "str"),
    ("algorithm", "name"), ("controller", "str"), ("dt", "float"),
    ("dt_min", "float"), ("dt_max", "float"), ("atol", "float"),
    ("rtol", "float"), ("gains", "json"), ("newton_atol", "float"),
    ("newton_rtol", "float"),
    ("transfers", "str"), ("package", "str"), ("key", "name"),
)
SPEC_FIELDS = tuple(name for name, _ in SPEC_TYPES)
TRIAL_FIELDS = tuple(f for f in SPEC_FIELDS if f not in ("transfers", "key"))
FINALS_FIELDS = TRIAL_FIELDS + ("key",)
ENSEMBLE_FIELDS = ("parameter", "grid_scale", "grid_min", "grid_max", "n", "grid_dtype")
GROUP_FIELDS = tuple(f for f in TRIAL_FIELDS
                     if f not in ENSEMBLE_FIELDS and f != "package")
ID_FIELDS = {"trial_id": TRIAL_FIELDS, "run_id": SPEC_FIELDS, "group_id": GROUP_FIELDS}
_ENUMS = {"precision": PRECISIONS, "grid_scale": GRID_SCALES,
          "grid_dtype": GRID_DTYPES, "transfers": TRANSFERS, "package": PACKAGES}

_ARROW = {"name": pa.string(), "str": pa.string(), "json": pa.string(),
          "float": pa.float64(), "int": pa.int64()}
SCHEMA = pa.schema([(name, _ARROW[kind]) for name, kind in SPEC_TYPES] + [
    ("run_id", pa.string()), ("trial_id", pa.string()), ("group_id", pa.string()),
    ("states", pa.int32()), ("min_ms", pa.float64()),
    ("samples_ms", pa.list_(pa.float64())), ("errored_pct", pa.float64()),
    ("build_s", pa.float64()), ("reason", pa.string()), ("finals", pa.string()),
    ("package_version", pa.string()), ("suite_rev", pa.string()),
    ("recorded_utc", pa.timestamp("us", tz="UTC")),
])
COLUMNS = tuple(SCHEMA.names)
FLOAT_VALUE_COLUMNS = ("min_ms", "errored_pct", "build_s")
TEXT_VALUE_COLUMNS = ("reason", "finals", "package_version", "suite_rev")

NAN = float("nan")
HASH_HEX = 16
T_FINAL_RTOL = 1e-4
LOCK_TIMEOUT_S = 120.0
LOCK_STALE_S = 300.0


def _float(value):
    """A float value; None, empty text and "nan" are NaN; infinities and other text are refused."""
    if value is None:
        return NAN
    if isinstance(value, str):
        if value.strip().lower() in ("", "nan"):
            return NAN
        value = float(value)
    value = float(value)
    if math.isinf(value):
        raise ValueError("infinite float")
    return value


def _int(name, value):
    """An integer value; a float or text is accepted only when it is a whole number."""
    if isinstance(value, bool):
        raise ValueError("{0} is a bool".format(name))
    if isinstance(value, str):
        value = float(value)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("{0} {1!r} is not an integer".format(name, value))
        value = int(value)
    return int(value)


def _text(value):
    return "" if value is None else str(value)


def _utc(value):
    """A timezone-aware UTC datetime from a datetime, an ISO string, or None (now)."""
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, str):
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def canonical_json(value):
    """A JSON object as its canonical text: sorted keys, no whitespace; accepts a dict, JSON text, or None ({})."""
    if isinstance(value, str):
        value = json.loads(value) if value.strip() else {}
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("a JSON object is required, got " + type(value).__name__)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _spec_value(name, kind, value):
    if kind == "float":
        return _float(value)
    if kind == "int":
        return _int(name, value)
    if kind == "json":
        return canonical_json(value)
    text = _text(value)
    if not text:
        raise ValueError("spec field {0} is empty".format(name))
    if kind == "name" and ("/" in text or "\\" in text):
        raise ValueError("bad {0} '{1}'".format(name, text))
    if name in _ENUMS and text not in _ENUMS[name]:
        raise ValueError("{0} '{1}' is not one of {2}".format(
            name, text, ", ".join(_ENUMS[name])))
    return text


def spec_of(fields, names=SPEC_FIELDS):
    """The validated spec columns named, from a row or trial dict; every one must be present."""
    missing = [f for f in names if f not in fields]
    if missing:
        raise ValueError("spec incomplete: " + ", ".join(missing))
    kinds = dict(SPEC_TYPES)
    spec = {name: _spec_value(name, kinds[name], fields[name]) for name in names}
    if "n" in spec and spec["n"] < 1:
        raise ValueError("n must be positive")
    return spec


def _hash_text(value, kind):
    if kind == "float":
        return '"nan"' if math.isnan(value) else format(value, ".17g")
    if kind == "int":
        return str(int(value))
    return json.dumps(value)


def canonical_spec_text(spec, names):
    """The hash input: a JSON object of the named fields in table order, floats %.17g, NaN "nan"."""
    kinds = dict(SPEC_TYPES)
    return "{" + ",".join('"{0}":{1}'.format(name, _hash_text(spec[name], kinds[name]))
                          for name in names) + "}"


def _hash(spec, names):
    text = canonical_spec_text(spec_of(spec, names), names)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:HASH_HEX]


def trial_id(spec):
    """sha1 of the canonical text of every spec field but transfers and key, first 16 hex."""
    return _hash(spec, TRIAL_FIELDS)


def run_id(spec):
    """sha1 of the canonical text of every spec field, first 16 hex; the store's replace key."""
    return _hash(spec, SPEC_FIELDS)


def group_id(spec):
    """sha1 of the canonical text of the system and stepping fields (no ensemble, transfers, package or key), first 16 hex; the comparison key across packages and grids."""
    return _hash(spec, GROUP_FIELDS)


def ids(spec):
    """{trial_id, run_id, group_id} of a spec; trial_id and group_id need every field but transfers and key."""
    return {name: _hash(spec, names) for name, names in ID_FIELDS.items()}


def make_row(**fields):
    """One complete row: the spec as given with its hashes, every value column defaulted (NaN, [], "", now)."""
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise ValueError("unknown columns: " + ", ".join(sorted(unknown)))
    row = spec_of(fields)
    row.update(ids(row))
    for name in ID_FIELDS:
        given = _text(fields.get(name))
        if given and given != row[name]:
            raise ValueError("{0} {1} does not hash the spec ({2})".format(
                name, given, row[name]))
    if fields.get("states") is None:
        raise ValueError("states is required")
    row["states"] = _int("states", fields["states"])
    for field in FLOAT_VALUE_COLUMNS:
        row[field] = _float(fields.get(field))
    samples = fields.get("samples_ms")
    if samples is None:
        samples = []
    elif isinstance(samples, str):
        raise ValueError("samples_ms is a list of ms, not a string")
    row["samples_ms"] = [float(s) for s in samples]
    for field in TEXT_VALUE_COLUMNS:
        row[field] = _text(fields.get(field))
    row["recorded_utc"] = _utc(fields.get("recorded_utc"))
    return {name: row[name] for name in COLUMNS}


def suite_rev(repo_root=REPO_ROOT):
    """`git rev-parse --short HEAD`; 'unknown' outside git."""
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=repo_root, capture_output=True, text=True,
                             timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return rev.stdout.strip() if rev.returncode == 0 else "unknown"


class _Lock:
    """A mkdir lock beside a results file; a stale directory is taken over."""

    def __init__(self, path, timeout=LOCK_TIMEOUT_S, stale=LOCK_STALE_S):
        self.path = path + ".lock"
        self.timeout, self.stale = timeout, stale

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                os.mkdir(self.path)
                return self
            except FileExistsError:
                try:
                    age = time.time() - os.stat(self.path).st_mtime
                except OSError:
                    age = 0.0
                if age > self.stale:
                    try:
                        os.rmdir(self.path)
                    except OSError:
                        pass
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError("result store locked: " + self.path)
                time.sleep(0.05)

    def __exit__(self, *_):
        try:
            os.rmdir(self.path)
        except OSError:
            pass


def _replace(scratch, path):
    # Retry the swap while a reader holds the old file.
    for attempt in range(20):
        try:
            os.replace(scratch, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def _write_parquet(path, table):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scratch = path + ".partial"
    pq.write_table(table, scratch)
    _replace(scratch, path)


def _lower_finite_wins(recorded, new):
    """Under --floor: the lower finite min_ms stays; NaN never displaces a finite time."""
    old, fresh = recorded["min_ms"], new["min_ms"]
    if math.isnan(old):
        return True
    if math.isnan(fresh):
        return False
    return fresh < old


def _place(rows, row, floor):
    """Replace the row of the same run_id in place (append when absent); returns the row that stands."""
    standing = row
    for index, existing in enumerate(rows):
        if existing["run_id"] == row["run_id"]:
            if floor and not _lower_finite_wins(existing, row):
                standing = existing
            rows[index] = standing
            return standing
    rows.append(row)
    return standing


def finals_name(spec):
    """finals/<trial_id>.parquet, relative to the package dir."""
    return "finals/" + trial_id(spec) + ".parquet"


def errored_mask(states, t_final, retcode, duration):
    """bool[m]: a non-finite state, a final time off duration by more than T_FINAL_RTOL, or a non-empty retcode."""
    states = np.asarray(states)
    t_final = np.asarray(t_final, dtype=np.float64).reshape(-1)
    bad_state = ~np.isfinite(states).all(axis=1) if states.ndim == 2 \
        else np.zeros(t_final.shape[0], dtype=bool)
    short = ~(np.abs(t_final - float(duration)) <= T_FINAL_RTOL * abs(float(duration)))
    coded = np.array([bool(_text(code)) for code in retcode], dtype=bool)
    return bad_state | short | coded


def errored_pct(states, t_final, retcode, duration):
    """Percent of trajectories errored_mask marks."""
    mask = errored_mask(states, t_final, retcode, duration)
    return 100.0 * float(mask.sum()) / mask.shape[0] if mask.shape[0] else NAN


class Store:
    """The parquet tree under root; `data` by default."""

    def __init__(self, root="data"):
        self.root = root

    def package_dir(self, package, key):
        return os.path.join(self.root, "key=" + key, "package=" + package)

    def results_path(self, package, key, problem, algorithm):
        return os.path.join(self.package_dir(package, key), "results",
                            "{0}__{1}.parquet".format(problem, algorithm))

    def _results_of(self, row):
        return self.results_path(row["package"], row["key"], row["problem"],
                             row["algorithm"])

    @staticmethod
    def _read_results(path):
        if not os.path.isfile(path):
            return []
        return pq.read_table(path).to_pylist()

    @staticmethod
    def _write_results(path, rows):
        if not rows:
            if os.path.isfile(path):
                os.remove(path)
            return
        _write_parquet(path, pa.Table.from_pylist(rows, schema=SCHEMA))

    def record(self, row, floor=False):
        """Replace the row of this run_id; under floor the lower finite min_ms stays. Returns the row that now stands."""
        return self.record_batch([row], floor=floor)[0]

    def record_batch(self, rows, floor=False):
        """Record rows in order, one lock and one rewrite per results file; returns the standing row of each."""
        made = [make_row(**row) for row in rows]
        standing = [None] * len(made)
        by_leg = {}
        for index, row in enumerate(made):
            by_leg.setdefault(self._results_of(row), []).append(index)
        for path, indices in by_leg.items():
            with _Lock(path):
                existing = self._read_results(path)
                for index in indices:
                    standing[index] = _place(existing, made[index], floor)
                self._write_results(path, existing)
        return standing

    def record_finals(self, spec, finals, t_final, retcode=None):
        """Write finals/<trial_id>.parquet of a trial: all n rows in the run precision, each trajectory's final time and the package's failure code text (empty on success or when it reports none); returns the path relative to the package dir."""
        ident = spec_of(spec, FINALS_FIELDS)
        dtype = np.float64 if ident["precision"] == "float64" else np.float32
        states = np.asarray(finals, dtype=dtype)
        if states.ndim != 2:
            raise ValueError("finals is rows x states")
        if states.shape[0] != ident["n"]:
            raise ValueError("finals has {0} rows for n = {1}".format(
                states.shape[0], ident["n"]))
        t_final = np.asarray(t_final, dtype=np.float64).reshape(-1)
        if t_final.shape[0] != states.shape[0]:
            raise ValueError("t_final has one time per finals row")
        if retcode is None:
            retcode = [""] * states.shape[0]
        retcode = [_text(code) for code in retcode]
        if len(retcode) != states.shape[0]:
            raise ValueError("retcode has one code per finals row")
        columns = {"traj": pa.array(np.arange(states.shape[0], dtype=np.int32),
                                    pa.int32())}
        arrow_type = pa.float64() if dtype is np.float64 else pa.float32()
        for k in range(states.shape[1]):
            columns["s{0}".format(k + 1)] = pa.array(states[:, k], arrow_type)
        columns["t_final"] = pa.array(t_final, pa.float64())
        columns["retcode"] = pa.array(retcode, pa.string())
        relative = finals_name(ident)
        _write_parquet(os.path.join(self.package_dir(ident["package"], ident["key"]),
                                    *relative.split("/")),
                       pa.table(columns))
        return relative

    def finals_readable(self, package, key, relative):
        """True when a package-relative finals path names a parquet file whose metadata reads."""
        if not relative:
            return False
        path = os.path.join(self.package_dir(package, key), *relative.split("/"))
        if not os.path.isfile(path):
            return False
        try:
            pq.read_metadata(path)
        except Exception:  # noqa: BLE001 - an unreadable file is a missing artifact
            return False
        return True

    def load_finals(self, package, key, relative):
        """(traj int32[m], states [m, k] in the stored precision, t_final float64[m], retcode str[m]) of a finals file by its package-relative path."""
        table = pq.read_table(os.path.join(self.package_dir(package, key),
                                           *relative.split("/")))
        names = [c for c in table.column_names if c[0] == "s" and c[1:].isdigit()]
        names.sort(key=lambda c: int(c[1:]))
        dtype = np.float64 if names and str(table.schema.field(names[0]).type) == "double" \
            else np.float32
        states = np.column_stack([table.column(c).to_numpy() for c in names]) \
            if names else np.zeros((table.num_rows, 0), dtype)
        return (table.column("traj").to_numpy(), states.astype(dtype),
                table.column("t_final").to_numpy(),
                np.asarray(table.column("retcode").to_pylist(), dtype=object))

    def status(self, run):
        """'absent', 'nan' or 'finite' for the row of a run_id (or of a spec's run_id)."""
        if isinstance(run, dict):
            run = run_id(run)
        matched = self.rows(run_id=run)
        if not matched:
            return "absent"
        if any(math.isfinite(r["min_ms"]) for r in matched):
            return "finite"
        return "nan"

    def results_files(self):
        pattern = os.path.join(self.root, "key=*", "package=*", "results",
                               "*.parquet")
        return sorted(glob.glob(pattern))

    def _connect(self):
        """A DuckDB connection with a `results` view over every results file, in UTC."""
        import duckdb
        con = duckdb.connect()
        con.execute("SET TimeZone = 'UTC'")
        if self.results_files():
            pattern = os.path.join(os.path.abspath(self.root), "key=*",
                                   "package=*", "results", "*.parquet")
            pattern = pattern.replace("\\", "/").replace("'", "''")
            con.execute(
                "CREATE VIEW results AS SELECT * FROM read_parquet('{0}', "
                "hive_partitioning = true, "
                "hive_types = {{'key': VARCHAR, 'package': VARCHAR}})".format(pattern))
        else:
            con.register("empty_results", SCHEMA.empty_table())
            con.execute("CREATE VIEW results AS SELECT * FROM empty_results")
        return con

    def query(self, sql):
        """The result of a SQL statement over the `results` view, as an Arrow table."""
        con = self._connect()
        try:
            return con.execute(sql).to_arrow_table()
        finally:
            con.close()

    def rows(self, sql_where="", **eq_filters):
        """Rows of the whole tree as dicts, filtered by exact equality on columns (NaN matches NaN) and an optional SQL predicate."""
        clauses, params = [], []
        for column, value in eq_filters.items():
            if column not in COLUMNS:
                raise ValueError("unknown column " + column)
            if isinstance(value, float) and math.isnan(value):
                clauses.append('isnan("{0}")'.format(column))
            else:
                clauses.append('"{0}" = ?'.format(column))
                params.append(value)
        if sql_where:
            clauses.append("(" + sql_where + ")")
        sql = "SELECT * FROM results"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        con = self._connect()
        try:
            return con.execute(sql, params).to_arrow_table().to_pylist()
        finally:
            con.close()

    def clear(self, **eq_filters):
        """Drop every row matching the equality filters; returns the count dropped."""
        unknown = set(eq_filters) - set(COLUMNS)
        if unknown:
            raise ValueError("unknown columns: " + ", ".join(sorted(unknown)))
        dropped = 0
        for path in self.results_files():
            with _Lock(path):
                rows = self._read_results(path)
                kept = [r for r in rows if not _matches(r, eq_filters)]
                if len(kept) != len(rows):
                    self._write_results(path, kept)
                    dropped += len(rows) - len(kept)
        return dropped


def _matches(row, filters):
    kinds = dict(SPEC_TYPES)
    for column, value in filters.items():
        if column not in COLUMNS:
            raise ValueError("unknown column " + column)
        if kinds.get(column) == "float" or column in FLOAT_VALUE_COLUMNS:
            a, b = _float(row[column]), _float(value)
            if not (a == b or (math.isnan(a) and math.isnan(b))):
                return False
        elif kinds.get(column) == "int" or column == "states":
            if int(row[column]) != int(value):
                return False
        elif kinds.get(column) == "json":
            if row[column] != canonical_json(value):
                return False
        elif str(row[column]) != str(value):
            return False
    return True


def _read_json(path):
    if path == "-":
        return json.load(sys.stdin)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_finals_csv(path):
    """(finals, t_final, retcode) from a CSV with header s1..sk,t_final and optional traj and retcode columns."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = [c for c in reader.fieldnames if c[0] == "s" and c[1:].isdigit()]
        names.sort(key=lambda c: int(c[1:]))
        finals, t_final, retcode = [], [], []
        for record in reader:
            finals.append([float(record[c]) for c in names])
            t_final.append(_float(record["t_final"]))
            retcode.append(record.get("retcode") or "")
    return finals, t_final, retcode


def _cli(argv):
    parser = argparse.ArgumentParser(prog="store.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data")
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record")
    record.add_argument("rows")
    record.add_argument("--floor", action="store_true")
    finals = commands.add_parser("finals")
    finals.add_argument("spec")
    finals.add_argument("finals")
    status = commands.add_parser("status")
    status.add_argument("run_id")
    query = commands.add_parser("query")
    query.add_argument("sql")
    clear = commands.add_parser("clear")
    clear.add_argument("filters")
    hash_ = commands.add_parser("hash")
    hash_.add_argument("spec")
    args = parser.parse_args(argv)
    store = Store(args.root)

    if args.command == "record":
        rows = _read_json(args.rows)
        if isinstance(rows, dict):
            rows = [rows]
        store.record_batch(rows, floor=args.floor)
        return 0
    if args.command == "finals":
        finals, t_final, retcode = _read_finals_csv(args.finals)
        print(store.record_finals(_read_json(args.spec), finals, t_final, retcode))
        return 0
    if args.command == "status":
        print(store.status(args.run_id))
        return 0
    if args.command == "query":
        table = store.query(args.sql)
        # Keep the LF terminator on Windows.
        sys.stdout.reconfigure(newline="")
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow(table.column_names)
        for row in table.to_pylist():
            writer.writerow([_cell(row[c]) for c in table.column_names])
        return 0
    if args.command == "clear":
        print(store.clear(**_read_json(args.filters)))
        return 0
    if args.command == "hash":
        print(json.dumps(ids(_read_json(args.spec))))
        return 0
    return 1


def _cell(value):
    """A CSV cell: NaN as nan, lists ';'-joined, timestamps ISO."""
    if isinstance(value, float):
        return "nan" if math.isnan(value) else format(value, ".17g")
    if isinstance(value, list):
        return ";".join(format(v, ".17g") for v in value)
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return "" if value is None else value


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
