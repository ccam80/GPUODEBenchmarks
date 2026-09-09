"""The result store: data/key=<os>_<gpu>/package=<pkg>/results/<problem>__<algorithm>.parquet per leg, finals/<trial_id>.parquet beside it, DuckDB over the tree; a row is its run spec, hashed to run_id (the replace key) and trial_id. CLI: store.py [--root DIR] record <rows.json|-> [--floor] | finals <spec.json> <finals.csv> | status <run_id> | query "<sql over results>" | clear <filter.json> | hash <spec.json>."""

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

from grid import grid_contains

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
GRID_FIELDS = ("grid_scale", "grid_min", "grid_max", "n", "grid_dtype")
_ENUMS = {"precision": PRECISIONS, "grid_scale": GRID_SCALES,
          "grid_dtype": GRID_DTYPES, "transfers": TRANSFERS, "package": PACKAGES}

_ARROW = {"name": pa.string(), "str": pa.string(), "json": pa.string(),
          "float": pa.float64(), "int": pa.int64()}
SCHEMA = pa.schema([(name, _ARROW[kind]) for name, kind in SPEC_TYPES] + [
    ("run_id", pa.string()), ("trial_id", pa.string()),
    ("states", pa.int32()), ("min_ms", pa.float64()),
    ("samples_ms", pa.list_(pa.float64())), ("errored_pct", pa.float64()),
    ("error", pa.float64()), ("reference", pa.string()),
    ("build_s", pa.float64()), ("reason", pa.string()), ("finals", pa.string()),
    ("package_version", pa.string()), ("suite_rev", pa.string()),
    ("recorded_utc", pa.timestamp("us", tz="UTC")),
])
COLUMNS = tuple(SCHEMA.names)
FLOAT_VALUE_COLUMNS = ("min_ms", "errored_pct", "error", "build_s")
TEXT_VALUE_COLUMNS = ("reference", "reason", "finals", "package_version", "suite_rev")

NAN = float("nan")
HASH_HEX = 16
LOCK_TIMEOUT_S = 120.0
LOCK_STALE_S = 300.0


def _float(value):
    """A float value; None and unparsable text are NaN; infinities are refused."""
    if value is None:
        return NAN
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return NAN
    value = float(value)
    if math.isinf(value):
        raise ValueError("infinite float")
    return value


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
        return int(value)
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


def make_row(**fields):
    """One complete row: the spec as given with its hashes, every value column defaulted (NaN, [], "", now)."""
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise ValueError("unknown columns: " + ", ".join(sorted(unknown)))
    row = spec_of(fields)
    row["run_id"], row["trial_id"] = run_id(row), trial_id(row)
    for name in ("run_id", "trial_id"):
        given = _text(fields.get(name))
        if given and given != row[name]:
            raise ValueError("{0} {1} does not hash the spec ({2})".format(
                name, given, row[name]))
    if fields.get("states") is None:
        raise ValueError("states is required")
    row["states"] = int(fields["states"])
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
    """`git rev-parse --short HEAD`, suffixed -dirty when tracked files have changed; 'unknown' outside git."""
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=repo_root, capture_output=True, text=True,
                             timeout=30)
        if rev.returncode != 0:
            return "unknown"
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    dirty = status.returncode == 0 and status.stdout.strip() != ""
    return rev.stdout.strip() + ("-dirty" if dirty else "")


class _Lock:
    """A mkdir lock beside a leg file; a stale directory is taken over."""

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


class Store:
    """The parquet tree under root; `data` by default."""

    def __init__(self, root="data"):
        self.root = root

    def package_dir(self, package, key):
        return os.path.join(self.root, "key=" + key, "package=" + package)

    def leg_path(self, package, key, problem, algorithm):
        return os.path.join(self.package_dir(package, key), "results",
                            "{0}__{1}.parquet".format(problem, algorithm))

    def _leg_of(self, row):
        return self.leg_path(row["package"], row["key"], row["problem"],
                             row["algorithm"])

    @staticmethod
    def _read_leg(path):
        if not os.path.isfile(path):
            return []
        return pq.read_table(path).to_pylist()

    @staticmethod
    def _write_leg(path, rows):
        if not rows:
            if os.path.isfile(path):
                os.remove(path)
            return
        _write_parquet(path, pa.Table.from_pylist(rows, schema=SCHEMA))

    def record(self, row, floor=False):
        """Replace the row of this run_id; under floor the lower finite min_ms stays. Returns the row that now stands."""
        return self.record_batch([row], floor=floor)[0]

    def record_batch(self, rows, floor=False):
        """Record rows in order, one lock and one rewrite per leg file; returns the standing row of each."""
        made = [make_row(**row) for row in rows]
        standing = [None] * len(made)
        by_leg = {}
        for index, row in enumerate(made):
            by_leg.setdefault(self._leg_of(row), []).append(index)
        for path, indices in by_leg.items():
            with _Lock(path):
                existing = self._read_leg(path)
                for index in indices:
                    standing[index] = _place(existing, made[index], floor)
                self._write_leg(path, existing)
        return standing

    def record_finals(self, spec, finals, converged):
        """Write finals/<trial_id>.parquet of a trial (all n rows, the run precision); returns the path relative to the package dir."""
        ident = spec_of(spec, FINALS_FIELDS)
        dtype = np.float64 if ident["precision"] == "float64" else np.float32
        states = np.asarray(finals, dtype=dtype)
        if states.ndim != 2:
            raise ValueError("finals is rows x states")
        if states.shape[0] != ident["n"]:
            raise ValueError("finals has {0} rows for n = {1}".format(
                states.shape[0], ident["n"]))
        converged = np.asarray(converged, dtype=bool).reshape(-1)
        if converged.shape[0] != states.shape[0]:
            raise ValueError("converged has one flag per finals row")
        columns = {"traj": pa.array(np.arange(states.shape[0], dtype=np.int32),
                                    pa.int32())}
        arrow_type = pa.float64() if dtype is np.float64 else pa.float32()
        for k in range(states.shape[1]):
            columns["s{0}".format(k + 1)] = pa.array(states[:, k], arrow_type)
        columns["converged"] = pa.array(converged, pa.bool_())
        relative = finals_name(ident)
        _write_parquet(os.path.join(self.package_dir(ident["package"], ident["key"]),
                                    *relative.split("/")),
                       pa.table(columns))
        return relative

    def load_finals(self, package, key, relative):
        """(traj int32[m], states [m, k] in the stored precision, converged bool[m]) of a finals file by its package-relative path."""
        return self.load_finals_at("/".join(("key=" + key, "package=" + package, relative)))

    def load_finals_at(self, store_relative):
        """The same for a store-relative path `key=<k>/package=<pkg>/finals/<id>.parquet`."""
        table = pq.read_table(os.path.join(self.root, *store_relative.split("/")))
        names = [c for c in table.column_names if c.startswith("s")]
        names.sort(key=lambda c: int(c[1:]))
        dtype = np.float64 if names and str(table.schema.field(names[0]).type) == "double" \
            else np.float32
        states = np.column_stack([table.column(c).to_numpy() for c in names]) \
            if names else np.zeros((table.num_rows, 0), dtype)
        return (table.column("traj").to_numpy(), states.astype(dtype),
                table.column("converged").to_numpy())

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

    def covering(self, spec, finals=False):
        """The row of the spec's run_id; with finals, a finals row matching the spec apart from the grid whose grid contains the spec's. None otherwise."""
        if not finals:
            rows = self.rows(run_id=run_id(spec))
            return rows[0] if rows else None
        spec = spec_of(spec, FINALS_FIELDS)
        filters = {name: spec[name] for name in FINALS_FIELDS
                   if name not in ("grid_max", "n")}
        candidates = [r for r in self.rows("finals <> ''", **filters)
                      if grid_contains(r, spec)]
        if not candidates:
            return None
        candidates.sort(key=lambda r: (r["n"] != spec["n"], r["n"]))
        return candidates[0]

    def leg_files(self):
        pattern = os.path.join(self.root, "key=*", "package=*", "results",
                               "*.parquet")
        return sorted(glob.glob(pattern))

    def _connect(self):
        """A DuckDB connection with a `results` view over every leg file, in UTC."""
        import duckdb
        con = duckdb.connect()
        con.execute("SET TimeZone = 'UTC'")
        if self.leg_files():
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
        for path in self.leg_files():
            with _Lock(path):
                rows = self._read_leg(path)
                kept = [r for r in rows if not _matches(r, eq_filters)]
                if len(kept) != len(rows):
                    self._write_leg(path, kept)
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
    """(finals, converged) from a CSV with header s1..sk,converged and an optional traj column."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = [c for c in reader.fieldnames if c.startswith("s")]
        names.sort(key=lambda c: int(c[1:]))
        finals, converged = [], []
        for record in reader:
            finals.append([float(record[c]) for c in names])
            converged.append(record["converged"].strip().lower()
                             in ("1", "true", "t", "yes"))
    return finals, converged


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
        finals, converged = _read_finals_csv(args.finals)
        print(store.record_finals(_read_json(args.spec), finals, converged))
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
        spec = _read_json(args.spec)
        print(json.dumps({"trial_id": trial_id(spec), "run_id": run_id(spec)}))
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
