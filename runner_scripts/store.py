"""The result store: one parquet file per leg under data/key=<os>_<gpu>/package=<pkg>/results/, finals beside it, DuckDB over the whole tree. The only writer, and the Python reader.

    Store(root="data").record(row, floor=False)            # one row, atomic leg-file swap under a mkdir lock
    Store.record_finals(identity, finals, converged) -> relative path
    Store.status(identity) -> "absent" | "nan" | "finite"
    Store.rows(sql_where="", **eq_filters) -> list[dict]     # DuckDB over the whole tree
    python store.py [--root DIR] record  <rows.json | ->      # JSON array of rows; samples_ms as a list
    python store.py [--root DIR] finals  <identity.json> <finals.csv>
    python store.py [--root DIR] status  <identity.json>
    python store.py [--root DIR] query   "<sql over results>"
    python store.py [--root DIR] clear   <filter.json>
"""

import argparse
import csv
import glob
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
MODES = ("fixed", "adaptive")
SETTING_KINDS = ("dt", "tol")
TIERS = ("default", "matched", "pi")
TRANSFERS = ("both", "none")

IDENTITY = ("package", "key", "problem", "algorithm", "mode", "setting_kind",
            "setting", "n", "states", "tier", "transfers")
# A finals file belongs to a trial, which both transfer legs share.
FINALS_IDENTITY = tuple(f for f in IDENTITY if f != "transfers")

SCHEMA = pa.schema([
    ("package", pa.string()), ("key", pa.string()), ("problem", pa.string()),
    ("algorithm", pa.string()), ("mode", pa.string()),
    ("setting_kind", pa.string()), ("setting", pa.float64()),
    ("n", pa.int64()), ("states", pa.int32()), ("tier", pa.string()),
    ("transfers", pa.string()), ("min_ms", pa.float64()),
    ("samples_ms", pa.list_(pa.float64())), ("errored_pct", pa.float64()),
    ("error", pa.float64()), ("build_s", pa.float64()),
    ("reason", pa.string()), ("finals", pa.string()),
    ("package_version", pa.string()), ("suite_rev", pa.string()),
    ("recorded_utc", pa.timestamp("us", tz="UTC")),
])
COLUMNS = tuple(SCHEMA.names)
FLOAT_COLUMNS = ("setting", "min_ms", "errored_pct", "error", "build_s")
TEXT_COLUMNS = ("reason", "finals", "package_version", "suite_rev")

NAN = float("nan")
SETTING_REL_TOL = 1e-8
LOCK_TIMEOUT_S = 120.0
LOCK_STALE_S = 300.0


def _float(value):
    """A float column value; None and unparsable text are NaN."""
    if value is None:
        return NAN
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return NAN
    return float(value)


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


def format_setting(setting):
    """The setting as it appears in finals file names."""
    return format(float(setting), ".10g")


def setting_matches(a, b):
    """Two settings name the same point within a relative 1e-8."""
    a, b = _float(a), _float(b)
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=SETTING_REL_TOL, abs_tol=0.0)


def make_row(**fields):
    """One complete row: the identity as given, every value column defaulted (NaN, [], "", now)."""
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise ValueError("unknown columns: " + ", ".join(sorted(unknown)))
    missing = [f for f in IDENTITY if f not in fields]
    if missing:
        raise ValueError("identity incomplete: " + ", ".join(missing))
    row = {}
    for field in ("package", "key", "problem", "algorithm", "mode",
                  "setting_kind", "tier", "transfers"):
        row[field] = str(fields[field])
    row["setting"] = float(fields["setting"])
    row["n"] = int(fields["n"])
    row["states"] = int(fields["states"])
    for field in ("min_ms", "errored_pct", "error", "build_s"):
        row[field] = _float(fields.get(field))
    samples = fields.get("samples_ms")
    if samples is None:
        samples = []
    elif isinstance(samples, str):
        raise ValueError("samples_ms is a list of ms, not a string")
    row["samples_ms"] = [float(s) for s in samples]
    for field in TEXT_COLUMNS:
        row[field] = _text(fields.get(field))
    row["recorded_utc"] = _utc(fields.get("recorded_utc"))
    _validate(row)
    return row


def _validate(row):
    checks = (("package", PACKAGES), ("mode", MODES),
              ("setting_kind", SETTING_KINDS), ("tier", TIERS),
              ("transfers", TRANSFERS))
    for field, allowed in checks:
        if row[field] not in allowed:
            raise ValueError("{0} '{1}' is not one of {2}".format(
                field, row[field], ", ".join(allowed)))
    for field in ("key", "problem", "algorithm"):
        if not row[field] or "/" in row[field] or "\\" in row[field]:
            raise ValueError("bad {0} '{1}'".format(field, row[field]))


def same_identity(row, ident):
    """True when a row carries every identity column given in ident."""
    for field in IDENTITY:
        if field not in ident:
            continue
        if field == "setting":
            if not setting_matches(row[field], ident[field]):
                return False
        elif field in ("n", "states"):
            if int(row[field]) != int(ident[field]):
                return False
        elif str(row[field]) != str(ident[field]):
            return False
    return True


def finals_name(ident):
    """<problem>__<algorithm>__<mode>__<kind>-<setting>__n<n>__s<states>__<tier>.parquet"""
    return "{problem}__{algorithm}__{mode}__{kind}-{setting}__n{n}__s{states}__{tier}.parquet".format(
        problem=ident["problem"], algorithm=ident["algorithm"],
        mode=ident["mode"], kind=ident["setting_kind"],
        setting=format_setting(ident["setting"]), n=int(ident["n"]),
        states=int(ident["states"]), tier=ident["tier"])


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
    # Windows refuses the swap while a reader still holds the old file.
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


class Store:
    """The parquet tree under root; `data` by default."""

    def __init__(self, root="data"):
        self.root = root

    def package_dir(self, package, key):
        return os.path.join(self.root, "key=" + key, "package=" + package)

    def leg_path(self, package, key, problem, algorithm, mode):
        return os.path.join(self.package_dir(package, key), "results",
                            "{0}__{1}__{2}.parquet".format(problem, algorithm, mode))

    def _leg_of(self, ident):
        for field in ("package", "key", "problem", "algorithm", "mode"):
            if field not in ident:
                raise ValueError("identity needs " + field)
        return self.leg_path(ident["package"], ident["key"], ident["problem"],
                             ident["algorithm"], ident["mode"])

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
        """Replace the row with this identity; under floor the lower finite min_ms stays. Returns the row that now stands."""
        row = make_row(**row)
        path = self._leg_of(row)
        with _Lock(path):
            rows = self._read_leg(path)
            standing = row
            for index, existing in enumerate(rows):
                if same_identity(existing, row):
                    if floor and not _lower_finite_wins(existing, row):
                        standing = existing
                    rows[index] = standing
                    break
            else:
                rows.append(row)
            self._write_leg(path, rows)
        return standing

    def record_finals(self, identity, finals, converged):
        """Write the finals file of a trial; returns its path relative to the package dir."""
        ident = {f: identity[f] for f in FINALS_IDENTITY if f in identity}
        missing = [f for f in FINALS_IDENTITY if f not in ident]
        if missing:
            raise ValueError("finals identity incomplete: " + ", ".join(missing))
        states = np.asarray(finals, dtype=np.float32)
        if states.ndim != 2:
            raise ValueError("finals is rows x states")
        converged = np.asarray(converged, dtype=bool).reshape(-1)
        if converged.shape[0] != states.shape[0]:
            raise ValueError("converged has one flag per finals row")
        columns = {"traj": pa.array(np.arange(states.shape[0], dtype=np.int32),
                                    pa.int32())}
        for k in range(states.shape[1]):
            columns["s{0}".format(k + 1)] = pa.array(states[:, k], pa.float32())
        columns["converged"] = pa.array(converged, pa.bool_())
        relative = "finals/" + finals_name(ident)
        _write_parquet(os.path.join(self.package_dir(ident["package"], ident["key"]),
                                    "finals", finals_name(ident)),
                       pa.table(columns))
        return relative

    def load_finals(self, package, key, relative):
        """(traj int32[m], states float32[m, k], converged bool[m]) of a finals file by its relative path."""
        table = pq.read_table(os.path.join(self.package_dir(package, key),
                                           *relative.split("/")))
        names = [c for c in table.column_names if c.startswith("s")]
        names.sort(key=lambda c: int(c[1:]))
        states = np.column_stack([table.column(c).to_numpy() for c in names]) \
            if names else np.zeros((table.num_rows, 0), np.float32)
        return (table.column("traj").to_numpy(), states.astype(np.float32),
                table.column("converged").to_numpy())

    def status(self, identity):
        """'absent', 'nan' or 'finite' for the rows carrying the given identity columns."""
        matched = [r for r in self._read_leg(self._leg_of(identity))
                   if same_identity(r, identity)]
        if not matched:
            return "absent"
        if any(math.isfinite(r["min_ms"]) for r in matched):
            return "finite"
        return "nan"

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
        """Rows of the whole tree as dicts, filtered by equality on columns and an optional SQL predicate."""
        clauses, params = [], []
        for column, value in eq_filters.items():
            if column not in COLUMNS:
                raise ValueError("unknown column " + column)
            if column == "setting":
                clauses.append("abs(setting - ?) <= ? * abs(?)")
                params += [float(value), SETTING_REL_TOL, float(value)]
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
    for column, value in filters.items():
        if column == "setting":
            if not setting_matches(row[column], value):
                return False
        elif column in ("n", "states"):
            if int(row[column]) != int(value):
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
    finals.add_argument("identity")
    finals.add_argument("finals")
    status = commands.add_parser("status")
    status.add_argument("identity")
    query = commands.add_parser("query")
    query.add_argument("sql")
    clear = commands.add_parser("clear")
    clear.add_argument("filters")
    args = parser.parse_args(argv)
    store = Store(args.root)

    if args.command == "record":
        rows = _read_json(args.rows)
        if isinstance(rows, dict):
            rows = [rows]
        for row in rows:
            store.record(row, floor=args.floor)
        return 0
    if args.command == "finals":
        finals, converged = _read_finals_csv(args.finals)
        print(store.record_finals(_read_json(args.identity), finals, converged))
        return 0
    if args.command == "status":
        print(store.status(_read_json(args.identity)))
        return 0
    if args.command == "query":
        table = store.query(args.sql)
        # Text-mode stdout would turn the line terminator into CRLF on Windows.
        sys.stdout.reconfigure(newline="")
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow(table.column_names)
        for row in table.to_pylist():
            writer.writerow([_cell(row[c]) for c in table.column_names])
        return 0
    if args.command == "clear":
        print(store.clear(**_read_json(args.filters)))
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
