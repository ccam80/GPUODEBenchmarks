"""Run-time constants from protocol.toml ([repeats], [watchdog], [optimize]); `python protocol.py get <table.key>` prints one value, `--cxx-header <path>` writes the C++ header."""

import os
import sys
import tomllib

PROTOCOL_TOML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "protocol.toml")

with open(PROTOCOL_TOML, "rb") as _handle:
    PROTOCOL = tomllib.load(_handle)

TABLES = ("repeats", "watchdog", "optimize")

_repeats = PROTOCOL["repeats"]
_watchdog = PROTOCOL["watchdog"]
_optimize = PROTOCOL["optimize"]

REPEAT_CAP = int(_repeats["cap"])
REPEAT_SCHEDULE = tuple((float(limit), int(floor), int(ceiling))
                        for limit, floor, ceiling in _repeats["schedule"])
REPEAT_SPREAD = float(_repeats["spread"])

WATCHDOG_SECONDS = float(_watchdog["seconds"])
WATCHDOG_EXIT_CODE = int(_watchdog["exit_code"])

OPTIMIZE_N = int(_optimize["n"])
OPTIMIZE_PER_POINT_FAMILIES = tuple(_optimize["per_point_families"])


def get(path):
    """One value by dotted table.key path."""
    node = PROTOCOL
    for part in path.split("."):
        node = node[part]
    return node


def _cxx_literal(value):
    if isinstance(value, float):
        return "INFINITY" if value == float("inf") else repr(value)
    return str(value)


def cxx_header():
    """The generated C++ header text: the repeat schedule and the watchdog."""
    lines = ["// Generated from runner_scripts/protocol.toml by protocol.py; do not edit.",
             "#pragma once", "#include <cmath>",
             "#define PROTOCOL_REPEAT_CAP {0}".format(REPEAT_CAP),
             "#define PROTOCOL_REPEAT_SPREAD {0}".format(_cxx_literal(REPEAT_SPREAD)),
             "#define PROTOCOL_WATCHDOG_SECONDS {0}".format(_cxx_literal(WATCHDOG_SECONDS)),
             "#define PROTOCOL_WATCHDOG_EXIT_CODE {0}".format(WATCHDOG_EXIT_CODE),
             "#define PROTOCOL_REPEAT_SCHEDULE_ROWS {0}".format(len(REPEAT_SCHEDULE)),
             "static const double PROTOCOL_REPEAT_SCHEDULE[][3] = {"]
    lines += ["    {{{0}, {1}, {2}}},".format(_cxx_literal(limit), floor, ceiling)
              for limit, floor, ceiling in REPEAT_SCHEDULE]
    lines.append("};")
    return "\n".join(lines) + "\n"


def write_cxx_header(path):
    """Write the header only when its text changes, so build caches stay valid."""
    text = cxx_header()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            if handle.read() == text:
                return
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main(argv):
    if len(argv) == 2 and argv[0] == "get":
        value = get(argv[1])
        print(" ".join(str(v) for v in value) if isinstance(value, list)
              else value)
        return 0
    if len(argv) == 2 and argv[0] == "--cxx-header":
        write_cxx_header(argv[1])
        return 0
    print("usage: protocol.py get <table.key> | --cxx-header <path>")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
