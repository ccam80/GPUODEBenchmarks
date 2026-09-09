"""A scripted runner for the bench.py tests: fake_runner.py --root DIR --key KEY --mode ok|crash|hang-once|hang-silent --trials PATH [--floor]; hang-once records each leg's first trial, writes the progress file and exits with the watchdog code, then behaves as ok."""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import store  # noqa: E402
import trials  # noqa: E402
from protocol import WATCHDOG_EXIT_CODE  # noqa: E402


def record(root, key, trial, floor):
    rows = store.Store(root)
    for transfers in trial.transfers:
        rows.record(dict(trial.identity(key, transfers), min_ms=1.0 + trial.ordinal,
                         samples_ms=[9.0, 1.0 + trial.ordinal], package_version="fake 1.0",
                         suite_rev="test"), floor=floor)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--trials", required=True)
    parser.add_argument("--floor", action="store_true")
    args = parser.parse_args(argv)
    items = trials.read_jsonl(args.trials)
    print("fake runner: {0} trials, floor={1}".format(len(items), args.floor))
    if args.mode == "crash":
        return 1
    if args.mode == "hang-silent":
        return WATCHDOG_EXIT_CODE
    marker = os.path.join(args.root, "fake_runner_hung")
    if args.mode == "hang-once" and not os.path.exists(marker):
        open(marker, "w").close()
        legs = {}
        for trial in items:
            if trial.kind == "solve":
                legs.setdefault(trial.leg_key, []).append(trial)
        first_leg = None
        for members in legs.values():
            members.sort(key=lambda t: t.ordinal)
            record(args.root, args.key, members[0], args.floor)
            if first_leg is None and len(members) > 1:
                first_leg = members
        if first_leg is not None:
            with open(args.trials + ".progress", "w", encoding="utf-8") as handle:
                json.dump({"id": first_leg[1].id, "started_utc": "2026-09-09T00:00:00Z"}, handle)
        return WATCHDOG_EXIT_CODE
    for trial in items:
        if trial.kind == "solve":
            record(args.root, args.key, trial, args.floor)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
