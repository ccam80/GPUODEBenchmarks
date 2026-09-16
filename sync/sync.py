"""The remote store, a data/ tree on the store box: pull copies the whole tree into the local mirror, leaving a local file that is newer than the box's alone; push copies this machine's key partition up, deleting nothing, and mirrors its own clocks files (calibration_<key>.csv and the <key>_<stamp>.csv log of each run) so one gone locally is deleted on the box, skipping the clocks when the partition holds no file; prune mirrors this key and its clocks files; check lists the differences under this key; unpushed lists this key's local files missing from or differing on the box, ignoring files only the box has. rclone drives the `box:` remote, rsync the `box` ssh host when rclone is absent; the same <host>:<path> string names both. CLI: sync.py [--root DIR] [--remote HOST:PATH] [--key KEY] [--tool rclone|rsync] [--dry-run] pull | push | sync | prune | check | unpushed."""

import argparse
import os
import shutil
import subprocess
import sys

REMOTE_ENV = "GPUODE_STORE_REMOTE"
DEFAULT_REMOTE = "box:/srv/gpuode/data"
# Files a store write leaves beside a parquet while it is in flight; never shipped.
TRANSIENT = ("*.partial", "*.lock", "*.lock/**")
TOOLS = ("rclone", "rsync")
COMMANDS = ("pull", "push", "sync", "prune", "check", "unpushed")
SETUP_HINT = "sync/README.md"


def remote_default():
    return os.environ.get(REMOTE_ENV) or DEFAULT_REMOTE


def _join(base, *parts):
    return "/".join([base.rstrip("/\\")] + list(parts))


def _local(root, *parts):
    return os.path.abspath(os.path.join(root, *parts))


def find_tool(preferred=None):
    """The sync program on PATH: the one named, else rclone, else rsync; None when neither is installed."""
    for name in ((preferred,) if preferred else TOOLS):
        if shutil.which(name):
            return name
    return None


def remote_host(remote):
    """The remote name before the first ':' (an rclone remote or an ssh Host); '' for a plain path, a Windows drive letter included."""
    head, sep, _ = remote.partition(":")
    if not sep or len(head) == 1 and sys.platform == "win32":
        return ""
    return head


def rclone_configured(remote):
    """True when the remote names a configured rclone remote or a plain path."""
    host = remote_host(remote)
    if not host:
        return True
    listed = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
    return listed.returncode == 0 and host + ":" in listed.stdout.split()


def unavailable(remote=None, tool=None):
    """Why a sync cannot run on this machine ('' when it can): no program on PATH, or an rclone remote that is not configured."""
    remote = remote or remote_default()
    tool = find_tool(tool)
    if tool is None:
        return "neither rclone nor rsync is on PATH ({0})".format(SETUP_HINT)
    if tool == "rclone" and not rclone_configured(remote):
        return "rclone has no remote '{0}:' ({1})".format(remote_host(remote), SETUP_HINT)
    return ""


def commands(command, root, key, remote, tool, dry_run=False, mirror_clocks=True):
    """The argv list of a command for the tool; sync is a push then a pull; a push copies the clocks files instead of mirroring them when mirror_clocks is False."""
    if command == "sync":
        return commands("push", root, key, remote, tool, dry_run, mirror_clocks) + \
            commands("pull", root, key, remote, tool, dry_run)
    key_dir = "key=" + key
    # This machine's clocks files: calibration_<key>.csv and the <key>_<stamp>.csv log of each run.
    own_clocks = ["*_" + key + ".csv", key + "_*.csv"]
    dry = ["--dry-run"] if dry_run else []
    if tool == "rclone":
        transient = [flag for pattern in TRANSIENT for flag in ("--exclude", pattern)]
        includes = [flag for pattern in own_clocks for flag in ("--include", pattern)]
        if command == "push":
            return [["rclone", "copy", _local(root, key_dir), _join(remote, key_dir)] + transient + dry,
                    ["rclone", "sync" if mirror_clocks else "copy", _local(root, "clocks"),
                     _join(remote, "clocks")] + includes + dry]
        if command == "pull":
            return [["rclone", "copy", remote, _local(root), "--update"] + transient + dry]
        if command == "prune":
            return [["rclone", "sync", _local(root, key_dir), _join(remote, key_dir)] + transient + dry,
                    ["rclone", "sync", _local(root, "clocks"), _join(remote, "clocks")] + includes + dry]
        if command == "check":
            return [["rclone", "check", _local(root, key_dir), _join(remote, key_dir)] + transient]
        if command == "unpushed":
            return [["rclone", "check", _local(root, key_dir), _join(remote, key_dir), "--one-way"] + transient]
    if tool == "rsync":
        transient = [flag for pattern in TRANSIENT[:2] for flag in ("--exclude", pattern)]
        includes = [flag for pattern in own_clocks for flag in ("--include", pattern)]
        if command == "push":
            return [["rsync", "-a"] + transient + dry
                    + [_local(root, key_dir) + "/", _join(remote, key_dir) + "/"],
                    ["rsync", "-a"] + (["--delete"] if mirror_clocks else []) + includes + ["--exclude", "*"] + dry
                    + [_local(root, "clocks") + "/", _join(remote, "clocks") + "/"]]
        if command == "pull":
            return [["rsync", "-au"] + transient + dry
                    + [remote.rstrip("/") + "/", _local(root) + "/"]]
        if command == "prune":
            return [["rsync", "-a", "--delete"] + transient + dry
                    + [_local(root, key_dir) + "/", _join(remote, key_dir) + "/"],
                    ["rsync", "-a", "--delete"] + includes + ["--exclude", "*"] + dry
                    + [_local(root, "clocks") + "/", _join(remote, "clocks") + "/"]]
        if command == "check":
            return [["rsync", "-aO", "--delete", "--dry-run", "--itemize-changes"] + transient
                    + [_local(root, key_dir) + "/", _join(remote, key_dir) + "/"]]
        if command == "unpushed":
            return [["rsync", "-aO", "--dry-run", "--itemize-changes"] + transient
                    + [_local(root, key_dir) + "/", _join(remote, key_dir) + "/"]]
    raise ValueError("unknown command {0} or tool {1}".format(command, tool))


def _has_files(directory):
    for _, _, names in os.walk(directory):
        if names:
            return True
    return False


def partition_has_files(root, key):
    """True when this key's local partition holds any file."""
    return _has_files(_local(root, "key=" + key))


def run(command, root, key, remote=None, tool=None, dry_run=False, out=sys.stdout):
    """Run the command's programs in order, stopping at the first failure; returns the exit code (2 before anything ran). pull creates root, push creates the partition and clocks dirs, prune and check need a partition with files, unpushed is 0 without one."""
    remote = remote or remote_default()
    reason = unavailable(remote, tool)
    if reason:
        out.write("store sync: " + reason + "\n")
        return 2
    tool = find_tool(tool)
    key_dir = _local(root, "key=" + key)
    if command in ("prune", "check") and not _has_files(key_dir):
        out.write("store sync: no files under {0}\n".format(key_dir))
        return 2
    if command == "unpushed" and not _has_files(key_dir):
        out.write("store sync: nothing under {0}\n".format(key_dir))
        return 0
    if command in ("push", "sync", "prune"):
        os.makedirs(key_dir, exist_ok=True)
        os.makedirs(_local(root, "clocks"), exist_ok=True)
    if command in ("pull", "sync"):
        os.makedirs(_local(root), exist_ok=True)
    mirror_clocks = _has_files(key_dir)
    if command in ("push", "sync") and not mirror_clocks:
        out.write("store sync: nothing under {0}; clocks files not mirrored\n".format(key_dir))
    for argv in commands(command, root, key, remote, tool, dry_run, mirror_clocks):
        out.write("store sync: " + subprocess.list2cmdline(argv) + "\n")
        out.flush()
        done = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if tool == "rsync" and command in ("check", "unpushed"):
            changes = [line for line in done.stdout.splitlines() if line.strip()]
            out.write(done.stdout)
            if done.returncode == 0 and changes:
                return 1
        elif done.stdout.strip():
            out.write(done.stdout if done.stdout.endswith("\n") else done.stdout + "\n")
        if done.returncode != 0:
            out.write("store sync: {0} exited {1}\n".format(argv[0], done.returncode))
            return done.returncode
    return 0


def _cli(argv):
    parser = argparse.ArgumentParser(prog="sync.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--root", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    parser.add_argument("--remote", default=None,
                        help="<rclone remote or ssh host>:<path>, default $" + REMOTE_ENV
                        + " or " + DEFAULT_REMOTE)
    parser.add_argument("--key", default=None, help="this machine's key (bench_key.py)")
    parser.add_argument("--tool", choices=TOOLS, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    key = args.key
    if not key:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runner_scripts"))
        from bench_key import dataset_key
        key = dataset_key()
        if key.endswith("_unknown-gpu"):
            raise SystemExit("store sync: the GPU is unnamed; pass --key")
    return run(args.command, args.root, key, args.remote, args.tool, args.dry_run)


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
