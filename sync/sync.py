"""The remote store, a data/ tree on the store box: pull copies the whole tree into the local mirror, leaving a local file that is newer than the box's alone; push copies this machine's key partition and its clocks files (calibration_<key>.csv and the <key>_<stamp>.csv log of each run) up, deleting nothing, then has the box prune the key's clock logs that no row on the box names (box_prune.py, run over ssh under the key's lock, which the push holds from its first copy; a failed upload prunes nothing), and drops the pruned logs from the mirror; prune mirrors this key (files gone locally are deleted on the box) and then prunes the same way; check lists the differences under this key; unpushed lists this key's local files missing from or differing on the box, ignoring files only the box has. rclone drives the `box:` remote, rsync the `box` ssh host when rclone is absent; the same <host>:<path> string names both, and the box-side script runs through ssh (the sftp remote's host, user and key) or in-process for a local tree. CLI: sync.py [--root DIR] [--remote HOST:PATH] [--key KEY] [--tool rclone|rsync] [--dry-run] pull | push | sync | prune | check | unpushed."""

import argparse
import json
import os
import shutil
import subprocess
import sys

REMOTE_ENV = "GPUODE_STORE_REMOTE"
DEFAULT_REMOTE = "box:/srv/gpuode/data"
# Files a store write leaves beside a parquet while it is in flight, and the box's key locks; never shipped.
TRANSIENT = ("*.partial", "*.lock", "*.lock/**")
TOOLS = ("rclone", "rsync")
COMMANDS = ("pull", "push", "sync", "prune", "check", "unpushed")
SETUP_HINT = "sync/README.md"
HERE = os.path.dirname(os.path.abspath(__file__))
BOX_SCRIPT = os.path.join(HERE, "box_prune.py")
REPO_ROOT = os.path.dirname(HERE)

sys.path.insert(0, HERE)
import box_prune  # noqa: E402


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


def rclone_remotes():
    """{name: config} of every configured rclone remote; {} when rclone cannot say."""
    dumped = subprocess.run(["rclone", "config", "dump"], capture_output=True, text=True)
    if dumped.returncode != 0:
        return {}
    try:
        return json.loads(dumped.stdout or "{}")
    except ValueError:
        return {}


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


class Box:
    """Where the store's box-side script runs: over ssh to the remote's host (an sftp remote's host, user and key, or an ssh Host of the same name), or in-process when the remote is a plain path or an alias of one. The script and its venv sit beside the data tree: <parent>/box_prune.py and <parent>/venv."""

    def __init__(self, remote, tool):
        host = remote_host(remote)
        path = remote.partition(":")[2] if host else remote
        self.host = host
        self.ssh = None
        if host and tool == "rclone":
            config = rclone_remotes().get(host, {})
            if config.get("type") == "alias" and not remote_host(config.get("remote", "")):
                self.host, path = "", _join(config["remote"], path.lstrip("/"))
            elif config.get("type") == "sftp":
                self.ssh = _sftp_ssh(config)
        if self.host and self.ssh is None:
            self.ssh = ["ssh", "-o", "BatchMode=yes", self.host]
        self.root = path if self.host else os.path.abspath(path)
        parent = self.root.rstrip("/").rsplit("/", 1)[0] if self.host else os.path.dirname(self.root)
        self.script = _join(parent, "box_prune.py") if self.host else os.path.join(parent, "box_prune.py")
        self.python = _join(parent, "venv", "bin", "python3") if self.host else sys.executable

    @property
    def local(self):
        return not self.host

    def ship(self, tool, dry_run=False):
        """The argv that copies box_prune.py to the box; None when the tree is local."""
        if self.local:
            return None
        if tool == "rclone":
            return ["rclone", "copyto", BOX_SCRIPT, self.host + ":" + self.script] + (["--dry-run"] if dry_run else [])
        return ["rsync"] + (["--dry-run"] if dry_run else []) + [BOX_SCRIPT, self.host + ":" + self.script]

    def command(self, verb, key, *flags):
        """The box-side command line: the venv's python on the shipped script."""
        return self.ssh + [self.python, self.script, verb, "--root", self.root, "--key", key] + list(flags)


def _sftp_ssh(config):
    argv = ["ssh", "-o", "BatchMode=yes"]
    if config.get("key_file"):
        argv += ["-i", config["key_file"]]
    if config.get("known_hosts_file"):
        argv += ["-o", "UserKnownHostsFile=" + config["known_hosts_file"]]
    if config.get("port"):
        argv += ["-p", str(config["port"])]
    target = config.get("host", "")
    if config.get("user"):
        target = config["user"] + "@" + target
    return argv + [target]


def box_ready(remote=None, tool=None, key="probe"):
    """'' when the box can run its script (the venv imports pyarrow and the tree exists), else why not."""
    remote = remote or remote_default()
    box = Box(remote, find_tool(tool))
    if box.local:
        return "" if os.path.isdir(box.root) else "no store at " + box.root
    probe = subprocess.run(box.ship(find_tool(tool)), capture_output=True, text=True)
    if probe.returncode != 0:
        return "cannot copy box_prune.py to the box: " + (probe.stdout + probe.stderr).strip()
    probe = subprocess.run(box.command("probe", key), capture_output=True, text=True)
    if probe.returncode != 0 or "ready" not in probe.stdout:
        return ("the box cannot run box_prune.py (run `sudo bash sync/store_box.sh` there): "
                + (probe.stdout + probe.stderr).strip())
    return ""


class Hold:
    """The key's lock on the box, held from before the first upload until the prune after it: in-process for a local tree, else box_prune.py serve over ssh."""

    def __init__(self, box, key, min_age_days=None, out=sys.stdout):
        self.box, self.key, self.out = box, key, out
        self.flags = [] if min_age_days is None else ["--min-age-days", str(min_age_days)]
        self.lock = None
        self.proc = None

    def __enter__(self):
        if self.box.local:
            self.lock = box_prune.KeyLock(self.box.root, self.key).__enter__()
            return self
        argv = self.box.command("serve", self.key, *self.flags)
        self.out.write("store sync: " + subprocess.list2cmdline(argv) + "\n")
        self.out.flush()
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
        if self.proc.stdout.readline().strip() != "held":
            self.proc.stdin.close()
            _, err = self.proc.communicate()
            raise RuntimeError("the box did not take the key's lock: " + err.strip())
        return self

    def prune(self, dry_run=False):
        """Prune under the held lock; returns the names of the logs pruned (or listed under dry_run)."""
        if self.box.local:
            days = float(self.flags[1]) if self.flags else box_prune.MIN_AGE_S / 86400.0
            return box_prune.prune(self.box.root, self.key, days * 86400.0, dry_run=dry_run)
        self.proc.stdin.write("prune" + (" --dry-run" if dry_run else "") + "\n")
        self.proc.stdin.close()
        out, err = self.proc.communicate()
        names = [line.strip() for line in out.splitlines() if line.strip()]
        if self.proc.returncode != 0 or not names or names[-1] != "released":
            raise RuntimeError("the box's prune failed: " + (out + err).strip())
        return names[:-1]

    def __exit__(self, *_):
        if self.lock is not None:
            self.lock.__exit__(None, None, None)
            self.lock = None
        if self.proc is not None and self.proc.poll() is None:
            # EOF makes serve release without pruning.
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


def commands(command, root, key, remote, tool, dry_run=False):
    """The argv list of a command for the tool; sync is a push then a pull; push and prune copy the clocks files, only prune deletes anything (under the key partition alone)."""
    if command == "sync":
        return commands("push", root, key, remote, tool, dry_run) + \
            commands("pull", root, key, remote, tool, dry_run)
    key_dir = "key=" + key
    # This machine's clocks files: calibration_<key>.csv and the <key>_<stamp>.csv log of each run.
    own_clocks = ["*_" + key + ".csv", key + "_*.csv"]
    dry = ["--dry-run"] if dry_run else []
    if tool == "rclone":
        transient = [flag for pattern in TRANSIENT for flag in ("--exclude", pattern)]
        includes = [flag for pattern in own_clocks for flag in ("--include", pattern)]
        clocks = ["rclone", "copy", _local(root, "clocks"), _join(remote, "clocks")] + includes + dry
        if command == "push":
            return [["rclone", "copy", _local(root, key_dir), _join(remote, key_dir)] + transient + dry, clocks]
        if command == "pull":
            return [["rclone", "copy", remote, _local(root), "--update"] + transient + dry]
        if command == "prune":
            return [["rclone", "sync", _local(root, key_dir), _join(remote, key_dir)] + transient + dry, clocks]
        if command == "check":
            return [["rclone", "check", _local(root, key_dir), _join(remote, key_dir)] + transient]
        if command == "unpushed":
            return [["rclone", "check", _local(root, key_dir), _join(remote, key_dir), "--one-way"] + transient]
    if tool == "rsync":
        transient = [flag for pattern in TRANSIENT[:2] for flag in ("--exclude", pattern)]
        includes = [flag for pattern in own_clocks for flag in ("--include", pattern)]
        clocks = ["rsync", "-a"] + includes + ["--exclude", "*"] + dry \
            + [_local(root, "clocks") + "/", _join(remote, "clocks") + "/"]
        if command == "push":
            return [["rsync", "-a"] + transient + dry
                    + [_local(root, key_dir) + "/", _join(remote, key_dir) + "/"], clocks]
        if command == "pull":
            return [["rsync", "-au"] + transient + dry
                    + [remote.rstrip("/") + "/", _local(root) + "/"]]
        if command == "prune":
            return [["rsync", "-a", "--delete"] + transient + dry
                    + [_local(root, key_dir) + "/", _join(remote, key_dir) + "/"], clocks]
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


def _run_argv(argv, tool, command, out):
    """Run one program; returns its exit code, 1 for an rsync check/unpushed that lists changes."""
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


def _forget_pruned(root, names, logs_dir, out):
    """Drop the pruned logs from the mirror and their run dirs from logs_dir."""
    for name in names:
        path = _local(root, "clocks", name)
        if os.path.isfile(path):
            os.remove(path)
        run_dir = os.path.join(logs_dir, name[:-4]) if logs_dir else ""
        if run_dir and os.path.isdir(run_dir):
            shutil.rmtree(run_dir, ignore_errors=True)
    if names:
        out.write("store sync: the box pruned {0} clock log(s) no row names: {1}\n".format(
            len(names), ", ".join(names)))


def run(command, root, key, remote=None, tool=None, dry_run=False, out=sys.stdout, logs_dir=None,
        min_age_days=None):
    """Run the command's programs in order, stopping at the first failure; returns the exit code (2 before anything ran). pull creates root, push creates the partition and clocks dirs, prune and check need a partition with files, unpushed is 0 without one. push and prune hold the key's lock on the box from their first copy through the box-side prune after it; logs_dir's <run>/ dirs of pruned logs go too."""
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
    if command in ("push", "sync", "prune"):
        upload = "prune" if command == "prune" else "push"
        box = Box(remote, tool)
        if box.local:
            os.makedirs(box.root, exist_ok=True)
        else:
            code = _run_argv(box.ship(tool, dry_run), tool, "ship", out)
            if code:
                return code
        try:
            with Hold(box, key, min_age_days, out) as hold:
                for argv in commands(upload, root, key, remote, tool, dry_run):
                    code = _run_argv(argv, tool, upload, out)
                    if code:
                        return code
                names = hold.prune(dry_run)
        except Exception as exc:  # noqa: BLE001 - a lock, ssh or box-side failure ends the push, reported
            out.write("store sync: {0}: {1}\n".format(type(exc).__name__, exc))
            return 2
        if not dry_run:
            _forget_pruned(root, names, logs_dir, out)
        elif names:
            out.write("store sync: the box would prune {0}\n".format(", ".join(names)))
        if command != "sync":
            return 0
        command = "pull"
    for argv in commands(command, root, key, remote, tool, dry_run):
        code = _run_argv(argv, tool, command, out)
        if code:
            return code
    return 0


def _cli(argv):
    parser = argparse.ArgumentParser(prog="sync.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--root", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--remote", default=None,
                        help="<rclone remote or ssh host>:<path>, default $" + REMOTE_ENV
                        + " or " + DEFAULT_REMOTE)
    parser.add_argument("--key", default=None, help="this machine's key (bench_key.py)")
    parser.add_argument("--tool", choices=TOOLS, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    key = args.key
    if not key:
        sys.path.insert(0, os.path.join(REPO_ROOT, "runner_scripts"))
        from bench_key import dataset_key
        key = dataset_key()
        if key.endswith("_unknown-gpu"):
            raise SystemExit("store sync: the GPU is unnamed; pass --key")
    return run(args.command, args.root, key, args.remote, args.tool, args.dry_run,
               logs_dir=os.path.join(REPO_ROOT, "logs"))


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
