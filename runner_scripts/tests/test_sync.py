"""sync.py and box_prune.py: the argv of every command for both tools, the availability checks, the box-side command lines, and a real rclone round trip between two local trees: push copies this key and its clocks without deleting and then has the box prune the clock logs no row names under the key's lock, pull brings the whole tree back and keeps a newer local file, prune deletes what is gone locally, check reports drift, unpushed reports only what the box lacks."""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "sync"))

import box_prune  # noqa: E402
import sync  # noqa: E402

KEY = "windows_RTX-4070-SUPER"
OTHER = "linux_RTX-2060-SUPER"
SYNC_PY = os.path.join(os.path.dirname(os.path.dirname(HERE)), "sync", "sync.py")
BOX_PY = os.path.join(os.path.dirname(os.path.dirname(HERE)), "sync", "box_prune.py")
REMOTE = "box:/srv/gpuode/data"
OLD_RUN = KEY + "_20260901T000000Z"
NEW_RUN = KEY + "_20260916T000000Z"


def _touch(root, relative, text="x"):
    path = os.path.join(root, *relative.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)
    return path


def _results(root, relative, runs):
    """A results parquet naming the given runs (None for a file from before the run column)."""
    path = os.path.join(root, *relative.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if runs is None:
        table = pa.table({"min_ms": pa.array([1.0], pa.float64())})
    else:
        table = pa.table({"min_ms": pa.array([1.0] * len(runs), pa.float64()),
                          "run": pa.array(list(runs), pa.string())})
    pq.write_table(table, path)
    return path


def _age(path, days):
    old = time.time() - days * 86400
    os.utime(path, (old, old))


def _files(root):
    """The tree's files, the box's key lock files left out."""
    found = []
    for base, _, names in os.walk(root):
        for name in names:
            if name.startswith(".sync_") and name.endswith(".lock"):
                continue
            found.append(os.path.relpath(os.path.join(base, name), root).replace("\\", "/"))
    return sorted(found)


class Commands(unittest.TestCase):
    def test_rclone_argv(self):
        push, clocks = sync.commands("push", "d", KEY, REMOTE, "rclone")
        self.assertEqual(push[:2], ["rclone", "copy"])
        self.assertTrue(push[2].endswith("key=" + KEY))
        self.assertEqual(push[3], REMOTE + "/key=" + KEY)
        self.assertIn("*.partial", push)
        self.assertIn("*.lock/**", push)
        # The clocks files are copied, never mirrored: what the box keeps is the box's decision.
        self.assertEqual(clocks[:2], ["rclone", "copy"])
        self.assertEqual(clocks[3], REMOTE + "/clocks")
        self.assertEqual(clocks[-4:], ["--include", "*_" + KEY + ".csv", "--include", KEY + "_*.csv"])
        (pull,) = sync.commands("pull", "d", KEY, REMOTE, "rclone")
        self.assertEqual(pull[:3], ["rclone", "copy", REMOTE])
        self.assertIn("--update", pull)
        self.assertIn("*.lock", pull)
        self.assertNotIn("key=" + KEY + "/**", pull)
        prune, prune_clocks = sync.commands("prune", "d", KEY, REMOTE, "rclone")
        self.assertEqual(prune[:2], ["rclone", "sync"])
        self.assertEqual(prune[3], REMOTE + "/key=" + KEY)
        self.assertEqual(prune_clocks, clocks)
        (check,) = sync.commands("check", "d", KEY, REMOTE, "rclone")
        self.assertEqual(check[:2], ["rclone", "check"])
        self.assertNotIn("--one-way", check)
        (unpushed,) = sync.commands("unpushed", "d", KEY, REMOTE, "rclone")
        self.assertEqual(unpushed[:4], check[:4])
        self.assertIn("--one-way", unpushed)
        self.assertIn("--dry-run", sync.commands("push", "d", KEY, "r:/p", "rclone", dry_run=True)[0])
        self.assertIn("--dry-run", sync.commands("prune", "d", KEY, "r:/p", "rclone", dry_run=True)[1])
        self.assertEqual(len(sync.commands("sync", "d", KEY, "r:/p", "rclone")), 3)

    def test_rsync_argv(self):
        push, clocks = sync.commands("push", "d", KEY, REMOTE, "rsync")
        self.assertEqual(push[:2], ["rsync", "-a"])
        self.assertNotIn("--delete", push)
        self.assertTrue(push[-2].endswith("key=" + KEY + "/"))
        self.assertEqual(push[-1], REMOTE + "/key=" + KEY + "/")
        self.assertEqual(clocks[1:8], ["-a", "--include", "*_" + KEY + ".csv", "--include", KEY + "_*.csv",
                                       "--exclude", "*"])
        self.assertNotIn("--delete", clocks)
        (pull,) = sync.commands("pull", "d", KEY, REMOTE, "rsync")
        self.assertEqual(pull[1], "-au")
        self.assertNotIn("key=" + KEY + "/", pull)
        self.assertEqual(pull[-2], REMOTE + "/")
        prune, prune_clocks = sync.commands("prune", "d", KEY, REMOTE, "rsync")
        self.assertIn("--delete", prune)
        self.assertNotIn("--dry-run", prune)
        self.assertEqual(prune_clocks, clocks)
        (check,) = sync.commands("check", "d", KEY, REMOTE, "rsync")
        self.assertIn("--dry-run", check)
        self.assertIn("--itemize-changes", check)
        (unpushed,) = sync.commands("unpushed", "d", KEY, REMOTE, "rsync")
        self.assertIn("--dry-run", unpushed)
        self.assertNotIn("--delete", unpushed)
        self.assertEqual(unpushed[-2:], check[-2:])

    def test_remote_host(self):
        self.assertEqual(sync.remote_host(REMOTE), "box")
        self.assertEqual(sync.remote_host("/srv/gpuode/data"), "")
        if sys.platform == "win32":
            self.assertEqual(sync.remote_host("C:/tmp/store"), "")

    def test_unavailable(self):
        self.assertIn("PATH", sync.unavailable("x:/p", tool="no-such-tool") or "PATH")
        old = os.environ.get(sync.REMOTE_ENV)
        os.environ[sync.REMOTE_ENV] = "elsewhere:/tree"
        try:
            self.assertEqual(sync.remote_default(), "elsewhere:/tree")
        finally:
            if old is None:
                del os.environ[sync.REMOTE_ENV]
            else:
                os.environ[sync.REMOTE_ENV] = old

    def test_the_box_side_command_runs_over_ssh_or_in_process(self):
        # An ssh Host of the remote's name serves rsync users.
        box = sync.Box(REMOTE, "rsync")
        self.assertFalse(box.local)
        self.assertEqual(box.ssh, ["ssh", "-o", "BatchMode=yes", "box"])
        self.assertEqual((box.root, box.script, box.python),
                         ("/srv/gpuode/data", "/srv/gpuode/box_prune.py", "/srv/gpuode/venv/bin/python3"))
        self.assertEqual(box.command("serve", KEY)[len(box.ssh):],
                         ["/srv/gpuode/venv/bin/python3", "/srv/gpuode/box_prune.py", "serve",
                          "--root", "/srv/gpuode/data", "--key", KEY])
        self.assertEqual(box.ship("rsync"), ["rsync", sync.BOX_SCRIPT, "box:/srv/gpuode/box_prune.py"])
        # An rclone sftp remote gives the ssh user, host and key.
        sftp = {"box": {"type": "sftp", "host": "chris-linux-dual", "user": "cca79",
                        "key_file": "C:/Users/me/.ssh/id_ed25519", "known_hosts_file": "C:/Users/me/.ssh/known_hosts"}}
        with mock.patch.object(sync, "rclone_remotes", return_value=sftp):
            box = sync.Box(REMOTE, "rclone")
        self.assertEqual(box.ssh, ["ssh", "-o", "BatchMode=yes", "-i", "C:/Users/me/.ssh/id_ed25519",
                                   "-o", "UserKnownHostsFile=C:/Users/me/.ssh/known_hosts", "cca79@chris-linux-dual"])
        self.assertEqual(box.ship("rclone"), ["rclone", "copyto", sync.BOX_SCRIPT, "box:/srv/gpuode/box_prune.py"])
        self.assertEqual(box.ship("rclone", dry_run=True)[-1], "--dry-run")
        # An alias of a local path, the box's own rclone remote, runs in-process against that path.
        alias = {"box": {"type": "alias", "remote": "/"}}
        with mock.patch.object(sync, "rclone_remotes", return_value=alias):
            box = sync.Box(REMOTE, "rclone")
        self.assertTrue(box.local)
        self.assertIsNone(box.ship("rclone"))
        self.assertEqual(box.root.replace("\\", "/").split(":")[-1], "/srv/gpuode/data")
        # A plain path is local for either tool.
        box = sync.Box(os.path.join(tempfile.gettempdir(), "store"), "rclone")
        self.assertTrue(box.local)
        self.assertEqual(box.python, sys.executable)


class BoxPrune(unittest.TestCase):
    """box_prune.py against a tree on disk: the orphan rule, the failing-read guard, the lock and the serve protocol."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="box_prune_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = os.path.join(self.tmp, "data")
        _results(self.root, "key={0}/package=cubie/results/lorenz__tsit5.parquet".format(KEY), [NEW_RUN, ""])
        _results(self.root, "key={0}/package=jax/results/lorenz__tsit5.parquet".format(KEY), None)
        _results(self.root, "key={0}/package=jax/results/lorenz__tsit5.parquet".format(OTHER),
                 [OTHER + "_20260901T000000Z"])
        self.logs = {}
        for name in (OLD_RUN, NEW_RUN, KEY + "_20260801T000000Z", OTHER + "_20260901T000000Z",
                     OTHER + "_20260801T000000Z", "calibration_" + KEY):
            self.logs[name] = _touch(self.root, "clocks/" + name + ".csv")
            _age(self.logs[name], 3)
        self.fresh = _touch(self.root, "clocks/" + KEY + "_20260915T000000Z.csv")

    def test_the_orphans_are_this_keys_old_logs_no_row_names(self):
        self.assertEqual(box_prune.runs_named(self.root, KEY), {NEW_RUN})
        self.assertEqual(box_prune.orphan_logs(self.root, KEY), [KEY + "_20260801T000000Z.csv", OLD_RUN + ".csv"])
        # Another key's logs, the calibration file, a named run and a fresh log are never orphans.
        self.assertEqual(box_prune.orphan_logs(self.root, OTHER), [OTHER + "_20260801T000000Z.csv"])
        self.assertEqual(box_prune.orphan_logs(self.root, KEY, min_age_s=0)[-1], KEY + "_20260915T000000Z.csv")
        self.assertEqual(box_prune.prune(self.root, KEY, dry_run=True), [KEY + "_20260801T000000Z.csv", OLD_RUN + ".csv"])
        self.assertTrue(os.path.exists(self.logs[OLD_RUN]))
        self.assertEqual(box_prune.prune(self.root, KEY), [KEY + "_20260801T000000Z.csv", OLD_RUN + ".csv"])
        self.assertFalse(os.path.exists(self.logs[OLD_RUN]))
        for name in (NEW_RUN, OTHER + "_20260901T000000Z", OTHER + "_20260801T000000Z", "calibration_" + KEY):
            self.assertTrue(os.path.exists(self.logs[name]), name)
        self.assertTrue(os.path.exists(self.fresh))
        self.assertEqual(box_prune.orphan_logs(os.path.join(self.tmp, "absent"), KEY), [])

    def test_a_results_file_that_cannot_be_read_stops_the_prune(self):
        # An unreadable file might name a run, so nothing is deleted.
        _touch(self.root, "key={0}/package=cpp/results/broken.parquet".format(KEY), "not parquet")
        with self.assertRaises(Exception):
            box_prune.prune(self.root, KEY)
        self.assertTrue(os.path.exists(self.logs[OLD_RUN]))

    def test_the_key_lock_excludes_a_second_holder_until_released(self):
        holder = subprocess.Popen([sys.executable, BOX_PY, "serve", "--root", self.root, "--key", KEY],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.addCleanup(holder.kill)
        self.assertEqual(holder.stdout.readline().strip(), "held")
        with self.assertRaises(TimeoutError):
            with box_prune.KeyLock(self.root, KEY, timeout=0.5):
                pass
        # Another key's lock is free.
        with box_prune.KeyLock(self.root, OTHER, timeout=0.5):
            pass
        # Closing stdin without `prune` releases the lock and deletes nothing.
        holder.stdin.close()
        out = holder.stdout.read()
        self.assertEqual(holder.wait(30), 0)
        self.assertEqual(out.split(), ["released"])
        self.assertTrue(os.path.exists(self.logs[OLD_RUN]))
        with box_prune.KeyLock(self.root, KEY, timeout=0.5):
            pass

    def test_serve_prunes_on_the_prune_line(self):
        serve = subprocess.Popen([sys.executable, BOX_PY, "serve", "--root", self.root, "--key", KEY],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.addCleanup(serve.kill)
        self.assertEqual(serve.stdout.readline().strip(), "held")
        out, _ = serve.communicate("prune --dry-run\n", timeout=30)
        self.assertEqual(out.split(), [KEY + "_20260801T000000Z.csv", OLD_RUN + ".csv", "released"])
        self.assertTrue(os.path.exists(self.logs[OLD_RUN]))
        serve = subprocess.Popen([sys.executable, BOX_PY, "serve", "--root", self.root, "--key", KEY],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.addCleanup(serve.kill)
        self.assertEqual(serve.stdout.readline().strip(), "held")
        out, _ = serve.communicate("prune\n", timeout=30)
        self.assertEqual(out.split(), [KEY + "_20260801T000000Z.csv", OLD_RUN + ".csv", "released"])
        self.assertFalse(os.path.exists(self.logs[OLD_RUN]))
        probe = subprocess.run([sys.executable, BOX_PY, "probe", "--root", self.root, "--key", KEY],
                               capture_output=True, text=True)
        self.assertEqual((probe.returncode, probe.stdout.strip()), (0, "ready"))
        absent = subprocess.run([sys.executable, BOX_PY, "probe", "--root", os.path.join(self.tmp, "no"), "--key", KEY],
                                capture_output=True, text=True)
        self.assertEqual(absent.returncode, 2)
        self.assertEqual(sync.box_ready(self.root), "")
        self.assertIn("no store", sync.box_ready(os.path.join(self.tmp, "no")))


@unittest.skipUnless(shutil.which("rclone"), "rclone is not installed")
class RcloneRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.local = os.path.join(self.tmp, "data")
        self.remote = os.path.join(self.tmp, "remote").replace("\\", "/")
        self.logs_dir = os.path.join(self.tmp, "logs")
        _results(self.local, "key={0}/package=cubie/results/lorenz__tsit5.parquet".format(KEY), [NEW_RUN])
        _touch(self.local, "key={0}/package=cubie/finals/abc.parquet".format(KEY))
        _touch(self.local, "key={0}/package=cubie/results/lorenz__rk4.parquet.partial".format(KEY))
        os.makedirs(os.path.join(self.local, "key=" + KEY, "package=cubie", "results",
                                 "lorenz__rk4.parquet.lock"))
        _touch(self.local, "key={0}/package=jax/results/stale.parquet".format(OTHER), "old")
        _touch(self.local, "clocks/calibration_{0}.csv".format(KEY))
        _touch(self.local, "clocks/{0}.csv".format(NEW_RUN))
        _touch(self.local, "clocks/calibration_{0}.csv".format(OTHER), "mine-not-to-push")
        _touch(self.local, "clocks/{0}_20260916T000000Z.csv".format(OTHER), "mine-not-to-push")
        _results(self.remote, "key={0}/package=cubie/results/gone.parquet".format(KEY), [])
        _touch(self.remote, "key={0}/package=jax/results/lorenz__tsit5.parquet".format(OTHER))
        newer = _touch(self.remote, "key={0}/package=jax/results/stale.parquet".format(OTHER), "new")
        # The box's copy is newer than the local one by a clear margin.
        os.utime(newer, (time.time() + 60, time.time() + 60))
        _touch(self.remote, "clocks/lightload_{0}.csv".format(OTHER))
        # An old log of this key that no row names, one of the other key's, and this key's old log kept locally too.
        _age(_touch(self.remote, "clocks/{0}.csv".format(OLD_RUN), "orphan"), 3)
        _age(_touch(self.remote, "clocks/{0}_20260901T000000Z.csv".format(OTHER), "not-mine"), 3)
        _age(_touch(self.local, "clocks/{0}.csv".format(OLD_RUN), "orphan"), 3)
        os.makedirs(os.path.join(self.logs_dir, OLD_RUN))
        os.makedirs(os.path.join(self.logs_dir, NEW_RUN))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_sync(self, command, **kw):
        out = io.StringIO()
        code = sync.run(command, self.local, KEY, self.remote, tool="rclone", out=out, logs_dir=self.logs_dir, **kw)
        return code, out.getvalue()

    def test_push_keeps_remote_files_and_the_box_prunes_the_logs_no_row_names(self):
        code, text = self.run_sync("push")
        self.assertEqual(code, 0, text)
        self.assertEqual(_files(self.remote), [
            "clocks/calibration_{0}.csv".format(KEY),
            "clocks/lightload_{0}.csv".format(OTHER),
            "clocks/{0}_20260901T000000Z.csv".format(OTHER),
            "clocks/{0}.csv".format(NEW_RUN),
            "key={0}/package=jax/results/lorenz__tsit5.parquet".format(OTHER),
            "key={0}/package=jax/results/stale.parquet".format(OTHER),
            "key={0}/package=cubie/finals/abc.parquet".format(KEY),
            "key={0}/package=cubie/results/gone.parquet".format(KEY),
            "key={0}/package=cubie/results/lorenz__tsit5.parquet".format(KEY),
        ])
        self.assertFalse(os.path.exists(os.path.join(
            self.remote, "key=" + KEY, "package=cubie", "results", "lorenz__rk4.parquet.lock")))
        # The box decided from its own rows: the orphan went there, and the mirror and logs/ followed.
        self.assertIn("pruned 1 clock log(s)", text)
        self.assertFalse(os.path.exists(os.path.join(self.local, "clocks", OLD_RUN + ".csv")))
        self.assertFalse(os.path.exists(os.path.join(self.logs_dir, OLD_RUN)))
        self.assertTrue(os.path.exists(os.path.join(self.logs_dir, NEW_RUN)))
        # The lock file stays for the next holder; the pull's excludes keep it off every mirror.
        self.assertTrue(os.path.isfile(os.path.join(self.remote, ".sync_" + KEY + ".lock")))
        with box_prune.KeyLock(self.remote, KEY, timeout=0.5):
            pass

    def test_a_dry_run_push_lists_what_the_box_would_prune(self):
        code, text = self.run_sync("push", dry_run=True)
        self.assertEqual(code, 0, text)
        self.assertIn("would prune " + OLD_RUN + ".csv", text)
        self.assertTrue(os.path.exists(os.path.join(self.remote, "clocks", OLD_RUN + ".csv")))
        self.assertTrue(os.path.exists(os.path.join(self.local, "clocks", OLD_RUN + ".csv")))

    def test_a_failed_upload_prunes_nothing(self):
        # The remote clocks dir is a file, so the second copy fails after the first succeeded.
        shutil.rmtree(os.path.join(self.remote, "clocks"))
        _touch(self.remote, "clocks", "in the way")
        code, text = self.run_sync("push")
        self.assertNotEqual(code, 0, text)
        self.assertNotIn("pruned", text)
        self.assertTrue(os.path.exists(os.path.join(self.local, "clocks", OLD_RUN + ".csv")))
        self.assertTrue(os.path.exists(os.path.join(self.logs_dir, OLD_RUN)))
        self.assertIn("key={0}/package=cubie/results/lorenz__tsit5.parquet".format(KEY), _files(self.remote))
        # The lock was released with the failure.
        with box_prune.KeyLock(self.remote, KEY, timeout=0.5):
            pass

    def test_a_results_file_the_box_cannot_read_stops_the_prune_and_fails_the_push(self):
        _touch(self.remote, "key={0}/package=cpp/results/broken.parquet".format(KEY), "not parquet")
        code, text = self.run_sync("push")
        self.assertEqual(code, 2, text)
        self.assertTrue(os.path.exists(os.path.join(self.remote, "clocks", OLD_RUN + ".csv")))
        self.assertIn("key={0}/package=cubie/results/lorenz__tsit5.parquet".format(KEY), _files(self.remote))

    def test_pull_brings_own_key_back(self):
        shutil.rmtree(os.path.join(self.local, "key=" + KEY))
        code, text = self.run_sync("pull")
        self.assertEqual(code, 0, text)
        self.assertEqual(_files(self.local), [
            "clocks/calibration_{0}.csv".format(OTHER),
            "clocks/calibration_{0}.csv".format(KEY),
            "clocks/lightload_{0}.csv".format(OTHER),
            "clocks/{0}_20260901T000000Z.csv".format(OTHER),
            "clocks/{0}_20260916T000000Z.csv".format(OTHER),
            "clocks/{0}.csv".format(OLD_RUN),
            "clocks/{0}.csv".format(NEW_RUN),
            "key={0}/package=jax/results/lorenz__tsit5.parquet".format(OTHER),
            "key={0}/package=jax/results/stale.parquet".format(OTHER),
            "key={0}/package=cubie/results/gone.parquet".format(KEY),
        ])
        with open(os.path.join(self.local, "key=" + OTHER, "package=jax", "results",
                               "stale.parquet")) as handle:
            self.assertEqual(handle.read(), "new")

    def test_pull_keeps_a_newer_local_file_and_never_copies_a_lock(self):
        # The newer local lorenz__tsit5 stays; the older local stale.parquet is replaced.
        mine = _touch(self.local, "key={0}/package=cubie/results/lorenz__tsit5.parquet".format(KEY), "unpushed rows")
        theirs = _touch(self.remote, "key={0}/package=cubie/results/lorenz__tsit5.parquet".format(KEY), "box")
        _touch(self.remote, ".sync_{0}.lock".format(KEY), "")
        now = time.time()
        os.utime(mine, (now, now))
        os.utime(theirs, (now - 3600, now - 3600))
        stale = os.path.join(self.local, "key=" + OTHER, "package=jax", "results", "stale.parquet")
        os.utime(stale, (now - 3600, now - 3600))
        code, text = self.run_sync("pull")
        self.assertEqual(code, 0, text)
        with open(mine) as handle:
            self.assertEqual(handle.read(), "unpushed rows")
        with open(stale) as handle:
            self.assertEqual(handle.read(), "new")
        self.assertFalse(os.path.exists(os.path.join(self.local, ".sync_" + KEY + ".lock")))
        self.assertTrue(sync.partition_has_files(self.local, KEY))
        self.assertFalse(sync.partition_has_files(self.local, "no-such-key"))

    def test_prune_and_check(self):
        code, text = self.run_sync("check")
        self.assertNotEqual(code, 0, text)
        code, text = self.run_sync("prune", dry_run=True)
        self.assertEqual(code, 0, text)
        self.assertIn("key={0}/package=cubie/results/gone.parquet".format(KEY), _files(self.remote))
        code, text = self.run_sync("prune")
        self.assertEqual(code, 0, text)
        self.assertNotIn("key={0}/package=cubie/results/gone.parquet".format(KEY), _files(self.remote))
        self.assertIn("key={0}/package=cubie/finals/abc.parquet".format(KEY), _files(self.remote))
        # The box then prunes this key's logs no remaining row names; another machine's stay.
        self.assertNotIn("clocks/{0}.csv".format(OLD_RUN), _files(self.remote))
        self.assertIn("clocks/{0}.csv".format(NEW_RUN), _files(self.remote))
        self.assertIn("clocks/{0}_20260901T000000Z.csv".format(OTHER), _files(self.remote))
        self.assertIn("clocks/lightload_{0}.csv".format(OTHER), _files(self.remote))
        code, text = self.run_sync("check")
        self.assertEqual(code, 0, text)

    def test_unpushed_ignores_files_only_the_box_has(self):
        # A box-only file is not unpushed, though check reports it.
        os.remove(os.path.join(self.local, "key=" + KEY, "package=cubie", "finals", "abc.parquet"))
        os.remove(os.path.join(self.local, "key=" + KEY, "package=cubie", "results", "lorenz__tsit5.parquet"))
        _touch(self.remote, "key={0}/package=cubie/finals/abc.parquet".format(KEY))
        _touch(self.local, "key={0}/package=cubie/finals/abc.parquet".format(KEY))
        code, text = self.run_sync("unpushed")
        self.assertEqual(code, 0, text)
        code, text = self.run_sync("check")
        self.assertNotEqual(code, 0, text)
        # A local-only file and a file that differs are both unpushed.
        _results(self.local, "key={0}/package=cubie/results/lorenz__tsit5.parquet".format(KEY), [NEW_RUN])
        code, text = self.run_sync("unpushed")
        self.assertNotEqual(code, 0, text)
        code, text = self.run_sync("push")
        self.assertEqual(code, 0, text)
        code, text = self.run_sync("unpushed")
        self.assertEqual(code, 0, text)
        _touch(self.local, "key={0}/package=cubie/finals/abc.parquet".format(KEY), "rewritten")
        code, text = self.run_sync("unpushed")
        self.assertNotEqual(code, 0, text)
        shutil.rmtree(os.path.join(self.local, "key=" + KEY))
        code, text = self.run_sync("unpushed")
        self.assertEqual(code, 0, text)
        self.assertIn("nothing under", text)

    def test_prune_needs_a_partition_with_files(self):
        shutil.rmtree(os.path.join(self.local, "key=" + KEY))
        os.makedirs(os.path.join(self.local, "key=" + KEY))
        code, text = self.run_sync("prune")
        self.assertEqual(code, 2)
        self.assertIn("no files under", text)
        self.assertIn("key={0}/package=cubie/results/gone.parquet".format(KEY), _files(self.remote))

    def test_push_from_an_empty_mirror_leaves_the_box_rows_and_prunes_by_the_box_rows_alone(self):
        shutil.rmtree(self.local)
        code, text = self.run_sync("push")
        self.assertEqual(code, 0, text)
        self.assertIn("key={0}/package=cubie/results/gone.parquet".format(KEY), _files(self.remote))
        # The box's rows name no run, so its old log of this key is an orphan whatever the empty mirror holds.
        self.assertNotIn("clocks/{0}.csv".format(OLD_RUN), _files(self.remote))
        self.assertIn("clocks/{0}_20260901T000000Z.csv".format(OTHER), _files(self.remote))

    def test_sync_and_cli(self):
        code, text = self.run_sync("sync")
        self.assertEqual(code, 0, text)
        self.assertIn("key={0}/package=cubie/results/gone.parquet".format(KEY), _files(self.local))
        self.assertIn("key={0}/package=cubie/finals/abc.parquet".format(KEY), _files(self.remote))
        done = subprocess.run([sys.executable, SYNC_PY, "check", "--root", self.local,
                               "--remote", self.remote, "--key", KEY, "--tool", "rclone"],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)


if __name__ == "__main__":
    unittest.main()
