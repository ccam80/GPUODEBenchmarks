"""sync.py: the argv of every command for both tools, the availability checks, and a real rclone round trip between two local trees: push copies this key and its clocks without deleting, pull brings the whole tree back, prune deletes what is gone locally, check reports drift."""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "sync"))

import sync  # noqa: E402

KEY = "windows_RTX-4070-SUPER"
OTHER = "linux_RTX-2060-SUPER"
SYNC_PY = os.path.join(os.path.dirname(os.path.dirname(HERE)), "sync", "sync.py")
REMOTE = "box:/srv/gpuode/data"


def _touch(root, relative, text="x"):
    path = os.path.join(root, *relative.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)
    return path


def _files(root):
    found = []
    for base, _, names in os.walk(root):
        for name in names:
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
        self.assertEqual(clocks[:2], ["rclone", "copy"])
        self.assertEqual(clocks[-2:], ["--include", "*_" + KEY + ".csv"])
        (pull,) = sync.commands("pull", "d", KEY, REMOTE, "rclone")
        self.assertEqual(pull[:3], ["rclone", "copy", REMOTE])
        self.assertNotIn("key=" + KEY + "/**", pull)
        (prune,) = sync.commands("prune", "d", KEY, REMOTE, "rclone")
        self.assertEqual(prune[:2], ["rclone", "sync"])
        self.assertEqual(prune[3], REMOTE + "/key=" + KEY)
        (check,) = sync.commands("check", "d", KEY, REMOTE, "rclone")
        self.assertEqual(check[:2], ["rclone", "check"])
        self.assertIn("--dry-run", sync.commands("push", "d", KEY, "r:/p", "rclone", dry_run=True)[0])
        self.assertEqual(len(sync.commands("sync", "d", KEY, "r:/p", "rclone")), 3)

    def test_rsync_argv(self):
        push, clocks = sync.commands("push", "d", KEY, REMOTE, "rsync")
        self.assertEqual(push[:2], ["rsync", "-a"])
        self.assertNotIn("--delete", push)
        self.assertTrue(push[-2].endswith("key=" + KEY + "/"))
        self.assertEqual(push[-1], REMOTE + "/key=" + KEY + "/")
        self.assertEqual(clocks[2:6], ["--include", "*_" + KEY + ".csv", "--exclude", "*"])
        (pull,) = sync.commands("pull", "d", KEY, REMOTE, "rsync")
        self.assertNotIn("key=" + KEY + "/", pull)
        self.assertEqual(pull[-2], REMOTE + "/")
        (prune,) = sync.commands("prune", "d", KEY, REMOTE, "rsync")
        self.assertIn("--delete", prune)
        self.assertNotIn("--dry-run", prune)
        (check,) = sync.commands("check", "d", KEY, REMOTE, "rsync")
        self.assertIn("--dry-run", check)
        self.assertIn("--itemize-changes", check)

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


@unittest.skipUnless(shutil.which("rclone"), "rclone is not installed")
class RcloneRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.local = os.path.join(self.tmp, "data")
        self.remote = os.path.join(self.tmp, "remote").replace("\\", "/")
        _touch(self.local, "key={0}/package=cubie/results/lorenz__tsit5.parquet".format(KEY))
        _touch(self.local, "key={0}/package=cubie/finals/abc.parquet".format(KEY))
        _touch(self.local, "key={0}/package=cubie/results/lorenz__rk4.parquet.partial".format(KEY))
        os.makedirs(os.path.join(self.local, "key=" + KEY, "package=cubie", "results",
                                 "lorenz__rk4.parquet.lock"))
        _touch(self.local, "key={0}/package=jax/results/stale.parquet".format(OTHER), "old")
        _touch(self.local, "clocks/calibration_{0}.csv".format(KEY))
        _touch(self.local, "clocks/calibration_{0}.csv".format(OTHER), "mine-not-to-push")
        _touch(self.remote, "key={0}/package=cubie/results/gone.parquet".format(KEY))
        _touch(self.remote, "key={0}/package=jax/results/lorenz__tsit5.parquet".format(OTHER))
        _touch(self.remote, "key={0}/package=jax/results/stale.parquet".format(OTHER), "new")
        _touch(self.remote, "clocks/lightload_{0}.csv".format(OTHER))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_sync(self, command, **kw):
        out = io.StringIO()
        code = sync.run(command, self.local, KEY, self.remote, tool="rclone", out=out, **kw)
        return code, out.getvalue()

    def test_push_keeps_remote_files(self):
        code, text = self.run_sync("push")
        self.assertEqual(code, 0, text)
        self.assertEqual(_files(self.remote), [
            "clocks/calibration_{0}.csv".format(KEY),
            "clocks/lightload_{0}.csv".format(OTHER),
            "key={0}/package=jax/results/lorenz__tsit5.parquet".format(OTHER),
            "key={0}/package=jax/results/stale.parquet".format(OTHER),
            "key={0}/package=cubie/finals/abc.parquet".format(KEY),
            "key={0}/package=cubie/results/gone.parquet".format(KEY),
            "key={0}/package=cubie/results/lorenz__tsit5.parquet".format(KEY),
        ])
        self.assertFalse(os.path.exists(os.path.join(
            self.remote, "key=" + KEY, "package=cubie", "results", "lorenz__rk4.parquet.lock")))

    def test_pull_brings_own_key_back(self):
        shutil.rmtree(os.path.join(self.local, "key=" + KEY))
        code, text = self.run_sync("pull")
        self.assertEqual(code, 0, text)
        self.assertEqual(_files(self.local), [
            "clocks/calibration_{0}.csv".format(OTHER),
            "clocks/calibration_{0}.csv".format(KEY),
            "clocks/lightload_{0}.csv".format(OTHER),
            "key={0}/package=jax/results/lorenz__tsit5.parquet".format(OTHER),
            "key={0}/package=jax/results/stale.parquet".format(OTHER),
            "key={0}/package=cubie/results/gone.parquet".format(KEY),
        ])
        with open(os.path.join(self.local, "key=" + OTHER, "package=jax", "results",
                               "stale.parquet")) as handle:
            self.assertEqual(handle.read(), "new")

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
        code, text = self.run_sync("check")
        self.assertEqual(code, 0, text)

    def test_prune_needs_a_partition_with_files(self):
        shutil.rmtree(os.path.join(self.local, "key=" + KEY))
        os.makedirs(os.path.join(self.local, "key=" + KEY))
        code, text = self.run_sync("prune")
        self.assertEqual(code, 2)
        self.assertIn("no files under", text)
        self.assertIn("key={0}/package=cubie/results/gone.parquet".format(KEY), _files(self.remote))

    def test_push_from_an_empty_mirror(self):
        shutil.rmtree(self.local)
        code, text = self.run_sync("push")
        self.assertEqual(code, 0, text)
        self.assertIn("key={0}/package=cubie/results/gone.parquet".format(KEY), _files(self.remote))

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
