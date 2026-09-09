"""sync.py: the push, pull and check argv for both tools, the availability checks, and a real rclone round trip between two local trees: the push mirrors this key and its clocks only, the pull brings the other keys without deleting, the check reports drift."""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import sync  # noqa: E402

KEY = "windows_RTX-4070-SUPER"
OTHER = "linux_RTX-2060-SUPER"
SYNC_PY = os.path.join(os.path.dirname(HERE), "sync.py")


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
        push, clocks = sync.commands("push", "d", KEY, "box:/srv/gpuode/data", "rclone")
        self.assertEqual(push[:2], ["rclone", "sync"])
        self.assertTrue(push[2].endswith("key=" + KEY))
        self.assertEqual(push[3], "box:/srv/gpuode/data/key=" + KEY)
        self.assertIn("*.partial", push)
        self.assertIn("*.lock/**", push)
        self.assertEqual(clocks[:2], ["rclone", "copy"])
        self.assertEqual(clocks[-2:], ["--include", "*_" + KEY + ".csv"])
        (pull,) = sync.commands("pull", "d", KEY, "box:/srv/gpuode/data", "rclone")
        self.assertEqual(pull[:3], ["rclone", "copy", "box:/srv/gpuode/data"])
        self.assertIn("key=" + KEY + "/**", pull)
        (check,) = sync.commands("check", "d", KEY, "box:/srv/gpuode/data", "rclone")
        self.assertEqual(check[:2], ["rclone", "check"])
        self.assertIn("--dry-run", sync.commands("push", "d", KEY, "r:/p", "rclone", dry_run=True)[0])
        self.assertEqual(len(sync.commands("sync", "d", KEY, "r:/p", "rclone")), 3)

    def test_rsync_argv(self):
        push, clocks = sync.commands("push", "d", KEY, "box:/srv/gpuode/data", "rsync")
        self.assertEqual(push[:3], ["rsync", "-a", "--delete"])
        self.assertTrue(push[-2].endswith("key=" + KEY + "/"))
        self.assertEqual(push[-1], "box:/srv/gpuode/data/key=" + KEY + "/")
        self.assertEqual(clocks[2:6], ["--include", "*_" + KEY + ".csv", "--exclude", "*"])
        (pull,) = sync.commands("pull", "d", KEY, "box:/srv/gpuode/data", "rsync")
        self.assertEqual(pull[2:4], ["--exclude", "key=" + KEY + "/"])
        self.assertEqual(pull[-2], "box:/srv/gpuode/data/")
        (check,) = sync.commands("check", "d", KEY, "box:/srv/gpuode/data", "rsync")
        self.assertIn("--dry-run", check)
        self.assertIn("--itemize-changes", check)

    def test_remote_host(self):
        self.assertEqual(sync.remote_host("box:/srv/gpuode/data"), "box")
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

    def test_push_pull_check(self):
        code, text = self.run_sync("check")
        self.assertNotEqual(code, 0, text)
        code, text = self.run_sync("push")
        self.assertEqual(code, 0, text)
        self.assertEqual(_files(self.remote), [
            "clocks/calibration_{0}.csv".format(KEY),
            "clocks/lightload_{0}.csv".format(OTHER),
            "key={0}/package=jax/results/lorenz__tsit5.parquet".format(OTHER),
            "key={0}/package=jax/results/stale.parquet".format(OTHER),
            "key={0}/package=cubie/finals/abc.parquet".format(KEY),
            "key={0}/package=cubie/results/lorenz__tsit5.parquet".format(KEY),
        ])
        self.assertFalse(os.path.exists(os.path.join(
            self.remote, "key=" + KEY, "package=cubie", "results", "lorenz__rk4.parquet.lock")))
        code, text = self.run_sync("check")
        self.assertEqual(code, 0, text)
        code, text = self.run_sync("pull")
        self.assertEqual(code, 0, text)
        self.assertEqual(_files(self.local), [
            "clocks/calibration_{0}.csv".format(OTHER),
            "clocks/calibration_{0}.csv".format(KEY),
            "clocks/lightload_{0}.csv".format(OTHER),
            "key={0}/package=jax/results/lorenz__tsit5.parquet".format(OTHER),
            "key={0}/package=jax/results/stale.parquet".format(OTHER),
            "key={0}/package=cubie/finals/abc.parquet".format(KEY),
            "key={0}/package=cubie/results/lorenz__rk4.parquet.partial".format(KEY),
            "key={0}/package=cubie/results/lorenz__tsit5.parquet".format(KEY),
        ])
        with open(os.path.join(self.local, "key=" + OTHER, "package=jax", "results",
                               "stale.parquet")) as handle:
            self.assertEqual(handle.read(), "new")

    def test_dry_run_and_sync(self):
        code, text = self.run_sync("push", dry_run=True)
        self.assertEqual(code, 0, text)
        self.assertNotIn("key=" + KEY, [p.split("/")[0] for p in _files(self.remote)
                                        if p.endswith("abc.parquet")])
        code, text = self.run_sync("sync")
        self.assertEqual(code, 0, text)
        self.assertIn("key={0}/package=jax/results/lorenz__tsit5.parquet".format(OTHER),
                      _files(self.local))
        self.assertIn("key={0}/package=cubie/finals/abc.parquet".format(KEY),
                      _files(self.remote))

    def test_missing_partition(self):
        shutil.rmtree(os.path.join(self.local, "key=" + KEY))
        code, text = self.run_sync("push")
        self.assertEqual(code, 2)
        self.assertIn("no local partition", text)

    def test_cli(self):
        done = subprocess.run([sys.executable, SYNC_PY, "push", "--root", self.local,
                               "--remote", self.remote, "--key", KEY, "--tool", "rclone"],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("key={0}/package=cubie/finals/abc.parquet".format(KEY),
                      _files(self.remote))


if __name__ == "__main__":
    unittest.main()
