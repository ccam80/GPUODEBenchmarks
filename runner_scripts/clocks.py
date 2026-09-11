"""GPU clock guard: pin the clocks (elevated shell), sample them at 1 Hz, and judge each timed step against its slice of the log."""

import os
import platform
import subprocess
import time
from datetime import datetime, timedelta

CLOCK_CONF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "gpu_clocks.conf")
TOL_MHZ = 15            # one clock step; anything beyond this is real drift
DRIFT_PCT = 1           # >this% of busy samples off target escalates to DRIFT
FIELDS = ("timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,"
          "utilization.gpu,clocks_event_reasons.active")
STAMP = "%Y/%m/%d %H:%M:%S"
IDLE_BIT = 0x1
BAD_BITS = 0xEC


def _smi(args, timeout=30):
    """nvidia-smi output and status; a missing binary reads as a failure."""
    try:
        out = subprocess.run(["nvidia-smi"] + list(args), capture_output=True,
                             text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""
    return out.returncode, out.stdout + out.stderr


def _privileged(args):
    """nvidia-smi as the administrator; refusal text counts as failure."""
    command = ["nvidia-smi"] + list(args)
    if platform.system() != "Windows" and os.geteuid() != 0:
        command = ["sudo", "-n"] + command
    try:
        out = subprocess.run(command, capture_output=True, text=True,
                             timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False, ""
    text = out.stdout + out.stderr
    if out.returncode != 0:
        return False, text
    for refusal in ("not supported", "Insufficient Permissions", "Unable to"):
        if refusal in text:
            return False, text
    return True, text


def is_admin():
    if platform.system() == "Windows":
        import ctypes
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
    if os.geteuid() == 0:
        return True
    # Per-command sudoers rules admit nvidia-smi alone, so probe with it.
    return _privileged(["-L"])[0]


def supported(kind, mhz):
    """True when nvidia-smi lists mhz among the kind's (gr|mem) clocks."""
    code, text = _smi(["--query-supported-clocks={0}".format(kind),
                       "--format=csv,noheader,nounits"])
    if code != 0:
        return False
    return any(line.strip() == str(mhz) for line in text.splitlines())


def configure(dataset_key, explicit="", conf=CLOCK_CONF):
    """(sm, mem) targets from an explicit "sm[,mem]" or the per-GPU table; (None, None) when unusable."""
    gpu = dataset_key.split("_", 1)[1] if "_" in dataset_key else dataset_key
    sm = mem = ""
    if explicit:
        parts = [p.strip() for p in explicit.split(",")]
        sm = parts[0]
        mem = parts[1] if len(parts) > 1 else ""
    elif os.path.isfile(conf):
        with open(conf, encoding="utf-8") as handle:
            for line in handle:
                cols = line.split()
                if len(cols) >= 2 and not cols[0].startswith("#") \
                        and cols[0] == gpu:
                    sm = cols[1]
                    mem = cols[2] if len(cols) > 2 else ""
                    break
    if not sm:
        print("No clock target for '{0}' in {1} and none given; continuing "
              "unlocked. Measure one with runner_scripts/calibrate/"
              "calibrate_clocks.py or pass --lock-clocks SM[,MEM]."
              .format(gpu, conf))
        return None, None
    if not supported("gr", sm):
        print("SM clock {0} MHz is not a supported clock on this GPU; "
              "continuing unlocked.".format(sm))
        return None, None
    if mem and not supported("mem", mem):
        print("Memory clock {0} MHz is not a supported clock on this GPU; "
              "continuing unlocked.".format(mem))
        return None, None
    return sm, (mem or None)


class ClockGuard:
    """Lock, sample and judge the GPU clocks for one run."""

    def __init__(self, sm=None, mem=None, tol_mhz=TOL_MHZ):
        self.sm, self.mem, self.tol = sm, mem, tol_mhz
        self.locked = False
        self.pm_restore = ""
        self.csv = None
        self.report = None
        self.monitor = None

    def lock(self):
        """Pin the clocks; False when not elevated or refused, with the manual command printed."""
        if not self.sm:
            return False
        if not is_admin():
            print("Not an elevated shell; clocks stay unlocked. Run first:\n"
                  "  nvidia-smi -pm 1\n  nvidia-smi -lgc {0},{0}".format(self.sm)
                  + ("\n  nvidia-smi -lmc {0},{0}".format(self.mem)
                     if self.mem else ""))
            return False
        self.pm_restore = _smi(["--query-gpu=persistence_mode",
                                "--format=csv,noheader"])[1].strip()
        if not _privileged(["-pm", "1"])[0]:
            print("Could not enable persistence mode.")
        ok, _ = _privileged(["-lgc", "{0},{0}".format(self.sm)])
        if not ok:
            print("Failed to lock the SM clock to {0} MHz.".format(self.sm))
            return False
        self.locked = True
        if self.mem:
            ok, text = _privileged(["-lmc", "{0},{0}".format(self.mem)])
            if not ok:
                print("Memory clock not lockable on this GPU; excluded from "
                      "the drift check.")
                self.mem = None
        now = _smi(["--query-gpu=clocks.sm,clocks.mem",
                    "--format=csv,noheader,nounits"])[1].strip()
        print("Clocks locked: SM={0} MHz{1}  (now reading {2})".format(
            self.sm, ", MEM={0} MHz".format(self.mem) if self.mem else "", now))
        return True

    def reset(self):
        if not self.locked:
            return
        self.locked = False
        if _privileged(["-rgc"])[0]:
            print("SM clock unlocked")
        else:
            print("Failed to reset the SM clock; run 'nvidia-smi -rgc' elevated")
        _privileged(["-rmc"])
        if self.pm_restore == "Disabled":
            _privileged(["-pm", "0"])

    def start_monitor(self, csv_path):
        """Begin 1 Hz sampling into csv_path; False when the sampler dies at once."""
        self.csv = csv_path
        self.report = os.path.join(os.path.dirname(csv_path),
                                   "clock_stability.tsv")
        open(self.report, "w").close()
        try:
            self.monitor = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=" + FIELDS, "--format=csv,nounits",
                 "-lms", "1000"],
                stdout=open(csv_path, "w"), stderr=open(csv_path + ".err", "w"))
        except OSError:
            self.monitor = None
        time.sleep(1)
        if self.monitor is None or self.monitor.poll() is not None:
            print("Clock monitor died immediately; drift will not be checked.")
            self.monitor = None
            return False
        return True

    def stop_monitor(self):
        if self.monitor is None:
            return
        try:
            self.monitor.kill()
            self.monitor.wait(5)
        except Exception:
            pass
        self.monitor = None

    @staticmethod
    def stamp(end=False):
        """Whole-second window edge in the sampler's timestamp format."""
        now = datetime.now().replace(microsecond=0)
        return now.strftime(STAMP) + (".999" if end else ".000")

    def check(self, start, end, label, critical=True):
        """Judge one window: True unless a critical step drifted; a row lands in the report."""
        if not self.csv or not os.path.isfile(self.csv) or not self.sm:
            return True
        verdict = slice_verdict(self.csv, start, end, int(self.sm),
                                int(self.mem) if self.mem else 0, self.tol)
        if verdict is None:
            return True
        with open(self.report, "a", encoding="utf-8") as handle:
            handle.write("\t".join(str(v) for v in (
                label, verdict["verdict"], verdict["n"], verdict["busy"],
                verdict["drift"], verdict["worst"], verdict["min_sm"],
                verdict["mem_drift"], verdict["throttled"],
                verdict["peak"])) + "\n")
        if verdict["verdict"] == "BLIP":
            print("  clock blip during {0}: {1}/{2} busy samples off {3} MHz "
                  "(worst {4} MHz, min {5} MHz)".format(
                      label, verdict["drift"], verdict["busy"], self.sm,
                      verdict["worst"], verdict["min_sm"]))
        elif verdict["verdict"] == "DRIFT":
            print("  {0}: clocks drifted during {1}: {2}/{3} busy samples off "
                  "{4} MHz ({5}%, worst {6} MHz low, min {7} MHz); throttled "
                  "{8}; peak {9}".format(
                      "ERROR" if critical else "WARNING", label,
                      verdict["drift"], verdict["busy"], self.sm,
                      verdict["pct"], verdict["worst"], verdict["min_sm"],
                      verdict["throttled"], verdict["peak"]))
            if critical:
                return False
        return True

    def report_text(self):
        """The per-step stability table, or "" when nothing was judged."""
        if not self.report or not os.path.isfile(self.report):
            return ""
        with open(self.report, encoding="utf-8") as handle:
            rows = [line.rstrip("\n").split("\t") for line in handle if line.strip()]
        if not rows:
            return ""
        target = "SM={0}".format(self.sm) if self.sm else "unlocked"
        if self.mem:
            target += " MEM={0}".format(self.mem)
        lines = ["CLOCK STABILITY  (target {0}, tol {1}MHz)".format(target, self.tol),
                 "{0:<26} {1:<7} {2:>8} {3:>8} {4:>9} {5:>10}".format(
                     "STEP", "STATUS", "SAMPLES", "OFF", "WORST", "PEAK")]
        for c in rows:
            if len(c) < 10:
                continue
            lines.append("{0:<26} {1:<7} {2:>8} {3:>8} {4:>9} {5:>10}".format(
                c[0], c[1], c[3], c[4], c[5] + "MHz", c[9]))
        return "\n".join(lines)


def slice_verdict(csv_path, start, end, sm, mem=0, tol=TOL_MHZ):
    """Drift statistics of the samples between two stamps; None when the window holds no sample."""
    n = busy = drift = worst = min_sm = mem_drift = throttled = 0
    tmax = pmax = 0.0
    with open(csv_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            f = [c.strip() for c in line.split(",")]
            if len(f) < 7 or not f[1][:1].isdigit():
                continue
            if f[0] < start or f[0] > end:
                continue
            try:
                sm_now = int(f[1])
            except ValueError:
                continue
            try:
                reasons = int(f[6].lower().replace("0x", ""), 16)
            except ValueError:
                reasons = 0
            n += 1
            for value, attr in ((f[3], "t"), (f[4], "p")):
                try:
                    number = float(value)
                except ValueError:
                    continue
                if attr == "t":
                    tmax = max(tmax, number)
                else:
                    pmax = max(pmax, number)
            if reasons & IDLE_BIT:
                continue
            busy += 1
            dev = abs(sm_now - sm)
            worst = max(worst, dev)
            if dev > tol:
                drift += 1
            min_sm = sm_now if min_sm == 0 else min(min_sm, sm_now)
            if mem:
                try:
                    if abs(int(f[2]) - mem) > tol:
                        mem_drift += 1
                except ValueError:
                    pass
            if reasons & BAD_BITS:
                throttled += 1
    if n == 0:
        return None
    pct = -(-drift * 100 // busy) if busy else 0
    if throttled or (pct > DRIFT_PCT and drift >= 3):
        verdict = "DRIFT"
    elif drift or mem_drift:
        verdict = "BLIP"
    else:
        verdict = "OK"
    return {"verdict": verdict, "n": n, "busy": busy, "drift": drift,
            "worst": worst, "min_sm": min_sm, "mem_drift": mem_drift,
            "throttled": throttled, "pct": pct,
            "peak": "{0:.0f}C/{1:.1f}W".format(tmax, pmax)}


def window_after(stamp, seconds):
    """The stamp `seconds` after another, for tests."""
    return (datetime.strptime(stamp[:19], STAMP)
            + timedelta(seconds=seconds)).strftime(STAMP) + stamp[19:]
