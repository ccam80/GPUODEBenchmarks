"""GPU clock guard: pin the clocks (elevated shell) from the per-GPU table, sample them at 25 Hz into a UTC-stamped CSV, and judge any window of that log for the rows recorded in it."""

import bisect
import os
import platform
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone

CLOCK_CONF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "gpu_clocks.conf")
TOL_MHZ = 15            # one clock step; anything beyond this is real drift
SAMPLE_MS = 40          # nvidia-smi -lms period; one process saturates near 20 ms
COVERAGE_GAP_S = 1.0    # a window counts as observed only when a sample lies within this of each edge
FIELDS = ("timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,"
          "utilization.gpu,clocks_event_reasons.active")
HEADER = "utc,sm_mhz,mem_mhz,temp_c,power_w,util_pct,reasons"
SMI_STAMP = "%Y/%m/%d %H:%M:%S.%f"
UTC_STAMP = "%Y-%m-%dT%H:%M:%S.%fZ"
IDLE_BIT = 0x1
BAD_BITS = 0xEC


class ClockError(Exception):
    """The clocks cannot be locked as asked; the message says what to do."""


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


def driver_version():
    """The nvidia-smi driver version, '' when it cannot be read."""
    code, text = _smi(["--query-gpu=driver_version", "--format=csv,noheader"])
    return text.strip().splitlines()[0].strip() if code == 0 and text.strip() else ""


def supported(kind, mhz):
    """True when nvidia-smi lists mhz among the kind's (gr|mem) clocks."""
    code, text = _smi(["--query-supported-clocks={0}".format(kind),
                       "--format=csv,noheader,nounits"])
    if code != 0:
        return False
    return any(line.strip() == str(mhz) for line in text.splitlines())


def gpu_slug(dataset_key):
    """The GPU half of a dataset key, the conf table's row name."""
    return dataset_key.split("_", 1)[1] if "_" in dataset_key else dataset_key


def conf_row(gpu, conf=CLOCK_CONF):
    """(sm, mem) text of the GPU's row in the conf table; (None, None) without one."""
    if not os.path.isfile(conf):
        return None, None
    with open(conf, encoding="utf-8") as handle:
        for line in handle:
            cols = line.split()
            if len(cols) >= 2 and not cols[0].startswith("#") and cols[0] == gpu:
                return cols[1], (cols[2] if len(cols) > 2 else None)
    return None, None


def write_conf_row(gpu, sm, mem, note="", conf=CLOCK_CONF):
    """Replace the GPU's row in the conf table (append when absent) with the comment block directly above it, a comment line from note before the new row; returns the row text."""
    row = "{0} {1} {2}".format(gpu, sm, mem if mem else "")
    row = row.rstrip()
    lines = []
    if os.path.isfile(conf):
        with open(conf, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    kept = []
    for index, line in enumerate(lines):
        cols = line.split()
        if cols and not cols[0].startswith("#") and cols[0] == gpu:
            # The comment block directly above a row belongs to it.
            while kept and kept[-1].startswith("#"):
                kept.pop()
            continue
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    if kept:
        kept.append("")
    if note:
        kept.append("# " + note)
    kept.append(row)
    with open(conf, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(kept) + "\n")
    return row


def configure(dataset_key, explicit="", conf=CLOCK_CONF):
    """(sm, mem) targets from an explicit "sm[,mem]" or the per-GPU table; ClockError when there is no target or the GPU does not offer it."""
    gpu = gpu_slug(dataset_key)
    sm = mem = ""
    if explicit:
        parts = [p.strip() for p in explicit.split(",")]
        sm = parts[0]
        mem = parts[1] if len(parts) > 1 else ""
    else:
        sm, mem = conf_row(gpu, conf)
    if not sm:
        raise ClockError(
            "No clock target for '{0}' in {1} and none given. Measure one with "
            "runner_scripts/calibrate/calibrate_clocks.py (it writes the row) or pass "
            "--lock-clocks SM[,MEM]; a run never times unlocked.".format(gpu, conf))
    if not supported("gr", sm):
        raise ClockError("SM clock {0} MHz is not a supported clock on this GPU.".format(sm))
    if mem and not supported("mem", mem):
        raise ClockError("Memory clock {0} MHz is not a supported clock on this GPU.".format(mem))
    return sm, (mem or None)


class ClockGuard:
    """Lock and sample the GPU clocks for one run."""

    def __init__(self, sm=None, mem=None, tol_mhz=TOL_MHZ):
        self.sm, self.mem, self.tol = sm, mem, tol_mhz
        self.locked = False
        self.pm_restore = ""
        self.csv = None
        self.monitor = None
        self.reader = None

    def lock(self):
        """Pin the clocks; ClockError when the shell is not elevated or the driver refuses."""
        if not self.sm:
            return False
        if not is_admin():
            raise ClockError(
                "Not an elevated shell, so the clocks cannot be locked. Run from an "
                "Administrator console (or with passwordless sudo nvidia-smi); a run "
                "never times unlocked.")
        self.pm_restore = _smi(["--query-gpu=persistence_mode",
                                "--format=csv,noheader"])[1].strip()
        if not _privileged(["-pm", "1"])[0]:
            print("Could not enable persistence mode.")
        ok, text = _privileged(["-lgc", "{0},{0}".format(self.sm)])
        if not ok:
            raise ClockError("Failed to lock the SM clock to {0} MHz: {1}".format(
                self.sm, text.strip() or "nvidia-smi refused"))
        self.locked = True
        if self.mem:
            ok, text = _privileged(["-lmc", "{0},{0}".format(self.mem)])
            if not ok:
                print("Memory clock not lockable on this GPU; SM lock alone.")
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

    def start_monitor(self, csv_path, sample_ms=SAMPLE_MS):
        """Begin sampling into csv_path with UTC stamps; ClockError when the sampler dies at once, since no row could then record its clocks."""
        self.csv = csv_path
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        try:
            self.monitor = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=" + FIELDS, "--format=csv,nounits",
                 "-lms", str(int(sample_ms))],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                errors="replace")
        except OSError:
            self.monitor = None
        if self.monitor is not None:
            self.reader = threading.Thread(target=_relay, args=(self.monitor.stdout, csv_path),
                                           daemon=True)
            self.reader.start()
        time.sleep(1)
        if self.monitor is None or self.monitor.poll() is not None:
            self.stop_monitor()
            return False

    def stop_monitor(self):
        if self.monitor is not None:
            try:
                self.monitor.kill()
                self.monitor.wait(5)
            except Exception:
                pass
            self.monitor = None
        if self.reader is not None:
            self.reader.join(5)
            self.reader = None

    def status(self):
        """'locked SM=<mhz>[ MEM=<mhz>]' or 'unlocked'."""
        if not self.locked:
            return "unlocked"
        text = "locked SM={0}".format(self.sm)
        if self.mem:
            text += " MEM={0}".format(self.mem)
        return text


def parse_sample(line):
    """(utc datetime, sm, mem, temp, power, util, reasons) of one nvidia-smi CSV line; None for the header or a diagnostic."""
    f = [c.strip() for c in line.split(",")]
    if len(f) < 7 or not f[1][:1].isdigit():
        return None
    try:
        stamp = datetime.strptime(f[0], SMI_STAMP)
        sm, mem, util = int(f[1]), int(f[2]), int(f[5])
        reasons = int(f[6].lower().replace("0x", ""), 16)
    except ValueError:
        return None
    temp = _number(f[3])
    power = _number(f[4])
    # nvidia-smi stamps the machine's local time; the zone at that instant makes it UTC.
    utc = stamp.astimezone().astimezone(timezone.utc)
    return utc, sm, mem, temp, power, util, reasons


def _number(text):
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _relay(stream, csv_path):
    """Copy the sampler's lines to csv_path with UTC stamps, one flush per line."""
    with open(csv_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(HEADER + "\n")
        handle.flush()
        for line in stream:
            sample = parse_sample(line)
            if sample is None:
                continue
            utc, sm, mem, temp, power, util, reasons = sample
            handle.write("{0},{1},{2},{3:g},{4:g},{5},0x{6:016x}\n".format(
                utc.strftime(UTC_STAMP), sm, mem, temp, power, util, reasons))
            handle.flush()


def load_samples(csv_path):
    """The clock log as parallel lists sorted by time: {"t": epoch seconds, "sm", "mem", "temp", "reasons"}; empty when the file is absent."""
    times, sms, mems, temps, reasons = [], [], [], [], []
    if not os.path.isfile(csv_path):
        return {"t": times, "sm": sms, "mem": mems, "temp": temps, "reasons": reasons}
    with open(csv_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            f = [c.strip() for c in line.split(",")]
            if len(f) < 7 or not f[1][:1].isdigit():
                continue
            try:
                stamp = datetime.strptime(f[0], UTC_STAMP).replace(tzinfo=timezone.utc)
                sm, mem = int(f[1]), int(f[2])
                mask = int(f[6].lower().replace("0x", ""), 16)
            except ValueError:
                continue
            times.append(stamp.timestamp())
            sms.append(sm)
            mems.append(mem)
            temps.append(_number(f[3]))
            reasons.append(mask)
    order = sorted(range(len(times)), key=times.__getitem__)
    return {"t": [times[i] for i in order], "sm": [sms[i] for i in order],
            "mem": [mems[i] for i in order], "temp": [temps[i] for i in order],
            "reasons": [reasons[i] for i in order]}


def window_stats(samples, start, end, max_gap_s=COVERAGE_GAP_S):
    """{clock_sm_mhz, clock_sm_min_mhz, clock_throttled} over the samples between two epoch seconds, the window extended outward to the nearest sample on each side when that sample lies within max_gap_s of the edge; the SM statistics are over busy samples alone (NaN with none); None when the selected samples do not cover the window, that is when either edge is more than max_gap_s from its nearest selected sample or two consecutive selected samples are more than max_gap_s apart, so an interval the sampler never observed, stopped observing or skipped stays unannotated."""
    times = samples["t"]
    if not times:
        return None
    first = bisect.bisect_left(times, start)       # the first sample at or after start
    last = bisect.bisect_right(times, end) - 1     # the last sample at or before end
    if first > 0 and start - times[first - 1] <= max_gap_s:
        first -= 1
    if last + 1 < len(times) and times[last + 1] - end <= max_gap_s:
        last += 1
    if last < first:
        return None
    # Coverage: from start through every selected sample to end, no step wider than the gap.
    edges = [start] + times[first:last + 1] + [end]
    if any(later - earlier > max_gap_s for earlier, later in zip(edges, edges[1:])):
        return None
    busy = [samples["sm"][i] for i in range(first, last + 1)
            if not samples["reasons"][i] & IDLE_BIT]
    throttled = sum(1 for i in range(first, last + 1) if samples["reasons"][i] & BAD_BITS)
    nan = float("nan")
    return {"clock_sm_mhz": float(statistics.median(busy)) if busy else nan,
            "clock_sm_min_mhz": float(min(busy)) if busy else nan,
            "clock_throttled": throttled}


def drifted(stats, lock_mhz, tol=TOL_MHZ):
    """True when a locked row's window shows throttling or a busy sample more than tol below the lock."""
    if not lock_mhz or stats is None:
        return False
    if stats["clock_throttled"]:
        return True
    low = stats["clock_sm_min_mhz"]
    return low == low and low < int(lock_mhz) - tol
