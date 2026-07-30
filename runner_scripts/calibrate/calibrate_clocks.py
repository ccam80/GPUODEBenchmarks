#!/usr/bin/env python3
"""Measure the clock a GPU holds under sustained load, for gpu_clocks.conf.

Builds clock_burn.cu, runs it under 1 Hz nvidia-smi sampling, and prints the row
to paste into runner_scripts/gpu_clocks.conf. Linux and Windows.

    python3 runner_scripts/calibrate/calibrate_clocks.py
"""

import csv
import glob
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime

MINUTES = 15          # below ~10 the heatsink may not have saturated
MATRIX = 4096         # SGEMM size; big enough to saturate the SMs
WARMUP_S = 300        # discarded before looking at the plateau
HEADROOM_PCT = 5      # applied only if the clock varied or throttled

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "runner_scripts"))
from bench_key import dataset_key                                # noqa: E402

# clocks_event_reasons.active bits. GpuIdle is expected; the ones below override
# a lock, so those samples say nothing about a sustainable rate.
BIT_GPU_IDLE = 0x1
THROTTLES = {0x4: "SwPowerCap", 0x8: "HwSlowdown", 0x20: "SwThermalSlowdown",
             0x40: "HwThermalSlowdown", 0x80: "HwPowerBrakeSlowdown"}

FIELDS = ("timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,"
          "utilization.gpu,clocks_event_reasons.active")


def smi(query):
    out = subprocess.run(["nvidia-smi", f"--query-{query}",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        return []
    return [l.strip() for l in out.stdout.splitlines() if l.strip()]


def find_nvcc():
    # Routinely installed but not on PATH; Windows versions its install dir.
    found = shutil.which("nvcc")
    if found:
        return found
    cands = sorted(glob.glob("/usr/local/cuda*/bin/nvcc")) + sorted(glob.glob(
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe"))
    return cands[-1] if cands else None


def build():
    nvcc = find_nvcc()
    if not nvcc:
        sys.exit("✗ nvcc not found")
    exe = os.path.join(HERE, "clock_burn.exe" if platform.system() == "Windows"
                             else "clock_burn")
    cmd = [nvcc, "-O3"]
    cap = smi("gpu=compute_cap")
    if cap and re.match(r"\d+\.\d+", cap[0]):
        cmd.append("-arch=sm_" + cap[0].replace(".", ""))
    cmd += [os.path.join(HERE, "clock_burn.cu"), "-lcublas", "-o", exe]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # On Windows the usual cause is nvcc without cl.exe on PATH (needs a
        # VS developer shell), which only the compiler output reveals.
        sys.exit("✗ build failed: {}\n{}".format(
            " ".join(cmd), (proc.stderr or proc.stdout).strip()))
    return exe


def run_load(exe, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as fh:
        # One long-lived sampler, not a query per second: it cannot fall behind
        # and leave gaps in the record.
        sampler = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,nounits",
             "-lms", "1000"], stdout=fh, stderr=subprocess.STDOUT)
        try:
            subprocess.run([exe, str(MINUTES * 60), str(MATRIX)])
            time.sleep(2)
        finally:
            sampler.terminate()
            try:
                sampler.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sampler.kill()


def analyse(csv_path):
    """Plateau statistics from a 1 Hz log.

    Idle samples are excluded: a clock drop with no kernel running says nothing
    about what the card sustains under load.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.reader(fh):
            try:
                rows.append((datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f"),
                             int(r[1]), int(r[2]), float(r[3]), float(r[4]),
                             int(r[5]), int(r[6].strip(), 16)))
            except (ValueError, IndexError):
                continue                      # header, or an nvidia-smi diagnostic
    busy = [r for r in rows if r[5] > 0 and not r[6] & BIT_GPU_IDLE]
    if not busy:
        sys.exit("✗ no busy samples — did the load run?")
    plateau = [r for r in busy
               if (r[0] - busy[0][0]).total_seconds() >= WARMUP_S] or busy
    sm = [r[1] for r in plateau]
    return {
        "n": len(plateau),
        "sm_min": min(sm), "sm_max": max(sm), "sm_mode": statistics.mode(sm),
        "mem": statistics.mode([r[2] for r in plateau]),
        "temp": max(r[3] for r in plateau),
        "power": max(r[4] for r in plateau),
        "throttles": sorted({n for r in plateau for b, n in THROTTLES.items()
                             if r[6] & b}),
    }


def recommend(res):
    """The plateau clock if it never varied and never throttled, otherwise
    HEADROOM_PCT below the lowest sample, rounded down to a clock the card
    offers."""
    if res["sm_min"] == res["sm_max"] and not res["throttles"]:
        return res["sm_min"], "flat plateau, no throttling — taken as-is"
    offered = sorted({int(v) for v in smi("supported-clocks=gr") if v.isdigit()},
                     reverse=True)
    want = res["sm_min"] * (1 - HEADROOM_PCT / 100)
    sm = next((v for v in offered if v <= want), int(want))
    return sm, (f"plateau varied {res['sm_min']}-{res['sm_max']}MHz"
                + (f", throttled by {', '.join(res['throttles'])}"
                   if res["throttles"] else "")
                + f" — {HEADROOM_PCT}% headroom applied")


def main():
    key = dataset_key()
    gpu = key.split("_", 1)[1] if "_" in key else key
    csv_path = os.path.join(REPO, "data", "clocks", f"calibration_{key}.csv")

    print(f"{(smi('gpu=name') or ['?'])[0]} — {MINUTES} min SGEMM n={MATRIX}")
    run_load(build(), csv_path)
    res = analyse(csv_path)
    sm, why = recommend(res)

    print(f"\n  {res['n']} plateau samples: SM {res['sm_min']}-{res['sm_max']}MHz "
          f"(mode {res['sm_mode']}), mem {res['mem']}MHz, "
          f"{res['temp']:.0f}C, {res['power']:.0f}W")
    print(f"  throttling: {', '.join(res['throttles']) or 'none'}")
    print(f"  log: {csv_path}\n")
    print(f"  {why}")
    print(f"\nAdd to runner_scripts/gpu_clocks.conf:\n\n    {gpu} {sm} {res['mem']}\n")


if __name__ == "__main__":
    main()
