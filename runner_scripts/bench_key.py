"""The dataset key "<os>_<gpu>" of this machine: the nvidia-smi GPU name tokenised on non-alphanumerics, "NVIDIA"/"GeForce" dropped, joined with '-' (RTX-2060-SUPER); `python bench_key.py` prints it."""

import platform
import re
import subprocess


def _gpu_name_raw():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                line = line.strip()
                if line:
                    return line
    except Exception:
        pass
    return ""


def _sanitize_gpu(raw):
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", raw)
              if t and t not in ("NVIDIA", "GeForce")]
    return "-".join(tokens) if tokens else "unknown-gpu"


def _os_key():
    return {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}.get(
        platform.system(), "unknown")


def dataset_key():
    """Return "<os>_<gpu>" for this machine."""
    return "{0}_{1}".format(_os_key(), _sanitize_gpu(_gpu_name_raw()))


if __name__ == "__main__":
    print(dataset_key())
