"""Best-effort description of the machine, stored with every eval run.

Speed numbers are meaningless without the hardware they came from, so the
README benchmark table is built from this, not typed by hand.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Any


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def cpu_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, OSError, AttributeError):  # Windows has no sysconf
        return None


def gpu_name() -> str | None:
    if shutil.which("nvidia-smi"):
        return _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "Apple Silicon (unified memory, Metal)"
    return None


def describe_hardware() -> dict[str, Any]:
    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu": cpu_name(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": ram_gb(),
        "gpu": gpu_name(),
        "python": platform.python_version(),
    }
