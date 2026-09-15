"""Resource guards for bounded training jobs."""

from __future__ import annotations

import time
import subprocess

import psutil


def wait_for_memory(max_percent: float, resume_percent: float, poll_seconds: float = 5.0) -> None:
    """Pause until system memory is below the configured resume threshold."""
    if not 0 < resume_percent <= max_percent <= 100:
        raise ValueError("memory thresholds must satisfy 0 < resume <= max <= 100")
    reported = False
    while True:
        used = psutil.virtual_memory().percent
        if used <= max_percent:
            return
        if not reported:
            print(
                f"RAM guard paused training at {used:.1f}% "
                f"(resume at <= {resume_percent:.1f}%)",
                flush=True,
            )
            reported = True
        while psutil.virtual_memory().percent > resume_percent:
            time.sleep(poll_seconds)
        print("RAM guard resumed training", flush=True)
        return


def _gpu_status() -> tuple[float, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU status")
    temperatures = []
    memory_ratios = []
    for row in rows:
        temperature, used, total = (float(value.strip()) for value in row.split(","))
        temperatures.append(temperature)
        memory_ratios.append(100.0 * used / total)
    return max(temperatures), max(memory_ratios)


def wait_for_gpu(
    max_temperature: float,
    resume_temperature: float,
    max_memory_percent: float,
    poll_seconds: float = 5.0,
) -> None:
    """Pause until GPU temperature and VRAM usage return to safe levels."""
    if not 0 < resume_temperature <= max_temperature:
        raise ValueError("GPU temperature thresholds must satisfy 0 < resume <= max")
    if not 0 < max_memory_percent <= 100:
        raise ValueError("GPU memory limit must satisfy 0 < limit <= 100")
    reported = False
    while True:
        temperature, memory_percent = _gpu_status()
        over_temperature = temperature > max_temperature
        over_memory = memory_percent > max_memory_percent
        if not over_temperature and not over_memory:
            return
        if not reported:
            reasons = []
            if over_temperature:
                reasons.append(f"{temperature:.1f}C")
            if over_memory:
                reasons.append(f"{memory_percent:.1f}% VRAM")
            print(f"GPU guard paused training ({', '.join(reasons)})", flush=True)
            reported = True
        while True:
            time.sleep(poll_seconds)
            temperature, memory_percent = _gpu_status()
            if temperature <= resume_temperature and memory_percent <= max_memory_percent:
                print("GPU guard resumed training", flush=True)
                return
