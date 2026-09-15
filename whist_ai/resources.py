"""Resource guards for bounded training jobs."""

from __future__ import annotations

import time

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
