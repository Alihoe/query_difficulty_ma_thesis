from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class Console:
    verbose: bool = True

    @staticmethod
    def format_duration(seconds: float) -> str:
        total_seconds = max(float(seconds), 0.0)
        if total_seconds < 1.0:
            return f"{total_seconds * 1000.0:.0f}ms"
        minutes, seconds_remainder = divmod(total_seconds, 60.0)
        hours, minutes_remainder = divmod(int(minutes), 60)
        if hours > 0:
            return f"{hours}h {minutes_remainder:02d}m {seconds_remainder:04.1f}s"
        if minutes_remainder > 0:
            return f"{minutes_remainder}m {seconds_remainder:04.1f}s"
        return f"{seconds_remainder:.2f}s"

    def log(self, message: str) -> None:
        if not self.verbose:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{stamp}] {message}", flush=True)

    def section(self, title: str) -> None:
        self.log(f"{'=' * 8} {title} {'=' * 8}")

    def log_timing(self, label: str, seconds: float, *, details: str | None = None) -> None:
        message = f"{label} in {self.format_duration(seconds)}"
        if details:
            message = f"{message} ({details})"
        self.log(message)

    def progress_kwargs(self, total: int) -> dict[str, Any]:
        total = max(int(total), 1)
        return {
            "disable": not self.verbose,
            "miniters": max(1, math.ceil(total / 100)),
        }
