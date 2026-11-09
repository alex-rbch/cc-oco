from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .proxy_delays import ParetoProxyGenerator, UniformProxyGenerator


@dataclass
class TrackedRound:
    round_index: int
    feedback_time: int
    proxy_expiry_time: Optional[int]


__all__ = ["TrackedRound", "ParetoProxyGenerator", "UniformProxyGenerator"]
