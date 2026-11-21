"""Short-horizon planar ball prediction from received ground-truth samples."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from statistics import median

from ..soccer_framework import BallState


@dataclass(frozen=True)
class BallMotion:
    speed: float
    direction_x: float
    direction_y: float
    deceleration: float

    def distance_after(self, seconds: float) -> float:
        if self.deceleration > 0.0:
            seconds = min(seconds, self.speed / self.deceleration)
        return (self.speed - 0.5 * self.deceleration * seconds) * seconds


class BallMotionTracker:
    _MAX_SAMPLE_GAP = 0.24
    _MIN_SAMPLE_GAP = 0.018
    _MAX_SPEED = 12.0

    def __init__(self) -> None:
        self._last: BallState | None = None
        self._intervals: deque[tuple[float, float, float]] = deque(maxlen=5)
        self.motion: BallMotion | None = None

    def reset(self) -> None:
        self._last = None
        self._intervals.clear()
        self.motion = None

    def update(self, ball: BallState | None, now: float | None) -> None:
        if (
            ball is None or now is None or ball.last_seen_at <= 0
            or now - ball.last_seen_at > self._MAX_SAMPLE_GAP
            or not all(math.isfinite(value) for value in (ball.x, ball.y, ball.last_seen_at))
        ):
            self.reset()
            return

        last = self._last
        if last is None:
            self._last = ball
            return
        elapsed = ball.last_seen_at - last.last_seen_at
        if elapsed == 0 or 0 < elapsed < self._MIN_SAMPLE_GAP:
            return
        self._last = ball
        if elapsed < 0 or elapsed > self._MAX_SAMPLE_GAP:
            self._intervals.clear()
            self.motion = None
            return

        vx = (ball.x - last.x) / elapsed
        vy = (ball.y - last.y) / elapsed
        speed = math.hypot(vx, vy)
        if speed > self._MAX_SPEED or speed < 0.20:
            self._intervals.clear()
            self.motion = None
            return
        if self._intervals:
            _, old_vx, old_vy = self._intervals[-1]
            old_speed = math.hypot(old_vx, old_vy)
            alignment = (vx * old_vx + vy * old_vy) / (speed * old_speed)
            if alignment < 0.83 or abs(speed - old_speed) > max(0.9, old_speed * 0.55):
                self._intervals.clear()
        self._intervals.append(((ball.last_seen_at + last.last_seen_at) / 2, vx, vy))
        self.motion = self._estimate(ball.last_seen_at)

    def _estimate(self, sample_time: float) -> BallMotion | None:
        if len(self._intervals) < 2:
            return None
        samples = list(self._intervals)
        last_three = samples[-3:]
        vx = median(sample[1] for sample in last_three)
        vy = median(sample[2] for sample in last_three)
        speed = math.hypot(vx, vy)
        if speed < 0.35:
            return None
        ux, uy = vx / speed, vy / speed
        decelerations = [
            ((earlier[1] - later[1]) * ux + (earlier[2] - later[2]) * uy)
            / (later[0] - earlier[0])
            for earlier, later in zip(samples, samples[1:])
        ]
        deceleration = 0.0
        if len(decelerations) >= 3:
            estimated = median(decelerations)
            if 0.15 <= estimated <= 4.0 and sum(value > 0.05 for value in decelerations) >= 3:
                deceleration = estimated
                latest_at, latest_vx, latest_vy = samples[-1]
                speed = max(
                    0.0,
                    latest_vx * ux + latest_vy * uy - deceleration * (sample_time - latest_at),
                )
        return BallMotion(speed, ux, uy, deceleration)
