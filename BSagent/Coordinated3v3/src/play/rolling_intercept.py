"""Short-horizon rollouts for receiving a rolling ball with the K1 soccer gait."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..soccer_framework import BallState, MoveIntent, Pose2D, RobotCommand, RobotState
from ..tactics.geometry import clamp, normalize_angle
from ..tactics.navigation import Obstacle
from .ball_motion import BallMotion

_STEP = 0.1
_CONTROL_PERIOD = 0.09
_STEPS = 26
_STANCE_OFFSET = 0.20
_FOOT_TOLERANCE = 0.21
_HEADING_TOLERANCE = 0.45
_SETTLED_SPEED = 0.18
_SETTLED_YAW = 0.24

# Command -> estimated steady gait response. The high-speed turn is deliberately
# excluded after the K1 probe measured similar ground speed and more stop drift.
# Submaximal commands are interpolation estimates pending calibration.
_PRIMITIVES = (
    (0.36, 0.0, 0.38, 0.0),
    (0.80, 0.0, 0.90, 0.0),
    (0.36, 0.40, 0.38, 0.52),
    (0.36, -0.40, 0.38, -0.52),
    (0.80, 0.50, 0.80, 0.67),
    (0.80, -0.50, 0.80, -0.67),
    (0.0, 0.40, 0.0, 0.55),
    (0.0, -0.40, 0.0, -0.55),
)


@dataclass(frozen=True)
class _State:
    x: float
    y: float
    heading: float
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0

    def step(self, vx: float, yaw: float) -> _State:
        velocity = self.vx + clamp(vx - self.vx, -_STEP * 0.65, _STEP * 1.5)
        lateral = self.vy + clamp(-self.vy, -_STEP * 0.65, _STEP * 0.65)
        rotation = self.yaw + clamp(yaw - self.yaw, -_STEP * 1.25, _STEP * 2.2)
        # Rotate the body velocity at the midpoint heading of this step.
        delta = (self.yaw + rotation) * _STEP / 2
        angle = self.heading + delta / 2
        forward = (self.vx + velocity) * _STEP / 2
        side = (self.vy + lateral) * _STEP / 2
        return _State(
            self.x + forward * math.cos(angle) - side * math.sin(angle),
            self.y + forward * math.sin(angle) + side * math.cos(angle),
            normalize_angle(self.heading + delta), velocity, lateral, rotation,
        )


@dataclass(frozen=True)
class InterceptPlan:
    command: RobotCommand
    seconds: float
    ball_x: float
    ball_y: float


class RollingIntercept:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sample: tuple[float, _State] | None = None
        self._next_plan_at = 0.0
        self._previous: InterceptPlan | None = None

    def _observe(self, robot: RobotState) -> _State:
        pose = robot.pose
        assert pose is not None
        at = robot.last_seen_at
        prior = self._sample
        if prior is not None and at == prior[0]:
            return prior[1]
        vx = vy = yaw = 0.0
        if prior is not None and 0.015 <= at - prior[0] <= 0.23:
            elapsed = at - prior[0]
            previous = prior[1]
            dx, dy = pose.x - previous.x, pose.y - previous.y
            turn = normalize_angle(pose.theta - previous.heading)
            angle = previous.heading + turn / 2
            forward = (dx * math.cos(angle) + dy * math.sin(angle)) / elapsed
            lateral = (-dx * math.sin(angle) + dy * math.cos(angle)) / elapsed
            vx = 0.5 * previous.vx + 0.5 * clamp(forward, -1.3, 1.3)
            vy = 0.5 * previous.vy + 0.5 * clamp(lateral, -0.8, 0.8)
            yaw = 0.5 * previous.yaw + 0.5 * clamp(turn / elapsed, -2.0, 2.0)
        state = _State(pose.x, pose.y, pose.theta, vx, vy, yaw)
        self._sample = at, state
        return state

    def plan(
        self, now: float, robot: RobotState, ball: BallState, motion: BallMotion | None,
        obstacles: tuple[Obstacle, ...], bounds: tuple[float, float],
    ) -> InterceptPlan | None:
        if robot.pose is None or not robot.is_recent(now, max_age_sec=0.20):
            self.reset()
            return None
        robot_state = self._observe(robot)
        if motion is None or motion.speed < 0.45:
            self._previous = None
            return None
        delta_x, delta_y = robot_state.x - ball.x, robot_state.y - ball.y
        along = delta_x * motion.direction_x + delta_y * motion.direction_y
        across = abs(delta_x * motion.direction_y - delta_y * motion.direction_x)
        if along < -0.20 or across > 1.5:
            self._previous = None
            return None
        if now < self._next_plan_at:
            return self._previous
        self._next_plan_at = now + _CONTROL_PERIOD
        self._previous = _select(
            robot_state, ball, motion, max(0.0, now - ball.last_seen_at), obstacles, bounds,
        )
        return self._previous


def _select(
    initial: _State, ball: BallState, motion: BallMotion, sample_age: float,
    obstacles: tuple[Obstacle, ...], bounds: tuple[float, float],
) -> InterceptPlan | None:
    ux, uy = motion.direction_x, motion.direction_y
    heading = math.atan2(-uy, -ux)
    targets = []
    for index in range(1, _STEPS + 1):
        t = index * _STEP + sample_age
        if motion.speed - t * motion.deceleration < 0.20:
            break
        distance = motion.distance_after(t) + _STANCE_OFFSET
        x, y = ball.x + ux * distance, ball.y + uy * distance
        if abs(x) > bounds[0] or abs(y) > bounds[1]:
            break
        targets.append((x, y))

    best = None
    for command_vx, command_yaw, speed, yaw in ((0.0, 0.0, 0.0, 0.0), *_PRIMITIVES):
        durations = (0,) if not (speed or yaw) else (1, 2, 3, 4, 6, 8, 10, 12, 15)
        for duration in durations:
            state = initial
            for index, (x, y) in enumerate(targets, 1):
                driving = index <= duration
                state = state.step(speed if driving else 0.0, yaw if driving else 0.0)
                if abs(state.x) > bounds[0] or abs(state.y) > bounds[1] or any(
                    (state.x - p.x) ** 2 + (state.y - p.y) ** 2 < (p.radius + 0.22) ** 2
                    for p in obstacles
                ):
                    break
                if driving or math.hypot(state.vx, state.vy) > _SETTLED_SPEED or abs(state.yaw) > _SETTLED_YAW:
                    continue
                position_error = math.hypot(state.x - x, state.y - y)
                if position_error > _FOOT_TOLERANCE:
                    continue
                heading_error = abs(normalize_angle(state.heading - heading))
                if heading_error > _HEADING_TOLERANCE:
                    continue
                # Prioritize readiness in a common time frame, then precision and shorter actuation.
                score = index * _STEP + position_error * 1.0 + heading_error * .2 + duration * .002
                if best is None or score < best[0]:
                    best = score, command_vx, command_yaw, index * _STEP, x, y
                break
    if best is None:
        return None
    _, vx, yaw, seconds, x, y = best
    if vx == yaw == 0.0:
        command = RobotCommand.stop("intercept: hold")
    else:
        action = "arc" if yaw and vx else "turn" if yaw else "straight"
        command = RobotCommand(MoveIntent(vx=vx, vyaw=yaw), f"intercept: {action}")
    return InterceptPlan(command, seconds, x - ux * _STANCE_OFFSET, y - uy * _STANCE_OFFSET)
