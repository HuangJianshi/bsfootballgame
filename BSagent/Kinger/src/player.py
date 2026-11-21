"""Player control handle for one robot, designed for direct user editing.

Platform primitives such as ``set_velocity``, ``kick``, ``release_kick``,
``request_mode``, and ``get_up`` delegate to the injected framework
``_backend``. State properties such as ``pose``, ``mode``, ``is_fallen``, and
``penalty`` read from ``self.context`` or the backend.

Movement behaviors such as ``walk_to``, ``face_to``, and ``ensure_ready`` are
also Player methods because they command this player and may need cross-frame
state for hysteresis or avoidance. Pure coordinate calculations such as
``dist``, ``angle_to``, and goal coordinates belong in utils/geom.

Instances live for the entire match, while the framework replaces
``self.context`` every frame. Add custom skills directly to this class.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from .framework.types import Context, Penalty, Pose2D
from .framework import debugdraw

from .param import (
    ANGULAR_GAIN,
    ARRIVE_DIST,
    BACKFIELD_UNSAFE_PASS_X_M,
    BACK_PASS_MAX_PRESSURE,
    CHASE_BEHIND_M,
    CLEARANCE_BALL_X_M,
    CLEARANCE_COOLDOWN_S,
    CLEARANCE_KEY,
    CLEARANCE_MIN_GAIN_X_M,
    CLEARANCE_MIN_PRESSERS,
    CLEARANCE_POWER,
    CLEARANCE_PRESSURE_M,
    DEFEND_APPROACH_CENTER_M,
    DEFEND_APPROACH_OFFSET_M,
    DEFEND_COMMIT_DIST_M,
    GLOBAL_PATH_LOOKAHEAD_M,
    GOAL_TARGET_DEPTH_M,
    GUARD_FACE_BALL,
    KICKOFF_FRONT_MARGIN,
    KICKOFF_LATERAL_TOL,
    KICKOFF_STAGE_M,
    KICK_AIM_MAX_RAD,
    KICK_ENTER_M,
    KICK_EXIT_M,
    KICK_FRONT_CONE_RAD,
    KICK_FRONT_CONE_SETTLE_RAD,
    KICK_PIVOT_MIN_DIST_M,
    KICK_POWER_BACKFIELD,
    KICK_POWER_DEFAULT,
    KICK_POWER_MAX,
    KICK_POWER_MIN,
    KICK_TARGET_MARK_SIZE_M,
    KICK_TURN_MAX_RAD,
    LATERAL_MIN_GAIN_Y_M,
    LINEAR_GAIN,
    LONG_BALL_BALL_X_M,
    LONG_BALL_CLEARANCE_M,
    LONG_BALL_COOLDOWN_S,
    LONG_BALL_KEY,
    LONG_BALL_MIN_GAIN_X_M,
    LONG_BALL_POWER,
    MAX_ANGULAR,
    MAX_LINEAR,
    OMNI_DIST,
    OPEN_GOAL_CLEAR_M,
    PASS_BACK_KEY,
    PASS_COOLDOWN_S,
    PASS_FORWARD_KEY,
    PASS_LANE_CLEARANCE_M,
    PASS_LATERAL_KEY,
    PASS_MIN_GAIN_X_M,
    PASS_POWER_BASE,
    PASS_POWER_PER_M,
    PASS_RECEIVER_SPACE_M,
    PLAN_CLEARANCE,
    PLAN_LOOKAHEAD,
    PLAN_MAX_OFFSET,
    PLAN_STEP,
    SCORE_TRAIL_SHOT_RANGE_BONUS_M,
    SHOT_COOLDOWN_S,
    SHOT_KEY,
    SHOT_LANE_CLEARANCE_M,
    SHOT_LANE_CLOSE_CLEARANCE_M,
    SHOT_POWER,
    SHOT_RANGE_CLOSE_M,
    SHOT_RANGE_M,
    SHOT_RANGE_OPEN_GOAL_M,
    SHOT_TARGET_SPREAD_M,
    SUPPORT_DIST_M,
    TURN_THRESHOLD,
    USE_GLOBAL_PATH_PLANNER,
)
from .utils.geom import (
    angle_to,
    clamp,
    clamp_inside_field,
    dist,
    normalize_angle,
    opponent_goal,
    own_goal,
    own_goal_area_center,
)
from .utils.obstacles import collect_obstacles
from .utils.path_planner import plan_global_path

if TYPE_CHECKING:
    from .framework.config import SoccerConfig
    from .framework.robot_backend import RobotBackend


__all__ = ["Player"]


_log = logging.getLogger(__name__)

def _heading_clearance(
    px: float, py: float, heading: float, obstacles: list,
) -> float:
    """Return clearance along a lookahead ray for local obstacle avoidance.

    Only obstacles ahead of (px, py), with projection t > 0, block this heading.
    Ignoring obstacles behind or to the rear prevents a nearby rear obstacle
    from reducing clearance in every direction. Return infinity when clear;
    larger values mean more space and negative values indicate a collision.
    """
    ux, uy = math.cos(heading), math.sin(heading)
    min_clear = math.inf
    for obs in obstacles:
        t = (obs.x - px) * ux + (obs.y - py) * uy
        if t <= 0.0:
            continue                      # Rear obstacles do not block this heading.
        if t > PLAN_LOOKAHEAD:
            t = PLAN_LOOKAHEAD
        nx, ny = px + ux * t, py + uy * t
        clear = math.hypot(obs.x - nx, obs.y - ny) - obs.radius
        if clear < min_clear:
            min_clear = clear
    return min_clear


def _behind_ball(
    ball_x: float, ball_y: float, aim: tuple[float, float], offset: float,
) -> tuple[float, float]:
    """Return a position ``offset`` behind the ball relative to ``aim``.

    This is used as the chase target so the ball lies between the robot and
    ``aim``, naturally aligning the robot on arrival. If ``aim`` coincides with
    the ball and the direction is undefined, return the ball position.
    """
    dx, dy = aim[0] - ball_x, aim[1] - ball_y
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return (ball_x, ball_y)
    ux, uy = dx / d, dy / d              # Unit vector from the ball to aim.
    return (ball_x - ux * offset, ball_y - uy * offset)   # Move away from aim.


class Player:
    """Handle for one player. Add new skills here.
    """

    def __init__(
        self,
        player_id: int,
        config: "SoccerConfig",
        _backend: "RobotBackend | None",
    ) -> None:
        self.id: int = player_id
        self.config: "SoccerConfig" = config
        self._backend = _backend           # Robot control wrapper; rarely changed
        self.context: Context | None = None

        # Current high-level action for visualization and debugging. Strategy
        # dispatch updates it each frame; actions such as guard refine substates
        # internally. The visualization pass in main.py uses it as a label.
        self.action: str = "init"

        # SDK cache fields updated automatically by the framework backend.
        self._mode: str | None = None
        self._fall_down_state: str | None = None

        # Cross-frame detour-side memory; None means no active detour.
        self._avoid_side: float | None = None

        # Cross-frame kick hysteresis state.
        self._kicking: bool = False

        # Aggressive actions (shot / clearance / long ball) keep a small
        # per-action cooldown so the decision does not flip every frame.
        self._action_cooldowns: dict[str, float] = {}

        # Indirect free kicks must not shoot straight at the goal; the strategy
        # sets this each frame before dispatching the attacker.
        self.allow_direct_shot: bool = True

        # Cross-frame block/guard hysteresis state.
        self._block_pressing: bool = False
        self._guard_threatened: bool = False
        # Cross-frame state for driving the ball over the goal line.
        self._goal_line_push: bool = False
        # Cross-frame state for temporarily attacking when support gets stuck.
        self._support_last_pos: tuple[float, float] | None = None
        self._support_stationary_since: float | None = None
        self._support_last_update_at: float | None = None

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def is_kicking(self) -> bool:
        """Return whether the player is currently kicking."""
        return self._kicking

    @property
    def pose(self) -> Pose2D | None:
        ctx = self.context
        if ctx is None:
            return None
        robot = ctx.teammates.get(self.id)
        return None if robot is None else robot.pose

    @property
    def mode(self) -> str | None:
        if self._backend is not None:
            return self._backend.mode
        return self._mode

    @property
    def is_fallen(self) -> bool:
        return self.fall_down_state not in (None, "normal")

    @property
    def fall_down_state(self) -> str | None:
        if self._backend is not None:
            return getattr(self._backend, "fall_down_state", None)
        return self._fall_down_state

    @property
    def penalty(self) -> Penalty:
        ctx = self.context
        game = ctx.game if ctx is not None else None
        if game is None:
            return Penalty.NONE
        state = game.get_player_state(self.config.team_id, self.id)
        return Penalty.NONE if state is None else state.penalty

    @property
    def is_penalized(self) -> bool:
        return self.penalty != Penalty.NONE

    # ------------------------------------------------------------------
    # Chassis control
    # ------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        if self._backend is None:
            _log.debug(
                "player %d set_velocity vx=%.3f vy=%.3f vyaw=%.3f (no backend)",
                self.id, vx, vy, vyaw,
            )
            return
        self._backend.set_velocity(vx, vy, vyaw)

    def stop(self) -> None:
        self.release_kick()
        self.set_velocity(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Kicking
    # ------------------------------------------------------------------

    def kick(
        self,
        kick_direction: float | None = None,
        power: float = KICK_POWER_DEFAULT,
    ) -> None:
        pose = self.pose
        if pose is None:
            _log.warning("player %d kick skipped: pose unknown", self.id)
            return

        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ball is None:
            _log.warning("player %d kick skipped: ball unknown", self.id)
            return

        if kick_direction is None:
            kick_plan = self.plan_kick()
            if kick_plan is None:
                _log.warning("player %d kick skipped: kick plan unavailable", self.id)
                return
            kick_direction, power = kick_plan

        if self._backend is None:
            _log.debug(
                "player %d kick ball=(%.3f, %.3f) dir=%.3f (no backend)",
                self.id, ball.x, ball.y, kick_direction,
            )
            return
        # Transform field coordinates to body coordinates using the current pose.
        dx = ball.x - pose.x
        dy = ball.y - pose.y
        cos_t = math.cos(pose.theta)
        sin_t = math.sin(pose.theta)
        ball_x_body = dx * cos_t + dy * sin_t
        ball_y_body = -dx * sin_t + dy * cos_t
        kick_direction = normalize_angle(kick_direction)
        direction_body = normalize_angle(kick_direction - pose.theta)
        power_clamped = max(KICK_POWER_MIN, min(KICK_POWER_MAX, power))
        if self._kick_threatens_own_goal(kick_direction):
            _log.warning(
                "player %d blocked kick toward own goal: dir=%.3f",
                self.id, kick_direction,
            )
            self.release_kick()
            return
        self._kicking = True
        self._backend.kick(direction_body, power_clamped, ball_x_body, ball_y_body)

    def plan_kick(self) -> tuple[float, float] | None:
        """Calculate kick direction and power.

        Aim from the current ball position toward the opponent's goal center.
        Return ``(kick_direction, kick_power)``, or None when the ball or context
        is unavailable.
        """
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return None

        kick_target = opponent_goal(ctx)
        kick_direction = angle_to(ball.x, ball.y, *kick_target)
        kick_target = self._goal_target_for_direction(kick_direction)
        kick_power = (
            KICK_POWER_BACKFIELD if self._in_backfield()
            else KICK_POWER_DEFAULT
        )

        self._draw_kick_target(kick_target)
        return kick_direction, kick_power

    def _goal_target_for_direction(
        self, kick_direction: float,
    ) -> tuple[float, float]:
        """Project the shot direction onto the opponent's goal line for display."""
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return (0.0, 0.0)

        dx = math.cos(kick_direction)
        if dx <= 1e-6:
            return opponent_goal(ctx)
        goal_x = ctx.field.length / 2.0
        t = max(0.0, (goal_x - ball.x) / dx)
        return (goal_x, ball.y + math.sin(kick_direction) * t)

    def _in_backfield(self) -> bool:
        """Return whether the ball is in our half, where kicks use more power."""
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return False
        # own_penalty_edge_x = -ctx.field.length / 2.0 + ctx.field.penalty_area_length
        # return ball.x < own_penalty_edge_x
        return ball.x < 0

    def _draw_kick_target(self, target: tuple[float, float]) -> None:
        """Mark the kick target selected by plan_kick with an X."""
        from .framework import debugdraw

        x, y = target
        s = KICK_TARGET_MARK_SIZE_M
        debugdraw.line(
            [(x - s, y - s), (x + s, y + s)],
            rgb=(1.0, 0.0, 1.0), ns="kick_target",
        )
        debugdraw.line(
            [(x - s, y + s), (x + s, y - s)],
            rgb=(1.0, 0.0, 1.0), ns="kick_target",
        )

    def release_kick(self) -> None:
        self._kicking = False  # Clear kick hysteresis and cube visualization.
        if self._backend is None:
            _log.debug("player %d release_kick (no backend)", self.id)
            return
        self._backend.release_kick()

    def kick_can_score(self, kick_direction: float) -> bool:
        """Return whether a straight kick in ``kick_direction`` can score.
        """
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return False

        goal_x = ctx.field.length / 2.0
        dx = math.cos(kick_direction)
        dy = math.sin(kick_direction)
        if dx <= 1e-6:
            return False

        half_goal = ctx.field.goal_width / 2.0
        if half_goal <= 0.0:
            return False
        if ball.x >= goal_x:
            y_at_goal = ball.y
        else:
            t = (goal_x - ball.x) / dx
            if t < 0.0:
                return False
            y_at_goal = ball.y + dy * t
        return -half_goal <= y_at_goal <= half_goal

    # ------------------------------------------------------------------
    # Slow operations, invoked asynchronously to avoid blocking
    # ------------------------------------------------------------------

    @property
    def is_getting_up(self) -> bool:
        """Return whether recovery is queued or still executing in the SDK."""
        return self._backend is not None and self._backend.is_getting_up

    def request_mode(self, mode: str) -> None:
        if self._backend is None:
            _log.debug("player %d request_mode -> %s (no backend)", self.id, mode)
            return
        self._backend.request_mode(mode)

    def get_up(self) -> None:
        if self._backend is None:
            _log.debug("player %d get_up (no backend)", self.id)
            return
        self._backend.get_up()

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def ensure_ready(self) -> bool:
        """Recover from falls and switch to walk mode asynchronously.

        Return whether the player is ready to act during this frame. Mode
        changes must wait for the get-up task, not just a normal fall snapshot.
        """
        if self.is_getting_up:
            return False
        if self.is_fallen:
            self.get_up()
            return False
        if self.mode != "walk":
            self.request_mode("walk")
            return False
        return True

    def face_to(self, target_theta: float) -> None:
        """Turn in place to the target heading."""
        pose = self.pose
        if pose is None:
            self.stop()
            return
        err = normalize_angle(target_theta - pose.theta)
        self.set_velocity(0.0, 0.0, self._angular(err))

    def walk_to(
        self,
        target: tuple[float, float],
        *,
        face: float | None = None,
        avoid_ball: bool = False,
        avoid_robots: bool = False,
        arrive_dist: float = ARRIVE_DIST,
    ) -> bool:
        """Walk toward a target point and return whether it has been reached.

        With ``avoid_ball`` or ``avoid_robots`` enabled, the simplified VFH
        local planner models the ball, robots, and goals as circular obstacles.
        It scans candidate headings and selects the one closest to the target
        that remains collision-free over ``PLAN_LOOKAHEAD``. This handles
        multiple obstacles, concave structures, and symmetric conflicts more
        reliably than single-obstacle detours or potential-field repulsion.

        Nearby targets use omnidirectional walking. When ``face`` is supplied,
        the body keeps looking in that direction while translating; otherwise
        distant targets use a turn-walk-turn sequence.
        """
        self.release_kick() # Walking is overridden while kicking, so release first.
        pose = self.pose
        if pose is None:
            self.stop()
            return False

        tx, ty = target
        dx = tx - pose.x
        dy = ty - pose.y
        distance = math.hypot(dx, dy)

        if distance < arrive_dist:
            # At the target, turn to the requested heading if needed.
            if face is not None:
                err = normalize_angle(face - pose.theta)
                if abs(err) > 0.1:
                    self.set_velocity(0.0, 0.0, self._angular(err))
                else:
                    self.stop()
            else:
                self.stop()
            return True

        # Prefer global A* planning and fall back to the legacy local planner.
        goal_dir = math.atan2(dy, dx)
        planned_path: list[tuple[float, float]] | None = None
        waypoint: tuple[float, float] | None = None
        ctx = self.context
        if (avoid_ball or avoid_robots) and ctx is not None:
            obstacles = collect_obstacles(
                ctx, self.id,
                ball=avoid_ball, robots=avoid_robots,
                goals=(avoid_ball or avoid_robots),
            )
            if USE_GLOBAL_PATH_PLANNER:
                planned_path = plan_global_path(
                    ctx,
                    (pose.x, pose.y),
                    (tx, ty),
                    obstacles,
                )
                if planned_path is not None:
                    waypoint = self._path_waypoint(pose, planned_path)
                    heading = angle_to(pose.x, pose.y, waypoint[0], waypoint[1])
                else:
                    heading = self._plan_heading(pose, goal_dir, obstacles)
            else:
                heading = self._plan_heading(pose, goal_dir, obstacles)
        else:
            heading = goal_dir

        # Visualize the target (green), direct line (gray), planned heading
        # (yellow), and forward lookahead probe (cyan).
        from .framework import debugdraw
        debugdraw.point(tx, ty, rgb=(0.0, 1.0, 0.0), scale=0.15, ns="target")
        debugdraw.line([(pose.x, pose.y), (tx, ty)], rgb=(0.4, 0.4, 0.4), ns="to_target")
        if planned_path is not None and len(planned_path) >= 2:
            debugdraw.line(planned_path, rgb=(0.2, 0.8, 1.0), ns="global_path")
        if waypoint is not None:
            debugdraw.point(
                waypoint[0], waypoint[1],
                rgb=(0.2, 0.8, 1.0), scale=0.12, ns="global_waypoint",
            )
        debugdraw.arrow(
            pose.x, pose.y,
            pose.x + math.cos(heading) * 0.6, pose.y + math.sin(heading) * 0.6,
            rgb=(1.0, 1.0, 0.0), ns="heading",
        )
        debugdraw.line(
            [(pose.x, pose.y),
             (pose.x + math.cos(heading) * PLAN_LOOKAHEAD,
              pose.y + math.sin(heading) * PLAN_LOOKAHEAD)],
            rgb=(0.0, 0.8, 0.8), ns="lookahead",
        )

        if distance <= OMNI_DIST or face is not None:
            # Nearby: translate along heading while turning toward face.
            wdx, wdy = math.cos(heading) * distance, math.sin(heading) * distance
            cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
            vx = LINEAR_GAIN * (wdx * cos_t + wdy * sin_t)
            vy = LINEAR_GAIN * (-wdx * sin_t + wdy * cos_t)
            speed = math.hypot(vx, vy)
            if speed > MAX_LINEAR:
                vx *= MAX_LINEAR / speed
                vy *= MAX_LINEAR / speed
            vyaw = (
                self._angular(normalize_angle(face - pose.theta))
                if face is not None else 0.0
            )
            self.set_velocity(vx, vy, vyaw)
        else:
            # Distant: turn, walk, and turn along heading.
            angle_err = normalize_angle(heading - pose.theta)
            if abs(angle_err) > TURN_THRESHOLD:
                self.set_velocity(0.0, 0.0, self._angular(angle_err))
            else:
                vx = clamp(
                    LINEAR_GAIN * distance * math.cos(angle_err), 0.0, MAX_LINEAR,
                )
                self.set_velocity(vx, 0.0, self._angular(angle_err))
        return False

    def _path_waypoint(
        self, pose: Pose2D, path: list[tuple[float, float]],
    ) -> tuple[float, float]:
        """Pick a short lookahead waypoint from a planned global path."""
        if not path:
            return (pose.x, pose.y)
        prev = (pose.x, pose.y)
        points = path[1:] if len(path) > 1 else path
        for point in points:
            seg_len = dist(prev[0], prev[1], point[0], point[1])
            if seg_len >= GLOBAL_PATH_LOOKAHEAD_M:
                ratio = GLOBAL_PATH_LOOKAHEAD_M / max(seg_len, 1e-6)
                return (
                    prev[0] + (point[0] - prev[0]) * ratio,
                    prev[1] + (point[1] - prev[1]) * ratio,
                )
            prev = point
        return path[-1]

    def _plan_heading(
        self, pose: Pose2D, goal_dir: float, obstacles: list,
    ) -> float:
        """Choose the clearest candidate heading nearest the target direction.

        Candidates are tested by increasing absolute offset from the target.
        Player ID parity chooses which side is tried first, breaking symmetry
        when two players avoid each other.
        """
        sign_first = 1.0 if self.id % 2 == 0 else -1.0
        best_h = goal_dir
        best_clear = -math.inf

        offsets = [0.0]
        k = 1
        while k * PLAN_STEP <= PLAN_MAX_OFFSET + 1e-9:
            offsets.append(sign_first * k * PLAN_STEP)
            offsets.append(-sign_first * k * PLAN_STEP)
            k += 1

        for off in offsets:
            h = goal_dir + off
            clear = _heading_clearance(pose.x, pose.y, h, obstacles)
            if clear >= PLAN_CLEARANCE:
                return h
            if clear > best_clear:
                best_clear, best_h = clear, h
        return best_h

    # ------------------------------------------------------------------
    # High-level actions called directly by the strategy in main.py
    # ------------------------------------------------------------------


    def block_path_projection(
        self, opponent_id: int,
    ) -> tuple[float, float, float, float] | None:
        """Project this player onto an opponent-to-ball segment.

        Return ``(x, y, perpendicular_distance, segment_parameter_t)``.
        """
        ctx = self.context
        pose = self.pose
        ball = ctx.ball if ctx is not None else None
        opponent = ctx.opponents.get(opponent_id) if ctx is not None else None
        if ctx is None or pose is None or ball is None or opponent is None:
            return None
        opponent_pose = opponent.pose
        if opponent_pose is None:
            return None

        ax, ay = opponent_pose.x, opponent_pose.y
        bx, by = ball.x, ball.y
        vx, vy = bx - ax, by - ay
        length2 = vx * vx + vy * vy
        if length2 < 1e-6:
            return None

        raw_t = ((pose.x - ax) * vx + (pose.y - ay) * vy) / length2
        t = clamp(raw_t, 0.0, 1.0)
        tx = ax + vx * t
        ty = ay + vy * t
        return tx, ty, dist(pose.x, pose.y, tx, ty), raw_t

    @staticmethod
    def _body_bearing(pose: Pose2D, x: float, y: float) -> float:
        """Return a field point's bearing in body coordinates (0 is straight ahead)."""
        dx, dy = x - pose.x, y - pose.y
        cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
        return math.atan2(-dx * sin_t + dy * cos_t, dx * cos_t + dy * sin_t)

    def _pivot_for_kick(
        self, pose: Pose2D, ball_bearing: float, kick_direction: float,
    ) -> float | None:
        """Return the body heading that brings the ball into the kick cone.

        The robot pivots by the smallest rotation that satisfies both limits:
        the ball must sit inside ``KICK_FRONT_CONE_RAD`` minus the settle margin,
        and the shot direction must stay within ``KICK_AIM_MAX_RAD`` so the kick
        stays inside the mechanical range. Return None when no such heading
        exists, which leaves the caller to walk around the ball instead.
        """
        cone = KICK_FRONT_CONE_RAD - KICK_FRONT_CONE_SETTLE_RAD
        aim_offset = normalize_angle(kick_direction - pose.theta)
        # Turning by dyaw moves both bearings by -dyaw in body coordinates.
        lo = max(aim_offset - KICK_AIM_MAX_RAD, ball_bearing - cone)
        hi = min(aim_offset + KICK_AIM_MAX_RAD, ball_bearing + cone)
        if lo > hi:
            return None
        # Prefer the rotation nearest the current heading, then the smallest one.
        dyaw = 0.0 if lo <= 0.0 <= hi else (lo if abs(lo) <= abs(hi) else hi)
        return normalize_angle(pose.theta + dyaw)

    # ------------------------------------------------------------------
    # Aggressive decisions: shot, clearance, long ball
    # ------------------------------------------------------------------

    def _cooldown_ready(self, key: str) -> bool:
        """Return whether an aggressive action is off cooldown."""
        ctx = self.context
        if ctx is None:
            return False
        until = self._action_cooldowns.get(key, 0.0)
        return ctx.now >= until

    def _start_cooldown(self, key: str, seconds: float) -> None:
        ctx = self.context
        if ctx is None:
            return
        self._action_cooldowns[key] = ctx.now + seconds

    def _lane_clearance(
        self, x0: float, y0: float, x1: float, y1: float,
    ) -> float:
        """Return the smallest opponent distance to the segment (x0,y0)-(x1,y1).

        Used as the "safe lane" test before committing to a long ball or shot.
        Opponents behind the start or past the end are ignored, and the value is
        infinity when no opponent stands in the lane.
        """
        ctx = self.context
        if ctx is None:
            return 0.0
        vx, vy = x1 - x0, y1 - y0
        length2 = vx * vx + vy * vy
        if length2 < 1e-6:
            return 0.0
        min_clear = math.inf
        for opponent in ctx.opponents.values():
            pose = opponent.pose
            if pose is None:
                continue
            t = ((pose.x - x0) * vx + (pose.y - y0) * vy) / length2
            if t <= 0.0 or t >= 1.0:
                continue
            px, py = x0 + vx * t, y0 + vy * t
            clear = dist(pose.x, pose.y, px, py)
            if clear < min_clear:
                min_clear = clear
        return min_clear

    def _shot_path_clearance(
        self, target: tuple[float, float],
    ) -> float:
        """Return clearance along the shot lane and around its goal-mouth end."""
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return 0.0
        lane = self._lane_clearance(ball.x, ball.y, *target)
        endpoint = math.inf
        for robot in (*ctx.teammates.values(), *ctx.opponents.values()):
            pose = robot.pose
            if pose is None or robot.player_id == self.id:
                continue
            endpoint = min(
                endpoint,
                dist(pose.x, pose.y, target[0], target[1]),
            )
        return min(lane, endpoint)

    def _kick_threatens_own_goal(self, kick_direction: float) -> bool:
        """Return whether a field-frame kick points into our own goal mouth."""
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return False
        dx = math.cos(kick_direction)
        if dx >= -1e-6:
            return False
        goal_x = -ctx.field.length / 2.0
        t = (goal_x - ball.x) / dx
        if t <= 0.0:
            return False
        y_at_goal = ball.y + math.sin(kick_direction) * t
        guard_half = (
            ctx.field.goal_width / 2.0 + SHOT_LANE_CLOSE_CLEARANCE_M
        )
        return abs(y_at_goal) <= guard_half

    def _count_pressers(self) -> int:
        """Return how many opponents are close enough to count as pressure."""
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return 0
        count = 0
        for opponent in ctx.opponents.values():
            pose = opponent.pose
            if pose is None:
                continue
            if dist(pose.x, pose.y, ball.x, ball.y) <= CLEARANCE_PRESSURE_M:
                count += 1
        return count

    def _team_score_difference(self) -> int:
        """Return our score minus the opponent score."""
        ctx = self.context
        game = ctx.game if ctx is not None else None
        if ctx is None or game is None:
            return 0
        ours = game.get_team_state(self.config.team_id)
        theirs = next(
            (team for team in game.teams if team.team_number != self.config.team_id),
            None,
        )
        if ours is None or theirs is None:
            return 0
        return ours.score - theirs.score

    def _best_shot_target(
        self, goal_x: float, goal_y: float,
    ) -> tuple[float, float]:
        """Pick the clearest point inside the goal mouth.

        Centre, inside-right, and inside-left aim points are compared by lane
        clearance. The inside points stay away from the posts, while the centre
        remains the tie-breaker when multiple lanes are equally open.
        """
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return (goal_x, goal_y)

        spread = min(SHOT_TARGET_SPREAD_M, ctx.field.goal_width * 0.3)
        candidates = (
            (goal_x, goal_y),
            (goal_x, goal_y + spread),
            (goal_x, goal_y - spread),
        )
        best = candidates[0]
        best_score = -math.inf
        for target in candidates:
            lane = self._shot_path_clearance(target)
            score = lane - 0.1 * abs(target[1] - goal_y)
            if score > best_score:
                best_score, best = score, target
        return best

    def _forward_candidates(
        self, ball, wide_only: bool = False,
    ) -> list[tuple[float, float]]:
        """Return upfield target points: goal, goal posts, wings, teammates."""
        ctx = self.context
        if ctx is None:
            return []
        half_l = ctx.field.length / 2.0
        half_w = ctx.field.width / 2.0
        goal_x = half_l + GOAL_TARGET_DEPTH_M
        wing_y = half_w - 0.8
        candidates: list[tuple[float, float]] = [
            (half_l - 1.0, wing_y),
            (half_l - 1.0, -wing_y),
            (ball.x + 3.0, wing_y),
            (ball.x + 3.0, -wing_y),
        ]
        if not wide_only:
            candidates.extend((
                (goal_x, 0.0),
                (goal_x, ctx.field.goal_width * 0.4),
                (goal_x, -ctx.field.goal_width * 0.4),
            ))
        for teammate in ctx.teammates.values():
            pose = teammate.pose
            if pose is None or teammate.player_id == self.id:
                continue
            if ctx.game is not None:
                state = ctx.game.get_player_state(
                    self.config.team_id, teammate.player_id,
                )
                if state is not None and state.penalty != Penalty.NONE:
                    continue
            candidates.append((pose.x, pose.y))
        return candidates

    def _pick_forward_target(
        self,
        ball,
        min_gain_m: float,
        min_lane_m: float,
        wide_only: bool = False,
    ) -> tuple[float, float] | None:
        """Pick the best safe upfield target for a long ball or clearance.

        Candidates must advance the ball by ``min_gain_m`` and keep at least
        ``min_lane_m`` clearance from opponents. Targets that would send the
        ball toward our own goal are rejected, so a clearance can never turn
        into an own-goal risk.
        """
        ctx = self.context
        if ctx is None:
            return None
        half_l = ctx.field.length / 2.0
        half_w = ctx.field.width / 2.0
        best: tuple[float, float] | None = None
        best_score = -math.inf
        for tx, ty in self._forward_candidates(ball, wide_only=wide_only):
            tx = clamp(tx, -half_l + 0.3, half_l)
            ty = clamp(ty, -half_w + 0.3, half_w - 0.3)
            gain = tx - ball.x
            if gain < min_gain_m or tx <= 0.0:
                continue
            lane = self._lane_clearance(ball.x, ball.y, tx, ty)
            if lane < min_lane_m:
                continue
            # Prefer the most advanced target, then the safest lane.
            score = gain + 0.5 * min(lane, 3.0)
            if score > best_score:
                best_score, best = score, (tx, ty)
        return best

    def _space_around(self, x: float, y: float) -> float:
        """Return the nearest opponent distance to a point, for receiver checks."""
        ctx = self.context
        if ctx is None:
            return 0.0
        nearest = math.inf
        for opponent in ctx.opponents.values():
            pose = opponent.pose
            if pose is None:
                continue
            clear = dist(pose.x, pose.y, x, y)
            if clear < nearest:
                nearest = clear
        return nearest

    def _pass_power(self, distance: float) -> float:
        """Return a pass power scaled by distance."""
        return clamp(
            PASS_POWER_BASE + PASS_POWER_PER_M * distance,
            KICK_POWER_MIN, KICK_POWER_MAX,
        )

    def _pick_pass_target(self, mode: str) -> tuple[float, float] | None:
        """Pick a teammate pass target in the requested direction.

        ``mode`` is ``forward`` (advance the ball), ``lateral`` (switch to the
        other wing), or ``back`` (keep possession with a backward/lateral pass).
        A target must have a lane clear of opponents and enough space for the
        receiver to control the ball.
        """
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return None
        best: tuple[float, float] | None = None
        best_score = -math.inf
        for teammate in ctx.teammates.values():
            pose = teammate.pose
            if pose is None or teammate.player_id == self.id:
                continue
            if ctx.game is not None:
                state = ctx.game.get_player_state(
                    self.config.team_id, teammate.player_id,
                )
                if state is not None and state.penalty != Penalty.NONE:
                    continue
            gain_x = pose.x - ball.x
            gain_y = pose.y - ball.y
            if mode == "forward":
                if gain_x < PASS_MIN_GAIN_X_M:
                    continue
            elif mode == "lateral":
                if abs(gain_y) < LATERAL_MIN_GAIN_Y_M:
                    continue
                if gain_x < -PASS_MIN_GAIN_X_M:
                    continue          # A lateral switch never goes backwards.
            else:  # back
                if gain_x > 0.0:
                    continue
            lane = self._lane_clearance(ball.x, ball.y, pose.x, pose.y)
            if lane < PASS_LANE_CLEARANCE_M:
                continue
            if self._space_around(pose.x, pose.y) < PASS_RECEIVER_SPACE_M:
                continue
            if mode == "forward":
                score = gain_x + 0.4 * lane
            elif mode == "lateral":
                score = abs(gain_y) + 0.4 * lane
            else:
                score = -abs(gain_x) + 0.4 * lane   # Prefer the shortest回传.
            if score > best_score:
                best_score, best = score, (pose.x, pose.y)
        return best

    def _pick_restart_target(self) -> tuple[float, float] | None:
        """Pick a low-risk restart outlet when normal pass thresholds fail."""
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return None

        gx, gy = own_goal(ctx)
        own_goal_angle = angle_to(ball.x, ball.y, gx, gy)
        best: tuple[float, float] | None = None
        best_score = -math.inf
        for teammate in ctx.teammates.values():
            pose = teammate.pose
            if pose is None or teammate.player_id == self.id:
                continue
            if ctx.game is not None:
                state = ctx.game.get_player_state(
                    self.config.team_id, teammate.player_id,
                )
                if state is not None and state.penalty != Penalty.NONE:
                    continue
            direction = angle_to(ball.x, ball.y, pose.x, pose.y)
            if math.cos(direction - own_goal_angle) > 0.65:
                continue
            lane = self._lane_clearance(ball.x, ball.y, pose.x, pose.y)
            if lane < PASS_LANE_CLEARANCE_M * 0.3:
                continue
            if self._space_around(pose.x, pose.y) < PASS_RECEIVER_SPACE_M * 0.6:
                continue
            gain_x = pose.x - ball.x
            gain_y = pose.y - ball.y
            score = gain_x + 0.5 * abs(gain_y) + 0.4 * min(lane, 3.0)
            if score > best_score:
                best_score, best = score, (pose.x, pose.y)
        if best is not None:
            return best

        half_l = ctx.field.length / 2.0
        if ball.x > half_l - 2.0:
            target_x = ball.x - 2.0
        else:
            target_x = ball.x + 2.0
        target_x = clamp(target_x, -half_l + 0.9, half_l - 0.9)
        target_y = ball.y * 0.25
        return clamp_inside_field(ctx, target_x, target_y, margin=0.6)

    def plan_aggressive_kick(self) -> tuple[float, float, str] | None:
        """Choose an on-ball action with possession-first priority.

        Order: shoot, safe forward pass, lateral switch, backward keep-ball
        pass. A long ball or a backfield clearance is a risk-control fallback
        used only when no safe pass exists and the ball is under pressure.
        Cooldowns stop the choice from flipping every frame; an indirect free
        kick disables the direct shot.
        """
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return None

        pressers = self._count_pressers()

        # 1. Shot: highest value action only when the lane is actually usable.
        if self.allow_direct_shot and self._cooldown_ready(SHOT_KEY):
            gx, gy = opponent_goal(ctx)
            goal_dist = dist(ball.x, ball.y, gx, gy)
            shot_range = SHOT_RANGE_M
            if self._team_score_difference() < 0:
                shot_range += SCORE_TRAIL_SHOT_RANGE_BONUS_M
            if self._space_around(gx, gy) >= OPEN_GOAL_CLEAR_M:
                shot_range = max(shot_range, SHOT_RANGE_OPEN_GOAL_M)
            if goal_dist <= shot_range:
                shot_target = self._best_shot_target(gx, gy)
                lane = self._shot_path_clearance(shot_target)
                close_lane = (
                    goal_dist <= SHOT_RANGE_CLOSE_M
                    and lane >= SHOT_LANE_CLOSE_CLEARANCE_M
                )
                if close_lane or lane >= SHOT_LANE_CLEARANCE_M:
                    self._start_cooldown(SHOT_KEY, SHOT_COOLDOWN_S)
                    return (
                        angle_to(ball.x, ball.y, *shot_target),
                        SHOT_POWER,
                        "shot",
                    )

        # 2. Safe forward pass to a teammate ahead of the ball.
        if self._cooldown_ready(PASS_FORWARD_KEY):
            target = self._pick_pass_target("forward")
            if target is not None:
                self._start_cooldown(PASS_FORWARD_KEY, PASS_COOLDOWN_S)
                return (
                    angle_to(ball.x, ball.y, *target),
                    self._pass_power(dist(ball.x, ball.y, *target)),
                    "pass_forward",
                )

        # Do not play risky lateral/back passes across our own goal mouth.
        unsafe_backfield = ball.x <= BACKFIELD_UNSAFE_PASS_X_M

        # 3. Switch play to the other wing when the forward lane is blocked.
        if not unsafe_backfield and self._cooldown_ready(PASS_LATERAL_KEY):
            target = self._pick_pass_target("lateral")
            if target is not None:
                self._start_cooldown(PASS_LATERAL_KEY, PASS_COOLDOWN_S)
                return (
                    angle_to(ball.x, ball.y, *target),
                    self._pass_power(dist(ball.x, ball.y, *target)),
                    "pass_lateral",
                )

        # Restarts must still produce a controlled first touch even when normal
        # lane and receiver-space thresholds are temporarily unavailable.
        if not self.allow_direct_shot and self._cooldown_ready(PASS_FORWARD_KEY):
            target = self._pick_restart_target()
            if target is not None:
                self._start_cooldown(PASS_FORWARD_KEY, PASS_COOLDOWN_S)
                return (
                    angle_to(ball.x, ball.y, *target),
                    max(
                        5.0,
                        self._pass_power(dist(ball.x, ball.y, *target)),
                    ),
                    "restart_pass",
                )

        # 4. Backfield under real pressure with no safe pass: clear it upfield.
        if (
            ball.x <= CLEARANCE_BALL_X_M
            and pressers >= CLEARANCE_MIN_PRESSERS
            and self._cooldown_ready(CLEARANCE_KEY)
        ):
            target = self._pick_forward_target(
                ball,
                CLEARANCE_MIN_GAIN_X_M,
                LONG_BALL_CLEARANCE_M,
                wide_only=True,
            )
            if target is None:
                target = self._pick_forward_target(
                    ball,
                    CLEARANCE_MIN_GAIN_X_M,
                    0.0,
                    wide_only=True,
                )
            if target is not None:
                self._start_cooldown(CLEARANCE_KEY, CLEARANCE_COOLDOWN_S)
                return (
                    angle_to(ball.x, ball.y, *target),
                    CLEARANCE_POWER,
                    "clearance",
                )

        # 5. Keep possession with a backward or lateral pass when pressure is low.
        if (
            not unsafe_backfield
            and pressers <= BACK_PASS_MAX_PRESSURE
            and self._cooldown_ready(PASS_BACK_KEY)
        ):
            target = self._pick_pass_target("back")
            if target is not None:
                self._start_cooldown(PASS_BACK_KEY, PASS_COOLDOWN_S)
                return (
                    angle_to(ball.x, ball.y, *target),
                    self._pass_power(dist(ball.x, ball.y, *target)),
                    "pass_back",
                )

        # 6. Long ball to a wide/forward space as a last forward option.
        if (
            pressers >= CLEARANCE_MIN_PRESSERS
            and ball.x >= LONG_BALL_BALL_X_M
            and self._cooldown_ready(LONG_BALL_KEY)
        ):
            target = self._pick_forward_target(
                ball,
                LONG_BALL_MIN_GAIN_X_M,
                LONG_BALL_CLEARANCE_M,
                wide_only=True,
            )
            if target is not None:
                self._start_cooldown(LONG_BALL_KEY, LONG_BALL_COOLDOWN_S)
                return (
                    angle_to(ball.x, ball.y, *target),
                    LONG_BALL_POWER,
                    "long_ball",
                )

        return None

    def attack(self, kick_target: tuple[float, float] | None = None) -> None:
        """Chase the ball and shoot toward ``kick_target`` or the opponent goal.

        Kick hysteresis enters below ENTER and exits above EXIT, where EXIT is
        greater than ENTER, to prevent threshold oscillation. Normal challenges
        approach the ball directly without avoidance. The chase target lies
        ``CHASE_BEHIND_M`` behind the ball on the ball-to-goal line, naturally
        aligning the player for a shot on arrival.

        In normal play, a kickable ball in the front cone is struck directly at
        the opponent goal with maximum power only when the whole shot lane is
        clear. Blocked shots fall back to safe passing or realignment. Restarts
        still use :meth:`plan_aggressive_kick` so their first touch remains
        legal.

        The visual kick only kicks a ball in front of the robot, so a ball beside
        the robot would otherwise be ignored. In that case the robot pivots in
        place just enough to bring the ball into the front cone and then shoots
        with the usual aim. If the required pivot would push the shot direction
        too far off the body axis, or the ball is effectively behind, the player
        keeps the original behaviour of walking around the ball.
        """
        ctx = self.context
        pose = self.pose
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None or pose is None:
            self.stop()
            return
        if kick_target is None:
            kick_target = opponent_goal(ctx)

        d = dist(pose.x, pose.y, ball.x, ball.y)
        if d <= (KICK_EXIT_M if self._kicking else KICK_ENTER_M):
            ball_bearing = self._body_bearing(pose, ball.x, ball.y)
            kick_limit = KICK_FRONT_CONE_RAD - KICK_FRONT_CONE_SETTLE_RAD
            if abs(ball_bearing) <= kick_limit:
                if self.allow_direct_shot:
                    gx, gy = opponent_goal(ctx)
                    goal_dist = dist(ball.x, ball.y, gx, gy)
                    shot_target = self._best_shot_target(gx, gy)
                    lane = self._shot_path_clearance(shot_target)
                    lane_required = (
                        SHOT_LANE_CLOSE_CLEARANCE_M
                        if goal_dist <= SHOT_RANGE_CLOSE_M
                        else SHOT_LANE_CLEARANCE_M
                    )
                    direct_range = max(SHOT_RANGE_M, SHOT_RANGE_OPEN_GOAL_M)
                    route_open = (
                        goal_dist <= direct_range
                        and lane >= lane_required
                    )
                    kick_direction = angle_to(ball.x, ball.y, *shot_target)
                    aim_offset = normalize_angle(kick_direction - pose.theta)
                    if route_open and abs(aim_offset) <= KICK_FRONT_CONE_RAD:
                        self._kicking = True
                        self.action = "attack:direct_goal"
                        self.kick(kick_direction, SHOT_POWER)
                        return
                    if route_open and d >= KICK_PIVOT_MIN_DIST_M:
                        heading = self._pivot_for_kick(
                            pose, ball_bearing, kick_direction,
                        )
                        if heading is not None:
                            self.action = "attack:goal_turn"
                            self.face_to(heading)
                            return
                aggressive = self.plan_aggressive_kick()
                if aggressive is not None:
                    kick_direction, kick_power, label = aggressive
                    self._kicking = True
                    self.action = f"attack:{label}"
                    self._draw_kick_target(
                        self._goal_target_for_direction(kick_direction)
                    )
                    self.kick(kick_direction, kick_power)
                    return
                self._kicking = False
                self.release_kick()

            # Pivoting is only safe with clearance from the ball; otherwise a
            # turn could nudge it and lose the chance, so walk around instead.
            if (
                abs(ball_bearing) > kick_limit
                and d >= KICK_PIVOT_MIN_DIST_M
                and abs(ball_bearing) <= KICK_TURN_MAX_RAD
            ):
                kick_direction = angle_to(ball.x, ball.y, *kick_target)
                heading = self._pivot_for_kick(
                    pose, ball_bearing, kick_direction,
                )
                if heading is not None:
                    self.action = "attack:turn"
                    self.face_to(heading)
                    return
        else:
            self._kicking = False
            self.release_kick()

        standoff = max(CHASE_BEHIND_M, KICK_PIVOT_MIN_DIST_M + 0.15)
        target = _behind_ball(ball.x, ball.y, kick_target, standoff)
        target = clamp_inside_field(ctx, *target, margin=0.4)
        self.action = "attack:align"
        self.walk_to(
            target,
            face=angle_to(target[0], target[1], ball.x, ball.y),
        )

    def guard(self) -> None:
        """Guard from the center of the goal area.

        Fall back to that position when no active threat is available.
        """
        ctx = self.context
        pose = self.pose
        home = own_goal_area_center(ctx) if ctx is not None else None
        if home is None or pose is None:
            self.action = "guard:stop"
            self.stop()
            return

        ball = ctx.ball if ctx is not None else None

        # Face the ball for faster reactions, or the opponent's goal if unseen.
        face = 0.0
        if GUARD_FACE_BALL and ball is not None:
            face = angle_to(
                pose.x, pose.y, ball.x, ball.y,
            )

        self.action = "guard:home"
        debugdraw.point(
            home[0], home[1], rgb=(0.0, 0.6, 1.0), scale=0.2, ns="guard_home",
        )
        self.walk_to(home, face=face, avoid_ball=True, avoid_robots=True)

    def support(self) -> None:
        """Support on the ball-to-own-goal line at ``SUPPORT_DIST_M``.

        This places the player on a blocking line between the ball and our goal.
        """
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            self.stop()
            return

        gx, gy = own_goal(ctx)
        bx, by = (ball.x, ball.y)
        dx, dy = gx - bx, gy - by
        d = math.hypot(dx, dy)
        if d < 1e-6:
            ux, uy = -1.0, 0.0
        else:
            ux, uy = dx / d, dy / d
        along = min(SUPPORT_DIST_M, d)
        tx = bx + ux * along
        ty = by + uy * along

        # Stay 0.3 m inside our goal line and clamp laterally inside the field.
        half_l = ctx.field.length / 2.0
        half_w = ctx.field.width / 2.0 - 0.3
        tx = clamp(tx, -half_l + 0.3, half_l)
        ty = clamp(ty, -half_w, half_w)
        self.move_to_position((tx, ty))

    def press(self) -> None:
        """Contain the ball carrier diagonally before committing to a tackle.

        The approach point sits goal-side of the ball and slightly toward the
        centre, which blocks the direct route to goal. Once close enough the
        normal attack action takes over and competes for the ball.
        """
        ctx = self.context
        pose = self.pose
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None or pose is None:
            self.stop()
            return

        if dist(pose.x, pose.y, ball.x, ball.y) <= DEFEND_COMMIT_DIST_M:
            self.attack()
            return

        gx, gy = own_goal(ctx)
        dx, dy = gx - ball.x, gy - ball.y
        d = math.hypot(dx, dy)
        if d < 1e-6:
            ux, uy = -1.0, 0.0
        else:
            ux, uy = dx / d, dy / d

        tx = ball.x + ux * DEFEND_APPROACH_OFFSET_M
        ty = ball.y + uy * DEFEND_APPROACH_OFFSET_M
        if abs(ty) > 1e-6:
            ty -= math.copysign(
                min(abs(ty), DEFEND_APPROACH_CENTER_M), ty,
            )

        if not self.action.startswith("defend:"):
            self.action = "defend:contain"
        self.move_to_position(
            clamp_inside_field(ctx, tx, ty, margin=0.55),
        )

    def take_kickoff(self, kick_target: tuple[float, float] | None = None) -> None:
        """Stage behind the ball for our restart, then approach and kick."""
        ctx = self.context
        pose = self.pose
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None or pose is None:
            self.stop()
            return
        if kick_target is None:
            kick_target = opponent_goal(ctx)
        kick_dir = angle_to(ball.x, ball.y, *kick_target)
        cos_k, sin_k = math.cos(kick_dir), math.sin(kick_dir)
        rel_x, rel_y = pose.x - ball.x, pose.y - ball.y
        behind = rel_x * cos_k + rel_y * sin_k          # Negative is our side.
        lateral = abs(-rel_x * sin_k + rel_y * cos_k)
        if behind > KICKOFF_FRONT_MARGIN or lateral > KICKOFF_LATERAL_TOL:
            stage = (
                ball.x - cos_k * KICKOFF_STAGE_M,
                ball.y - sin_k * KICKOFF_STAGE_M,
            )
            self.release_kick()
            self.walk_to(stage, face=kick_dir, avoid_ball=True)
        else:
            self.attack(kick_target)

    def move_to_position(self, target: tuple[float, float] | None) -> None:
        """Move to a support/defense position while facing and avoiding the ball."""
        if target is None:
            self.stop()
            return
        face = None
        ctx = self.context
        pose = self.pose
        ball = ctx.ball if ctx is not None else None
        if ball is not None and pose is not None:
            face = angle_to(pose.x, pose.y, ball.x, ball.y)
        self.release_kick()
        self.walk_to(
            target,
            face=face,
            avoid_ball=True,
            avoid_robots=True,
        )

    @staticmethod
    def _angular(err: float) -> float:
        return clamp(ANGULAR_GAIN * err, -MAX_ANGULAR, MAX_ANGULAR)
