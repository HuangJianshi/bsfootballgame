"""Coordinate possession and support in open play."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

from ..behavior_tree.blackboard import BlackboardKeys
from ..soccer_framework import BallState, PlayContext, Pose2D, RobotCommand, SetPlay
from ..tactics.geometry import clamp, normalize_angle
from ..tactics.navigation import Obstacle
from .ball_motion import BallMotion, BallMotionTracker
from .chaser_fallback import ChaserFallbackPlaybook, is_open_play
from .default_roles import ChaserRole, GoalkeeperRole, SupporterRole
from .nodes import AttackSubtreeConfig, MoveToTarget
from .triad_tactics import keeper_intercept_y, needs_controlled_touch, restart_touch_target
from .playbook import ROLE_CHASER, ROLE_GOALKEEPER, ROLE_SUPPORTER, RoleAssignment
from .rolling_intercept import InterceptPlan, RollingIntercept
from .role import RoleStrategy

if TYPE_CHECKING:
    import py_trees

    from ..runtime import SoccerKit


_INTERCEPTOR = "interceptor"
_SWITCH_ADVANTAGE_SEC = 0.30
_KEEPER_SWITCH_ADVANTAGE_M = 0.65
_MAX_CHASE_POSE_AGE_SEC = 0.55
_RESTART_WAIT = "restart_wait"


def _defending(ball: BallState) -> bool:
    return ball.x <= -0.35


def _keeper_home(kit: SoccerKit, ball: BallState, motion: BallMotion | None = None) -> Pose2D:
    goal = kit.field.own_goal_x()
    depth = 0.82
    x = goal + depth
    y = clamp(ball.y * depth / max(depth, ball.x - goal), -0.85, 0.85)
    predicted = keeper_intercept_y(ball, motion, x, goal, kit.config.goal_width)
    if predicted is not None:
        y = clamp(predicted, -1.0, 1.0)
    return Pose2D(x, y, kit.field.face_ball_theta(x, y, ball))


class CoordinatedPlaybook(ChaserFallbackPlaybook):
    """Assign one field player to the ball, with the other available for the next play."""

    def __init__(self, kit: SoccerKit):
        super().__init__(kit)
        self.ball_motion = BallMotionTracker()
        self.intercept: InterceptPlan | None = None
        self._owner: int | None = None
        self.keeper: int | None = None
        self._restart_key: tuple[SetPlay, int] | None = None
        self._restart_taker: int | None = None
        self._cover_sides: dict[int, float] = {}
        self._last_tick: float | None = None
        self._receivers = {player: RollingIntercept() for player in kit.config.player_ids}
        self.role_registry.replace(CoordinatedChaser(self))
        self.role_registry.replace(CoordinatedSupport(self))
        self.role_registry.replace(InterceptRole(self))
        self.role_registry.replace(CoordinatedKeeper(self))
        self.register_role(RestartWaitRole())

    def _reset(self) -> None:
        self._owner = None
        self.keeper = None
        self._cover_sides.clear()
        self._last_tick = None
        self.intercept = None
        self.ball_motion.reset()
        for receiver in self._receivers.values():
            receiver.reset()

    def assign_roles(self, context: PlayContext) -> RoleAssignment:
        ball = context.ball
        now = self._board.read(BlackboardKeys.NOW)
        if context.game is None or ball is None or now is None:
            self._reset()
            self._clear_restart()
            return RoleAssignment()
        if not is_open_play(context):
            self._reset()
            if context.known_game.is_restart_for_team(self.kit.config.team_id):
                return self._assign_restart(context)
            self._clear_restart()
            return super().assign_roles(context)
        self._clear_restart()
        if self._last_tick is not None and not 0 <= now - self._last_tick <= 0.25:
            self._reset()
        self._last_tick = now
        self.ball_motion.update(ball, now)

        config = self.kit.config
        poses = {
            player: pose
            for player in config.player_ids
            if (pose := self._chase_pose(player, context)) is not None
        }
        keeper = self._select_keeper(poses)
        keeper_changed = self.keeper != keeper
        self.keeper = keeper
        roles = {keeper: ROLE_GOALKEEPER} if keeper is not None else {}
        approach: dict[int, float] = {}
        plans: dict[int, InterceptPlan] = {}
        touching: dict[int, float] = {}
        field_distances: list[float] = []
        opponents = self.kit.obstacles.opponent_obstacles(context)
        for player, receiver in self._receivers.items():
            if player == keeper or player not in poses:
                receiver.reset()
                continue
            pose = poses[player]
            robot = context.teammates[player]
            roles[player] = ROLE_SUPPORTER
            dx, dy = ball.x - pose.x, ball.y - pose.y
            distance = math.hypot(dx, dy)
            field_distances.append(distance)
            turn = abs(normalize_angle(math.atan2(dy, dx) - pose.theta))
            tuning = config.strategy
            approach[player] = (
                max(0.0, distance - 0.25) / max(0.1, tuning.max_linear_speed)
                + max(0.0, turn - 0.20) / max(0.1, tuning.max_angular_speed)
            )
            forward = dx * math.cos(pose.theta) + dy * math.sin(pose.theta)
            across = -dx * math.sin(pose.theta) + dy * math.cos(pose.theta)
            if 0.0 <= forward <= 0.40 and abs(across) <= 0.21:
                touching[player] = math.hypot(forward - 0.20, across)

            motion = self.ball_motion.motion if (
                ball.x > self.kit.field.own_goal_x() + config.goal_area_length + 0.65
                and not self.kit.targeting.ball_in_own_defensive_area(ball)
                and tuning.max_linear_speed >= 0.8 and tuning.max_angular_speed >= 0.5
            ) else None
            plan = receiver.plan(
                now, robot, ball, motion,
                opponents + self.kit.obstacles.teammate_obstacles(player, context),
                (config.field_length / 2 - 0.35, config.field_width / 2 - 0.35),
            )
            if plan is not None:
                plans[player] = plan

        self.intercept = None
        if touching:
            owner = self._owner if self._owner in touching else min(touching, key=touching.__getitem__)
        elif keeper is not None and self._keeper_claims(keeper, context, field_distances):
            owner = keeper
        elif plans:
            owner = self._choose({player: plan.seconds for player, plan in plans.items()})
            self.intercept = plans[owner]
        else:
            owner = self._choose(approach) if approach else None
        if keeper is not None and owner == keeper:
            field_players = sorted(approach, key=lambda pid: (poses[pid].y, pid))
            # Keep existing lanes on later ticks. A player can return from a penalty mid-clearance.
            if keeper_changed or self._owner != keeper or set(self._cover_sides) != set(field_players):
                if len(field_players) == 1:
                    player = field_players[0]
                    self._cover_sides = {player: -1.0 if poses[player].y < 0.0 else 1.0}
                else:
                    self._cover_sides = {
                        pid: -1.0 if index == 0 else 1.0 for index, pid in enumerate(field_players)
                    }
        self._owner = owner
        if owner is not None and owner != keeper:
            roles[owner] = _INTERCEPTOR if self.intercept is not None else ROLE_CHASER
        return RoleAssignment(roles)

    def _chase_pose(self, player_id: int, context: PlayContext) -> Pose2D | None:
        pose = super()._chase_pose(player_id, context)
        now = self._board.read(BlackboardKeys.NOW)
        if pose is None or now is None:
            return None
        if not context.teammates[player_id].is_recent(now, _MAX_CHASE_POSE_AGE_SEC):
            return None
        return pose if all(math.isfinite(value) for value in (pose.x, pose.y, pose.theta)) else None

    def _select_keeper(self, poses: dict[int, Pose2D]) -> int | None:
        preferred = self.kit.config.goalkeeper_player_id()
        if preferred in poses:
            return preferred
        if not poses:
            return None
        goal = self.kit.field.own_goal_x()
        distance = {pid: math.hypot(pose.x - goal, pose.y) for pid, pose in poses.items()}
        best = min(distance, key=lambda pid: (distance[pid], pid))
        if self.keeper in distance and distance[self.keeper] <= distance[best] + _KEEPER_SWITCH_ADVANTAGE_M:
            return self.keeper
        return best

    def _clear_restart(self) -> None:
        self._restart_key = None
        self._restart_taker = None

    def _assign_restart(self, context: PlayContext) -> RoleAssignment:
        game, ball = context.known_game, context.known_ball
        key = game.set_play, game.kicking_team
        keeper = self.kit.config.goalkeeper_player_id()
        poses = {
            player: pose
            for player in self.kit.config.player_ids
            if (pose := self._chase_pose(player, context)) is not None
        }
        if not poses:
            return RoleAssignment()
        candidates = tuple(poses)
        if keeper in poses and len(poses) > 1 and game.set_play != SetPlay.GOAL_KICK:
            candidates = tuple(player for player in poses if player != keeper)
        if key != self._restart_key or self._restart_taker not in candidates:
            self._restart_taker = min(
                candidates, key=lambda player: (math.hypot(poses[player].x - ball.x, poses[player].y - ball.y), player)
            )
            self._restart_key = key
        kickoff = game.set_play == SetPlay.NONE
        return RoleAssignment({
            player: ROLE_CHASER if player == self._restart_taker else
            _RESTART_WAIT if kickoff or player == keeper else ROLE_SUPPORTER
            for player in poses
        })

    def _keeper_claims(self, keeper: int, context: PlayContext, field_distances: list[float]) -> bool:
        ball = context.known_ball
        config = self.kit.config
        goal = self.kit.field.own_goal_x()
        pose = self._chase_pose(keeper, context)
        if pose is None:
            return False
        if not field_distances:
            # When alone, sweep our defensive third. Stay goal-side of distant attacks.
            return goal - 0.12 <= ball.x <= goal + 3.6 and abs(ball.y) <= 2.7
        if not goal - 0.12 <= ball.x <= goal + config.goal_area_length + 0.55:
            return False
        if abs(ball.y) > config.goal_width / 2 + 0.45:
            return False
        distance = math.hypot(ball.x - pose.x, ball.y - pose.y)
        nearest_field = min(field_distances, default=math.inf)
        if self._owner == keeper:
            return distance <= 1.12 and distance <= nearest_field + 0.08
        return distance <= 0.88 and distance + 0.32 < nearest_field

    def _choose(self, seconds: dict[int, float]) -> int:
        best = min(seconds, key=seconds.__getitem__)
        previous = self._owner
        if previous is not None and seconds.get(previous, math.inf) <= seconds[best] + _SWITCH_ADVANTAGE_SEC:
            return previous
        return best


class CoordinatedKeeper(GoalkeeperRole):
    def __init__(self, playbook: CoordinatedPlaybook) -> None:
        self._playbook = playbook

    def wants_to_kick(self, kit: SoccerKit, context: PlayContext) -> bool:
        if is_open_play(context):
            keeper = self._playbook.keeper
            return keeper is not None and self._playbook._owner == keeper
        return super().wants_to_kick(kit, context)

    def target(self, kit: SoccerKit, context: PlayContext) -> Pose2D:
        if is_open_play(context) and _defending(context.known_ball) and not self.wants_to_kick(kit, context):
            return _keeper_home(kit, context.known_ball, self._playbook.ball_motion.motion)
        return super().target(kit, context)

    def _attack_config(self, kit: SoccerKit, player_id: int) -> AttackSubtreeConfig:
        def release(context: PlayContext) -> RobotCommand | None:
            if is_open_play(context) and kit.kicker.is_active(player_id) and not self.wants_to_kick(kit, context):
                return RobotCommand.stop("defense: keeper yields ball")
            return None

        return replace(super()._attack_config(kit, player_id), command_fn=release)


class CoordinatedChaser(ChaserRole):
    def __init__(self, playbook: CoordinatedPlaybook) -> None:
        super().__init__(fast_move_when=is_open_play)
        self._playbook = playbook
        self._aim_by_player: dict[int, tuple[PlayContext, Pose2D | None]] = {}

    def _open_play_aim(self, kit: SoccerKit, player_id: int, context: PlayContext) -> Pose2D | None:
        cached = self._aim_by_player.get(player_id)
        if cached is not None and cached[0] is context:
            return cached[1]
        # Select passes only to players available to receive. The keeper stays in goal.
        aim = kit.targeting.select_open_play_kick_target(
            player_id, context,
            lambda game, pid: pid != self._playbook.keeper and self._playbook._chase_pose(pid, context) is not None,
        )
        self._aim_by_player[player_id] = context, aim
        return aim

    def kick_target(self, kit: SoccerKit, player_id: int, context: PlayContext) -> Pose2D:
        if is_open_play(context):
            aim = self._open_play_aim(kit, player_id, context)
            return aim if aim is not None else kit.targeting.dribble_target(context.known_ball)
        if needs_controlled_touch(context.known_game, kit.config.team_id):
            aim = restart_touch_target(kit, player_id, context)
            if aim is None:
                raise ValueError("no legal first-touch direction")
            return aim
        return super().kick_target(kit, player_id, context)

    def _attack_config(self, kit: SoccerKit, player_id: int) -> AttackSubtreeConfig:
        def align_touch(context: PlayContext) -> bool:
            return is_open_play(context) or needs_controlled_touch(context.known_game, kit.config.team_id)

        def aim_target(context: PlayContext) -> Pose2D | None:
            if is_open_play(context):
                return self._open_play_aim(kit, player_id, context)
            return restart_touch_target(kit, player_id, context)

        def wants_kick(context: PlayContext) -> bool:
            if not align_touch(context):
                return True
            aim = aim_target(context)
            return aim is not None and kit.motion.kick_setup_move(player_id, context, aim) is None

        def prepare_kick(context: PlayContext) -> RobotCommand | None:
            if not align_touch(context):
                return None
            aim = aim_target(context)
            if aim is None:
                if not is_open_play(context) or kit.kicker.is_active(player_id):
                    return RobotCommand.stop("kick: lane blocked")
                return None
            setup = kit.motion.kick_setup_move(player_id, context, aim)
            if setup is not None and kit.kicker.is_active(player_id):
                return RobotCommand.stop("kick setup: release kick control")
            return setup

        return replace(
            super()._attack_config(kit, player_id),
            wants_kick_fn=wants_kick,
            command_fn=prepare_kick,
        )


class RestartWaitRole(RoleStrategy):
    name = _RESTART_WAIT

    def build_subtree(self, kit: SoccerKit, player_id: int) -> py_trees.behaviour.Behaviour:
        return MoveToTarget(
            kit,
            player_id,
            lambda context: Pose2D(context.known_ball.x, context.known_ball.y),
            command_fn=lambda context: RobotCommand.stop("restart: wait for the taker"),
        )


class InterceptRole(RoleStrategy):
    name = _INTERCEPTOR

    def __init__(self, playbook: CoordinatedPlaybook) -> None:
        self._playbook = playbook

    def build_subtree(self, kit: SoccerKit, player_id: int) -> py_trees.behaviour.Behaviour:
        def command(context: PlayContext) -> RobotCommand:
            plan = self._playbook.intercept
            return plan.command if plan is not None else RobotCommand.stop("intercept: plan cleared")

        return MoveToTarget(kit, player_id, lambda context: Pose2D(context.known_ball.x, context.known_ball.y), command_fn=command)


class CoordinatedSupport(SupporterRole):
    _BALL_CONTROL_RADIUS_M = 1.1
    _RACE_MARGIN_M = 0.25
    _COVER_DEPTH_M = 1.8
    _TRAIL_DEPTH_M = 1.45
    _TRAIL_WIDTH_M = 1.25
    _PASS_LANE_MIN = 0.86
    _RECEIVER_CLEARANCE_M = 1.1
    _RECEIVER_OFFSETS_M = ((0.6, 1.25), (0.95, 1.35))

    def __init__(self, playbook: CoordinatedPlaybook) -> None:
        self._playbook = playbook

    def build_subtree(self, kit: SoccerKit, player_id: int) -> py_trees.behaviour.Behaviour:
        reason = "supporter hold"

        def target(context: PlayContext) -> Pose2D:
            nonlocal reason
            pose, reason = self._plan(kit, player_id, context)
            return pose

        def release(context: PlayContext) -> RobotCommand | None:
            if is_open_play(context) and _defending(context.known_ball) and kit.kicker.is_active(player_id):
                return RobotCommand.stop("defense: field player yields ball")
            return None

        return MoveToTarget(
            kit, player_id, target, reason_fn=lambda: reason,
            fast_move_when=lambda context: is_open_play(context) and not _defending(context.known_ball),
            command_fn=release,
        )

    def _plan(self, kit: SoccerKit, player_id: int, context: PlayContext) -> tuple[Pose2D, str]:
        ball = context.known_ball
        if not is_open_play(context):
            return super().target(kit, player_id, context), "supporter hold"
        if _defending(ball):
            return self._defensive_target(kit, player_id, context), "defense: cover passing lane"

        assignment = self._playbook._board.read(BlackboardKeys.ROLES)
        if not isinstance(assignment, RoleAssignment):
            return super().target(kit, player_id, context), "supporter hold"
        chasers = assignment.players_of(ROLE_CHASER) or assignment.players_of(_INTERCEPTOR)
        receiver = self._playbook._chase_pose(player_id, context)
        chaser = self._playbook._chase_pose(chasers[0], context) if chasers else None
        if receiver is None or chaser is None:
            return super().target(kit, player_id, context), "supporter hold"

        teammates = kit.obstacles.teammate_obstacles(player_id, context)
        opponents = kit.obstacles.opponent_obstacles(context)
        intercept = self._playbook.intercept
        if intercept is not None:
            projected_ball = BallState(intercept.ball_x, intercept.ball_y)
            outlet = self._passing_outlet(kit, projected_ball, receiver, opponents, teammates)
            if outlet is not None:
                return outlet, "support: outlet for interceptor"
            trail = self._trailing_position(kit, projected_ball, receiver, opponents, teammates)
            if trail is not None:
                return trail, "support: cover interception"

        own_distance = math.hypot(ball.x - chaser.x, ball.y - chaser.y)
        their_distance = min(
            (math.hypot(ball.x - p.x, ball.y - p.y) for p in opponents),
            default=math.inf,
        )
        if (
            opponents and own_distance <= self._BALL_CONTROL_RADIUS_M
            and own_distance + self._RACE_MARGIN_M < their_distance
        ):
            outlet = self._passing_outlet(kit, ball, receiver, opponents, teammates)
            if outlet is not None:
                return outlet, "support: show for pass"
        if (
            their_distance <= self._BALL_CONTROL_RADIUS_M
            and their_distance + self._RACE_MARGIN_M < own_distance
        ):
            cover = self._cover_position(kit, ball)
            if self._has_teammate_spacing(kit, cover, teammates):
                return cover, "support: cover center"

        trail = self._trailing_position(kit, ball, receiver, opponents, teammates)
        if trail is not None:
            return trail, "support: follow play"
        return super().target(kit, player_id, context), "supporter hold"

    def _defensive_target(self, kit: SoccerKit, player_id: int, context: PlayContext) -> Pose2D:
        ball = context.known_ball
        pose = self._playbook._chase_pose(player_id, context)
        goal = kit.field.own_goal_x()
        # Leave the keeper's area clear even when the ball is already behind the cover player.
        box_front = goal + kit.config.goal_area_length + 1.60
        depth = clamp(ball.x - 1.35, box_front, -0.25)
        keeper = self._playbook.keeper
        keeper_claim = keeper is not None and self._playbook._owner == keeper
        teammates = kit.obstacles.teammate_obstacles(player_id, context)
        home = _keeper_home(kit, ball)
        spacing = max(1.15, kit.config.strategy.support_min_spacing_m)
        if keeper_claim:
            side = self._playbook._cover_sides.get(player_id, -1.0 if ball.y >= 0.0 else 1.0)
            lanes = (1.35 * side, 2.15 * side, 2.75 * side)
        else:
            # Screen the middle first, with an off-ball-side outlet when the central route is occupied.
            center = clamp(0.12 * ball.y, -0.40, 0.40)
            far_side = -1.0 if ball.y >= 0 else 1.0
            lanes = (center, 0.95 * far_side, 1.70 * far_side, -1.25 * far_side, 2.45 * far_side)
        candidates = []
        for row, x in enumerate((depth, min(-0.15, depth + 0.70), min(-0.15, depth + 1.40))):
            for lane, y in enumerate(lanes):
                target = kit.field.clamp_inside_field(Pose2D(x, y), margin=0.45)
                if (
                    math.hypot(target.x - ball.x, target.y - ball.y) < spacing
                    or (keeper is not None and math.hypot(target.x - home.x, target.y - home.y) < spacing)
                    or any(math.hypot(target.x - p.x, target.y - p.y) < spacing for p in teammates)
                ):
                    continue
                travel = math.hypot(target.x - pose.x, target.y - pose.y) if pose is not None else 0.0
                candidates.append((0.45 * row + 0.35 * lane + 0.18 * travel, target))
        if candidates:
            target = min(candidates, key=lambda candidate: candidate[0])[1]
        else:
            # When boxed in, keep the current position rather than closing toward the ball.
            target = pose if pose is not None else Pose2D(depth, lanes[0])
        return self._face_ball(kit, target, ball)

    def _trailing_position(
        self,
        kit: SoccerKit,
        ball: BallState,
        receiver: Pose2D,
        opponents: tuple[Obstacle, ...],
        teammates: tuple[Obstacle, ...],
    ) -> Pose2D | None:
        # Stay behind the ball across midfield and choose the reachable side with room.
        x = max(
            kit.field.own_goal_x() + kit.config.goal_area_length + 0.65,
            ball.x - self._TRAIL_DEPTH_M,
        )
        options: list[tuple[float, Pose2D]] = []
        for side in (-1, 1):
            target = kit.field.clamp_inside_field(
                Pose2D(x, ball.y + side * self._TRAIL_WIDTH_M), margin=0.55,
            )
            if not self._has_teammate_spacing(kit, target, teammates):
                continue
            space = min(
                (math.hypot(target.x - p.x, target.y - p.y) - p.radius for p in opponents),
                default=3.0,
            )
            travel = math.hypot(target.x - receiver.x, target.y - receiver.y)
            score = min(space, 3.0) - 0.55 * travel - 0.12 * abs(target.y)
            options.append((score, self._face_ball(kit, target, ball)))
        return max(options, key=lambda option: option[0])[1] if options else None

    def _passing_outlet(
        self,
        kit: SoccerKit,
        ball: BallState,
        receiver: Pose2D,
        opponents: tuple[Obstacle, ...],
        teammates: tuple[Obstacle, ...],
    ) -> Pose2D | None:
        tuning = kit.config.strategy
        if not tuning.pass_enabled:
            return None
        max_x = kit.field.opponent_goal_x() - kit.config.goal_area_length - 0.35
        options: list[tuple[float, Pose2D]] = []
        for forward, lateral in self._RECEIVER_OFFSETS_M:
            for side in (-1, 1):
                target = kit.field.clamp_inside_field(
                    Pose2D(min(ball.x + forward, max_x), ball.y + lateral * side),
                    margin=0.55,
                )
                if target.x - ball.x < tuning.pass_min_forward_m:
                    continue
                if (
                    math.hypot(target.x - ball.x, target.y - ball.y) < tuning.support_min_spacing_m
                    or not self._has_teammate_spacing(kit, target, teammates)
                ):
                    continue
                lane = kit.targeting.lane_clear_score(ball.x, ball.y, target.x, target.y, opponents)
                space = min(
                    (math.hypot(target.x - p.x, target.y - p.y) - p.radius for p in opponents),
                    default=3.0,
                )
                if lane < self._PASS_LANE_MIN or space < self._RECEIVER_CLEARANCE_M:
                    continue
                travel = math.hypot(target.x - receiver.x, target.y - receiver.y)
                score = 2.0 * lane + 0.25 * min(space, 3.0) - 0.40 * travel
                options.append((score, self._face_ball(kit, target, ball)))
        return max(options, key=lambda option: option[0])[1] if options else None

    def _cover_position(self, kit: SoccerKit, ball: BallState) -> Pose2D:
        own_goal = kit.field.own_goal_x()
        x = max(own_goal + kit.config.goal_area_length + 0.65, ball.x - self._COVER_DEPTH_M)
        goal_line_fraction = (x - own_goal) / (ball.x - own_goal)
        target = kit.field.clamp_inside_field(Pose2D(x, ball.y * goal_line_fraction))
        return self._face_ball(kit, target, ball)

    @staticmethod
    def _has_teammate_spacing(kit: SoccerKit, target: Pose2D, teammates: tuple[Obstacle, ...]) -> bool:
        minimum = kit.config.strategy.support_min_spacing_m
        return all(math.hypot(target.x - p.x, target.y - p.y) >= minimum for p in teammates)

    @staticmethod
    def _face_ball(kit: SoccerKit, target: Pose2D, ball: BallState) -> Pose2D:
        return Pose2D(target.x, target.y, kit.field.face_ball_theta(target.x, target.y, ball))
