"""Choose a stable ball challenger and offer a wide outlet in open play."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..behavior_tree.blackboard import BlackboardClient, BlackboardKeys, robot_status_key
from ..soccer_framework import GameState, PlayContext, Pose2D, RobotRuntimeStatus
from ..tactics.geometry import normalize_angle
from ..tactics.targeting.attack import should_make_restart_touch
from .default_roles import ChaserRole, SupporterRole
from .playbook import (
    ROLE_CHASER,
    ROLE_GOALKEEPER,
    ROLE_SUPPORTER,
    DefaultPlaybook,
    RoleAssignment,
)

if TYPE_CHECKING:
    from ..runtime import SoccerKit


class TransitionPlaybook(DefaultPlaybook):
    """Keep one available field player on the ball and adapt the second player's lane."""

    _TURN_COST_M_PER_RAD = 0.25
    _SWITCH_MARGIN_M = 0.55
    _MAX_TICK_GAP_SEC = 0.25

    def __init__(self, kit: SoccerKit):
        super().__init__(kit)
        self._board = BlackboardClient(name="TransitionPlaybook")
        self._previous_chaser: int | None = None
        self._previous_tick: float | None = None
        self.role_registry.replace(RestartAwareChaser())
        self.role_registry.replace(WideSupport(self._board))

    def assign_roles(self, context: PlayContext) -> RoleAssignment:
        game = context.known_game
        ball = context.known_ball
        config = self.kit.config
        keeper = config.goalkeeper_player_id()
        open_play = _is_open_play(context)
        now = self._board.read(BlackboardKeys.NOW)
        scores: dict[int, float] = {}
        roles: dict[int, str] = {}

        for player_id in config.player_ids:
            if not self.kit.is_player_allowed(game, player_id):
                continue
            if player_id == keeper:
                roles[player_id] = ROLE_GOALKEEPER
                continue
            robot = context.teammates.get(player_id)
            status = self._board.read(robot_status_key(player_id))
            if robot is None or robot.pose is None or (
                isinstance(status, RobotRuntimeStatus) and not status.is_fall_down_normal
            ):
                continue
            pose = robot.pose
            slot = config.ready_slot_for_player(player_id)
            score = self.kit.targeting.ball_claim_score(slot, pose, ball)
            dx, dy = ball.x - pose.x, ball.y - pose.y
            if open_play and math.hypot(dx, dy) > 0.25:
                turn = normalize_angle(math.atan2(dy, dx) - pose.theta)
                score += abs(turn) * self._TURN_COST_M_PER_RAD
            scores[player_id] = score
            roles[player_id] = ROLE_SUPPORTER

        chaser = min(scores, key=lambda player_id: (scores[player_id], player_id), default=None)
        previous = self._previous_chaser
        if (
            open_play and chaser is not None and previous in scores
            and now is not None and self._previous_tick is not None
            and 0 <= now - self._previous_tick <= self._MAX_TICK_GAP_SEC
            and scores[previous] <= scores[chaser] + self._SWITCH_MARGIN_M
        ):
            chaser = previous
        if chaser is not None:
            roles[chaser] = ROLE_CHASER
        self._previous_chaser = chaser if open_play else None
        self._previous_tick = now if open_play else None
        return RoleAssignment(roles)


def _is_open_play(context: PlayContext) -> bool:
    game = context.known_game
    return game.state == GameState.PLAYING and not game.stopped and not game.has_kicking_team()


class RestartAwareChaser(ChaserRole):
    def kick_target(self, kit: SoccerKit, player_id: int, context: PlayContext) -> Pose2D:
        # A replacement chaser uses the template's restart-touch path even from the SIDE slot.
        if should_make_restart_touch(kit.config, context.known_game):
            return kit.targeting.select_kick_target(player_id, context, kit.is_player_allowed)
        return super().kick_target(kit, player_id, context)


class WideSupport(SupporterRole):
    _MAX_CHASER_DISTANCE_M = 1.2
    _POSSESSION_MARGIN_M = 0.45
    _OUTLET_FORWARD_M = 0.9
    _OUTLET_WIDTH_M = 2.0
    _MIN_OUTLET_SEPARATION_M = 1.15

    def __init__(self, board: BlackboardClient):
        self._board = board

    def target(self, kit: SoccerKit, player_id: int, context: PlayContext) -> Pose2D:
        ball = context.known_ball
        assignment = self._board.read(BlackboardKeys.ROLES)
        if (
            not kit.config.strategy.pass_enabled or not _is_open_play(context)
            or ball.x <= 0 or not isinstance(assignment, RoleAssignment)
        ):
            return super().target(kit, player_id, context)

        chasers = assignment.players_of(ROLE_CHASER)
        chaser = context.teammates.get(chasers[0]) if chasers else None
        if chaser is None or chaser.pose is None:
            return super().target(kit, player_id, context)
        opponents = [robot.pose for robot in context.opponents.values() if robot.pose is not None]
        chaser_distance = math.hypot(ball.x - chaser.pose.x, ball.y - chaser.pose.y)
        closest_opponent = min(
            (math.hypot(ball.x - pose.x, ball.y - pose.y) for pose in opponents),
            default=0.0,
        )
        if (
            chaser_distance > self._MAX_CHASER_DISTANCE_M
            or closest_opponent < chaser_distance + self._POSSESSION_MARGIN_M
        ):
            return super().target(kit, player_id, context)

        # Spread to the opposite lane without running into the opponent's goal area.
        direction = -1 if ball.y > 0 else 1
        x = min(
            ball.x + self._OUTLET_FORWARD_M,
            kit.field.opponent_goal_x() - kit.config.goal_area_length - 0.3,
        )
        if x - ball.x < kit.config.strategy.pass_min_forward_m:
            return super().target(kit, player_id, context)
        outlet = kit.field.clamp_inside_field(Pose2D(x, direction * self._OUTLET_WIDTH_M, 0.0))
        if any(
            math.hypot(outlet.x - pose.x, outlet.y - pose.y) < self._MIN_OUTLET_SEPARATION_M
            for pose in [chaser.pose, *opponents]
        ):
            return super().target(kit, player_id, context)
        return Pose2D(outlet.x, outlet.y, kit.field.face_ball_theta(outlet.x, outlet.y, ball))
