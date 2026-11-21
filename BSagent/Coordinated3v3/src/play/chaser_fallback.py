"""Keep the official playbook's chase decisions with an open-play fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..behavior_tree.blackboard import BlackboardClient, robot_status_key
from ..soccer_framework import GameState, PlayContext, Pose2D, RobotRuntimeStatus, SetPlay
from .playbook import ROLE_CHASER, ROLE_SUPPORTER, DefaultPlaybook, RoleAssignment

if TYPE_CHECKING:
    from ..runtime import SoccerKit


class ChaserFallbackPlaybook(DefaultPlaybook):
    def __init__(self, kit: SoccerKit):
        super().__init__(kit)
        self._board = BlackboardClient(name="ChaserFallbackPlaybook")

    def assign_roles(self, context: PlayContext) -> RoleAssignment:
        assignment = super().assign_roles(context)
        if not is_open_play(context):
            return assignment

        chasers = assignment.players_of(ROLE_CHASER)
        if not chasers or self._chase_pose(chasers[0], context) is not None:
            return assignment

        config = self.kit.config
        ball = context.known_ball
        candidates: list[tuple[float, int]] = []
        for player_id in assignment.players_of(ROLE_SUPPORTER):
            pose = self._chase_pose(player_id, context)
            if pose is not None:
                score = self.kit.targeting.ball_claim_score(
                    config.ready_slot_for_player(player_id), pose, ball,
                )
                candidates.append((score, player_id))
        if not candidates:
            return assignment

        _, replacement = min(candidates)
        roles = dict(assignment.by_player)
        del roles[chasers[0]]
        roles[replacement] = ROLE_CHASER
        return RoleAssignment(roles)

    def _chase_pose(self, player_id: int, context: PlayContext) -> Pose2D | None:
        robot = context.teammates.get(player_id)
        status = self._board.read(robot_status_key(player_id))
        if (
            robot is None or not self.kit.is_player_allowed(context.known_game, player_id)
            or isinstance(status, RobotRuntimeStatus) and not status.is_fall_down_normal
        ):
            return None
        return robot.pose


def is_open_play(context: PlayContext) -> bool:
    game = context.known_game
    return (
        game.state == GameState.PLAYING and not game.stopped
        and game.set_play == SetPlay.NONE and not game.has_kicking_team()
        and game.secondary_time <= 0
    )
