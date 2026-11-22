"""Geometry for defendable shots and conservative first touches."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..soccer_framework import BallState, GameControlState, GameState, PlayContext, Pose2D, SetPlay
from ..tactics.navigation import Obstacle

if TYPE_CHECKING:
    from ..runtime import SoccerKit
    from .ball_motion import BallMotion


def keeper_intercept_y(
    ball: BallState,
    motion: BallMotion | None,
    keeper_x: float,
    goal_x: float,
    goal_width: float,
) -> float | None:
    """Project a fast incoming ball only when it can reach the goal mouth."""
    if motion is None or motion.speed < 0.8 or motion.direction_x >= -0.25 or ball.x <= keeper_x:
        return None
    goal_distance = (goal_x - ball.x) / motion.direction_x
    if goal_distance > motion.distance_after(2.5):
        return None
    goal_y = ball.y + goal_distance * motion.direction_y
    if abs(goal_y) > goal_width / 2 + 0.12:
        return None
    return ball.y + (keeper_x - ball.x) * motion.direction_y / motion.direction_x


def needs_controlled_touch(game: GameControlState, team_id: int) -> bool:
    return (
        game.state == GameState.PLAYING
        and not game.stopped
        and game.is_restart_for_team(team_id)
        and game.set_play in {SetPlay.NONE, SetPlay.INDIRECT_FREE_KICK, SetPlay.THROW_IN, SetPlay.GOAL_KICK}
    )


def restart_touch_target(kit: SoccerKit, player_id: int, context: PlayContext) -> Pose2D | None:
    """Keep the first touch on a legal, in-field lane with no shot at either goal."""
    ball = context.known_ball

    opponent_goal = kit.field.opponent_goal_x()
    own_goal = kit.field.own_goal_x()
    half_goal = kit.config.goal_width / 2 + 0.20
    obstacles = (
        kit.obstacles.opponent_obstacles(context)
        + kit.obstacles.teammate_obstacles(player_id, context)
    )
    candidates = [
        kit.field.clamp_inside_field(Pose2D(ball.x + dx, ball.y + dy), margin=0.40)
        for dx, dy in ((0.65, -1.5), (0.65, 1.5), (1.5, -1.1), (1.5, 1.1), (0.0, -1.6), (0.0, 1.6))
    ]

    def candidate_score(target: Pose2D, blockers: tuple[Obstacle, ...]) -> float:
        dx, dy = target.x - ball.x, target.y - ball.y
        if math.hypot(dx, dy) < 0.60:
            return -math.inf
        for goal in (own_goal, opponent_goal):
            if dx and (goal - ball.x) / dx > 0:
                if abs(ball.y + (goal - ball.x) * dy / dx) < half_goal:
                    return -math.inf
        lane = kit.targeting.lane_clear_score(ball.x, ball.y, target.x, target.y, blockers)
        return 2.0 * lane + 0.24 * dx - 0.15 * abs(target.y)

    scored = ((candidate_score(target, obstacles), target) for target in candidates)
    score, target = max(scored, key=lambda item: item[0])
    return target if math.isfinite(score) else None
