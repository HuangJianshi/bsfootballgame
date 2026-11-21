"""Chase/kick decisions and pass scoring.

Callers first use :mod:`predicates` to rule out sideline/recovery cases, then use
this module to choose this tick's kick target:

center view: pass, then shoot, then dribble
side view: pass if possible, otherwise clear, without shooting
choose best passing teammate by score
generic lane-obstacle score
"""

from __future__ import annotations

import math
from collections.abc import Callable

from ...soccer_framework import (
    BallState,
    GameControlState,
    Pose2D,
    SetPlay,
    SoccerConfig,
    PlayContext,
)
from ..geometry import clamp, field_to_relative, normalize_angle
from ..navigation import Obstacle, ObstacleCollector
from ..geometry import TeamFieldFrame
from . import recovery
from .predicates import ball_near_sideline


__all__ = [
    "PlayerAllowed",
    "best_pass_target",
    "dribble_target",
    "kick_reason",
    "lane_clear_score",
    "select_clear_or_pass_target",
    "select_kick_target",
    "select_open_play_kick_target",
    "shot_lane_is_clear",
    "should_make_restart_touch",
]


# Shared with upper layers: test whether a teammate can legally join tactics, excluding penalized/substitute players.
PlayerAllowed = Callable[[GameControlState, int], bool]


# Top-level selection


def select_kick_target(
    config: SoccerConfig,
    field: TeamFieldFrame,
    obstacles: ObstacleCollector,
    player_id: int,
    context: PlayContext,
    is_player_allowed: PlayerAllowed,
) -> Pose2D:
    """Decide this tick's aim target for a center chaser.

    Decision order: sideline recovery, restart touch, best pass, clear shot lane,
    and finally dribble forward.
    """

    ball = context.known_ball
    if ball_near_sideline(config, ball):
        return recovery.sideline_recovery_target(config, field, ball)

    game = context.known_game
    if should_make_restart_touch(config, game):
        teammate = best_pass_target(
            config, obstacles,
            player_id, context, is_player_allowed,
        )
        if teammate is not None:
            return Pose2D(teammate.x, teammate.y, 0.0)
        return dribble_target(config, field, ball)

    teammate = best_pass_target(
        config, obstacles,
        player_id, context, is_player_allowed,
    )
    if teammate is not None:
        return Pose2D(teammate.x, teammate.y, 0.0)

    if shot_lane_is_clear(config, field, obstacles, context):
        return Pose2D(field.opponent_goal_x(), 0.0, 0.0)
    return dribble_target(config, field, ball)


# Leave a ball-sized margin to the inside of each post and to nearby blockers.
_POST_MARGIN_M = 0.24
_SHOT_CLEARANCE_M = 0.08
_KICK_OBSTACLE_SCALE = 0.75  # Navigation disks include more room than the ball needs.
_GOAL_SAMPLES = 17
_QUICK_SHOT_RANGE_M = 3.4
_QUICK_SHOT_CLEARANCE_M = 0.22
_QUICK_SHOT_MAX_TURN_RAD = 0.17
_QUICK_SHOT_CONTEST_M = 0.45


def select_open_play_kick_target(
    config: SoccerConfig,
    field: TeamFieldFrame,
    obstacles: ObstacleCollector,
    player_id: int,
    context: PlayContext,
    is_player_allowed: PlayerAllowed,
) -> Pose2D | None:
    """Take a ready shot, or choose a pass, shot, or touch as the play develops."""
    ball = context.known_ball
    opponents = _kick_blockers(obstacles.opponent_obstacles(context))
    blockers = opponents + _kick_blockers(obstacles.teammate_obstacles(player_id, context))
    robot = context.teammates.get(player_id)
    shooter = robot.pose if robot is not None else None
    goal_x = field.opponent_goal_x()
    half_mouth = config.goal_width / 2.0 - _POST_MARGIN_M
    goal_targets = []
    shot_blockers = blockers
    if ball.x < goal_x and half_mouth > 0.0:
        shot_blockers += (
            Obstacle(goal_x, -config.goal_width / 2.0, 0.18),
            Obstacle(goal_x, config.goal_width / 2.0, 0.18),
        )
        goal_targets = [
            Pose2D(goal_x, half_mouth * (2.0 * i / (_GOAL_SAMPLES - 1) - 1.0))
            for i in range(_GOAL_SAMPLES)
        ]
        quick_shot = _quick_shot_target(ball, shooter, goal_targets, shot_blockers, opponents)
        if quick_shot is not None:
            return quick_shot

    teammate = best_pass_target(config, obstacles, player_id, context, is_player_allowed)
    if teammate is not None and _kick_clearance(ball, teammate, opponents) >= _SHOT_CLEARANCE_M:
        return Pose2D(teammate.x, teammate.y)

    if ball_near_sideline(config, ball):
        target = recovery.sideline_recovery_target(config, field, ball)
        if _kick_clearance(ball, target, blockers) >= _SHOT_CLEARANCE_M:
            return target

    if goal_targets:
        shot = _best_clear_target(ball, shooter, goal_targets, shot_blockers)
        if shot is not None:
            return shot

    # If the goal is covered, test controlled touches into either side lane.
    advance = config.strategy.dribble_advance_m
    touches = [
        field.clamp_inside_field(Pose2D(
            ball.x + advance * math.cos(angle), ball.y + advance * math.sin(angle),
        ), margin=0.20)
        for angle in (0.0, -0.55, 0.55, -1.0, 1.0)
    ]
    return _best_clear_target(ball, shooter, touches, blockers)


def _quick_shot_target(
    ball: BallState, shooter: Pose2D | None, targets: list[Pose2D],
    blockers: tuple[Obstacle, ...], opponents: tuple[Obstacle, ...],
) -> Pose2D | None:
    """Use an already aligned shot when the ball is near, close to goal, and uncontested."""
    if shooter is None or targets[0].x - ball.x > _QUICK_SHOT_RANGE_M:
        return None
    relative = field_to_relative(ball.x, ball.y, shooter)
    if not 0.10 <= relative.x <= 0.65 or abs(relative.y) > 0.16:
        return None
    if any(math.hypot(ball.x - p.x, ball.y - p.y) < _QUICK_SHOT_CONTEST_M for p in opponents):
        return None

    eligible = []
    for target in targets:
        dx, dy = target.x - ball.x, target.y - ball.y
        turn = abs(normalize_angle(math.atan2(dy, dx) - shooter.theta))
        if (
            turn <= _QUICK_SHOT_MAX_TURN_RAD
            and math.hypot(dx, dy) <= _QUICK_SHOT_RANGE_M
            and _kick_clearance(ball, target, blockers) >= _QUICK_SHOT_CLEARANCE_M
        ):
            eligible.append((turn, abs(target.y), target))
    if not eligible:
        return None
    return min(eligible, key=lambda item: item[:2])[2]


def _kick_blockers(obstacles: tuple[Obstacle, ...]) -> tuple[Obstacle, ...]:
    return tuple(Obstacle(o.x, o.y, o.radius * _KICK_OBSTACLE_SCALE) for o in obstacles)


def _best_clear_target(
    ball: BallState, shooter: Pose2D | None, targets: list[Pose2D], blockers: tuple[Obstacle, ...],
) -> Pose2D | None:
    best = None
    # The run-in angle captures which side of the ball the shooter is on.
    run_in = None
    if shooter is not None and math.hypot(ball.x - shooter.x, ball.y - shooter.y) >= 0.15:
        run_in = math.atan2(ball.y - shooter.y, ball.x - shooter.x)
    for target in targets:
        if math.hypot(target.x - ball.x, target.y - ball.y) < 0.18:
            continue
        clearance = _kick_clearance(ball, target, blockers)
        if clearance < _SHOT_CLEARANCE_M:
            continue
        heading = math.atan2(target.y - ball.y, target.x - ball.x)
        score = min(clearance, 0.60) - 0.14 * abs(target.y) - 0.09 * abs(heading)
        if shooter is not None:
            score -= 0.13 * abs(normalize_angle(heading - shooter.theta))
        if run_in is not None:
            score -= 0.13 * abs(normalize_angle(heading - run_in))
        if best is None or score > best[0]:
            best = score, target
    return best[1] if best is not None else None


def _kick_clearance(ball: BallState, target: Pose2D, blockers: tuple[Obstacle, ...]) -> float:
    """Smallest gap between the ball-to-target segment and an obstacle's edge."""
    dx, dy = target.x - ball.x, target.y - ball.y
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        return -math.inf
    clearance = math.inf
    for obstacle in blockers:
        rx, ry = obstacle.x - ball.x, obstacle.y - ball.y
        progress = (rx * dx + ry * dy) / length_sq
        if progress <= 0.0:
            continue  # The shot moves away from this obstacle.
        progress = min(progress, 1.0)
        gap = math.hypot(rx - progress * dx, ry - progress * dy) - obstacle.radius
        clearance = min(clearance, gap)
    return clearance


def select_clear_or_pass_target(
    config: SoccerConfig,
    field: TeamFieldFrame,
    obstacles: ObstacleCollector,
    player_id: int,
    context: PlayContext,
    is_player_allowed: PlayerAllowed,
) -> Pose2D:
    """Side-lane view: pass if possible, otherwise clear toward the opponent goal without shooting or dribbling.

    Side chasers prefer clearing toward the middle rather than dribbling along the sideline into pressure.
    """

    ball = context.known_ball
    if ball_near_sideline(config, ball):
        return recovery.sideline_recovery_target(config, field, ball)

    teammate = best_pass_target(
        config, obstacles,
        player_id, context, is_player_allowed,
    )
    if teammate is not None:
        return Pose2D(teammate.x, teammate.y, 0.0)
    return Pose2D(field.opponent_goal_x(), 0.0, 0.0)


# Restart touch


def should_make_restart_touch(
    config: SoccerConfig,
    game: GameControlState,
) -> bool:
    """Whether our team should actively touch the ball on restart: kickoff, throw-in, or indirect free kick."""

    return game.is_kickoff_for_team(config.team_id) or (
        game.is_restart_for_team(config.team_id)
        and game.set_play in {SetPlay.THROW_IN, SetPlay.INDIRECT_FREE_KICK}
    )


# Pass scoring


def best_pass_target(
    config: SoccerConfig,
    obstacles: ObstacleCollector,
    player_id: int,
    context: PlayContext,
    is_player_allowed: PlayerAllowed,
) -> Pose2D | None:
    """Select this tick's best legal passing target; return ``None`` when no candidate qualifies.

    Score weights are lane clearance 0.55, forward gain 0.30, center pull 0.15,
    minus distance penalty. Candidates below ``pass_min_score`` are discarded.
    """

    if not config.strategy.pass_enabled:
        return None
    ball = context.known_ball
    game = context.known_game
    candidates: list[tuple[float, Pose2D]] = []
    for teammate_id, robot in context.teammates.items():
        if teammate_id == player_id or robot.pose is None:
            continue
        if not is_player_allowed(game, teammate_id):
            continue
        forward_gain = robot.pose.x - ball.x
        if forward_gain < config.strategy.pass_min_forward_m:
            continue
        lane_score = lane_clear_score(
            config,
            ball.x,
            ball.y,
            robot.pose.x,
            robot.pose.y,
            obstacles.opponent_obstacles(context),
        )
        distance = math.hypot(robot.pose.x - ball.x, robot.pose.y - ball.y)
        field_score = clamp(
            forward_gain / max(1.0, config.field_length),
            0.0,
            1.0,
        )
        center_score = 1.0 - clamp(
            abs(robot.pose.y) / (config.field_width / 2.0),
            0.0,
            1.0,
        )
        score = 0.55 * lane_score + 0.30 * field_score + 0.15 * center_score
        score -= clamp(distance / 12.0, 0.0, 0.25)
        if score >= config.strategy.pass_min_score:
            candidates.append((score, robot.pose))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def shot_lane_is_clear(
    config: SoccerConfig,
    field: TeamFieldFrame,
    obstacles: ObstacleCollector,
    context: PlayContext,
) -> bool:
    """Treat a shot lane as directly shootable when clearance score is at least 0.55."""

    ball = context.known_ball
    return lane_clear_score(
        config,
        ball.x,
        ball.y,
        field.opponent_goal_x(),
        0.0,
        obstacles.opponent_obstacles(context),
    ) >= 0.55


def lane_clear_score(
    config: SoccerConfig,
    start_x: float,
    start_y: float,
    target_x: float,
    target_y: float,
    obstacles: tuple[Obstacle, ...],
) -> float:
    """Generic lane-obstacle score in [0, 1], where 1 is clear and 0 is blocked.

    Project each obstacle onto the lane; if it lies within the segment and inside
    lateral clearance, reduce score proportionally. Multiple obstacles take the worst score.
    """

    if not obstacles:
        return 1.0
    seg_dx = target_x - start_x
    seg_dy = target_y - start_y
    seg_len = math.hypot(seg_dx, seg_dy)
    if seg_len < 1e-6:
        return 0.0
    dir_x = seg_dx / seg_len
    dir_y = seg_dy / seg_len
    left_x = -dir_y
    left_y = dir_x
    score = 1.0
    for obstacle in obstacles:
        rel_x = obstacle.x - start_x
        rel_y = obstacle.y - start_y
        along = rel_x * dir_x + rel_y * dir_y
        if along <= 0.0 or along >= seg_len:
            continue
        lateral = abs(rel_x * left_x + rel_y * left_y)
        clearance = max(config.strategy.pass_lane_clearance, obstacle.radius)
        if lateral < clearance:
            score = min(score, clamp(lateral / clearance, 0.0, 1.0))
    return score


# Dribble and reason text


def dribble_target(
    config: SoccerConfig,
    field: TeamFieldFrame,
    ball: BallState,
) -> Pose2D:
    """Simple dribble target: advance by ``dribble_advance_m`` along ``+x`` and pull ``y`` toward center."""

    target_x = ball.x + config.strategy.dribble_advance_m
    target_y = ball.y * config.strategy.dribble_center_pull
    return field.clamp_inside_field(
        Pose2D(target_x, target_y, field.attack_theta())
    )


def kick_reason(
    config: SoccerConfig,
    target: Pose2D,
    default: str = "chaser kick",
) -> str:
    """Choose the reason suffix from whether target points at the opponent goal mouth.

    When target is near the goal mouth, keep ``default``; otherwise append
    ``" to target"`` so logs distinguish shooting from pass/clear targets.
    """

    if target.x >= config.field_length / 2.0 - 0.2 and abs(target.y) <= config.goal_width / 2.0:
        return default
    return f"{default} to target"
