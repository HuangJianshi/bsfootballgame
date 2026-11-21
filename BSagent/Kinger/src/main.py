"""SoccerSim strategy entry point and primary match logic.

Structure, from high level to low level:
- main.py (this file): match strategy. ``play()`` dispatches through the Phase
  state machine to ``_act_*`` functions. Each function selects an attacker,
  usually the player nearest the ball, and invokes Player actions directly.
- player.py: Player control handle and high-level actions such as ``attack``,
  ``take_kickoff``, ``move_to_position``, and ``walk_to``. Add new skills there.
- utils/: movement, geometry, and obstacle-avoidance helpers such as
  ``opponent_goal``, ``dist``, and ``angle_to``.
- framework/: platform plumbing that users normally do not modify.

To change the playing style, primarily edit the Phase state machine, ``_act_*``
behaviors, and positioning formulas in this file.
"""

from __future__ import annotations

import logging
import math
from enum import Enum

from booster_agent_framework import AgentBase

from .framework.agent import SoccerAgentMixin
from .framework.types import (
    KICKING_TEAM_NONE,
    Context,
    GameState,
    RobotState,
    SetPlay,
)
from .param import (
    ARRIVE_DIST,
    ATTACKER_KEEP_DIST_MARGIN_M,
    ATTACK_FAR_FORWARD_M,
    ATTACK_FAR_WIDTH_M,
    ATTACK_NEAR_SUPPORT_DIST_M,
    BALL_CONTROL_MARGIN_M,
    BALL_OWNED_DIST_M,
    COUNTER_PRESS_MAX_DIST_M,
    COUNTER_PRESS_WINDOW_S,
    DEFEND_BALANCE_BALL_TRACK,
    DEFEND_BALANCE_SETBACK_M,
    DEFEND_BLOCK_DEPTH_M,
    DEFEND_COVER_DIST_M,
    DEFEND_COVER_LATERAL_M,
    DEFEND_ENGAGE_BALL_X_M,
    DEFEND_GOAL_LINE_OFFSET_M,
    DEFEND_LATE_PUSH_BONUS_M,
    DEFEND_MARK_DIST_M,
    DEFEND_MARK_LEAD_M,
    DEFEND_PRESS_DIST_M,
    FALLEN_COST,
    GUARD_THREAT_ENTER_X,
    KICK_POWER_OUR_KICKOFF,
    OMNI_DIST,
    OPP_SET_PLAY_CLEARANCE_M,
    OPP_SET_PLAY_FIELD_MARGIN_M,
    PRESS_BACK_TO_GOAL_COS,
    PRESS_LOOSE_TOUCH_M,
    PRESS_SIDELINE_Y_M,
    ROLE_KEEP_DIST_MARGIN_M,
    SCORE_LATE_REMAINING_S,
    SCORE_LEAD_PRESS_SHIFT_X_M,
    SCORE_TRAIL_PRESS_SHIFT_X_M,
)
from .player import Player
from .utils import angle_to, clamp_inside_field, dist, opponent_goal, own_goal


_log = logging.getLogger(__name__)

_BALL_OURS = "ours"
_BALL_OPPONENTS = "opponents"
_BALL_LOOSE = "loose"


# ======================================================================
# Phase state machine: match phase classification
# ======================================================================


class Phase(Enum):
    """Top-level match phase controlling normal play, restarts, and stops."""
    NORMAL = "normal"              # Normal contest while PLAYING
    OUR_KICKOFF = "our_kickoff"    # Our kickoff during early SET/PLAYING
    OPP_KICKOFF = "opp_kickoff"    # Opponent kickoff; keep clear
    OUR_SET_PLAY = "our_set_play"  # Our free kick, corner, or goal kick
    OPP_SET_PLAY = "opp_set_play"  # Opponent set play; keep clear
    READY = "ready"                # Move into READY positions
    STOPPED = "stopped"            # Non-kickoff SET, INITIAL, FINISHED, or stopped


def get_phase(context: Context) -> Phase:
    """Determine the current phase from the GameController state."""
    g = context.game
    if g is None:
        return Phase.STOPPED

    state = g.state

    # READY: move to ready positions.
    if state == GameState.READY:
        return Phase.READY

    # PLAYING: normal play or an active kickoff/set play.
    if state == GameState.PLAYING and not g.stopped:
        # For a set play, kicking_team identifies the team taking it.
        if g.set_play != SetPlay.NONE and g.kicking_team != KICKING_TEAM_NONE:
            our_team = context.team_id
            if g.kicking_team == our_team:
                return Phase.OUR_SET_PLAY
            else:
                return Phase.OPP_SET_PLAY

        # During the kickoff countdown, kicking_team identifies the taker.
        if g.secondary_time > 0 and g.kicking_team != KICKING_TEAM_NONE:
            our_team = context.team_id
            if g.kicking_team == our_team:
                return Phase.OUR_KICKOFF
            else:
                return Phase.OPP_KICKOFF

        # Normal play.
        return Phase.NORMAL

    # Remain stationary during SET, INITIAL, FINISHED, or a stoppage.
    return Phase.STOPPED

def get_set_play_type(context: Context) -> SetPlay:
    """Return the active set-play type, or ``SetPlay.NONE`` when unavailable.

    This reads the GameController's ``set_play`` field without identifying the
    taking team. :func:`get_phase` distinguishes OUR_SET_PLAY from OPP_SET_PLAY;
    this function only identifies the type.

    The seven possible values from framework.types.SetPlay are:
    - ``NONE``: no set play, such as normal play or a kickoff
    - ``DIRECT_FREE_KICK``: a direct free kick that may score immediately
    - ``INDIRECT_FREE_KICK``: another player must touch the ball before a goal
    - ``PENALTY_KICK``: a penalty kick
    - ``THROW_IN``: a kick-in from touch
    - ``GOAL_KICK``: a goal kick
    - ``CORNER_KICK``: a corner kick
    """
    g = context.game
    if g is None:
        return SetPlay.NONE
    return g.set_play


# ======================================================================
# Agent entry point
# ======================================================================


class SoccerSimAgent(SoccerAgentMixin, AgentBase):
    """3v3 SoccerSim agent."""

    player_class = Player

    def init_store(self, store) -> None:
        _log.info("init_store called")
        store.prev_phase = None       # Previous phase, used to detect transitions
        store.cur_phase = None
        store.kickoff_taker = None    # Locked taker ID, reselected for each kickoff
        store.normal_attacker = None
        store.attack_near = None
        store.defense_presser = None
        store.defense_cover = None
        store.defense_balance = None
        store.ball_owner = _BALL_LOOSE
        store.loss_at = -math.inf
        store.loss_player_id = None

    @staticmethod
    def play(context: Context, players: list[Player], store) -> None:
        phase = get_phase(context)
        store.prev_phase = store.cur_phase
        store.cur_phase = phase

        # Draw per-frame visualizations.
        _analyze_and_draw(context, players, store)

        # Draw the current phase as a label just outside the field.
        from .framework import debugdraw
        g = context.game
        game_state = g.state.value if g is not None else "none"
        set_play = g.set_play.value if g is not None else "none"
        secondary_time = g.secondary_time if g is not None else 0.0
        debugdraw.text(
            0.0, context.field.width / 2.0 + 0.2,
            f"phase={phase.value} state={game_state} set={set_play} secondary={secondary_time:.1f}",
            rgb=(1.0, 1.0, 0.0), ns="phase",
        )

        # Handle readiness and retain only players that can act this frame.
        # ensure_ready asynchronously gets up or switches to walk mode without
        # moving. Penalized players also prepare so they can rejoin immediately.
        # Penalized or unready players are excluded from role assignment to avoid
        # selecting an attacker who cannot move.
        active: list[Player] = []
        for p in players:
            # Direct shots are allowed by default; set plays may disable them.
            p.allow_direct_shot = True
            ready = p.ensure_ready()
            if p.is_penalized:
                p.action = "penalized"     # May prepare, but cannot move
                p.stop()
            elif not ready:
                p.action = "fallen" if p.is_fallen else "switching_mode"
            elif p.pose is None:
                p.action = "no_pose"       # Exclude players with unknown positions
                p.stop()
            else:
                active.append(p)

        # Dispatch the team once per phase; _act_* owns team-wide role assignment.
        if phase == Phase.NORMAL:
            _act_normal(context, active, store)
        elif phase == Phase.OUR_KICKOFF:
            _clear_normal_sticky(store)
            _act_our_kickoff(context, active, store)
        elif phase == Phase.OPP_KICKOFF:
            _clear_normal_sticky(store)
            _act_opp_kickoff(context, active)
        elif phase == Phase.OUR_SET_PLAY:
            _clear_normal_sticky(store)
            _act_our_set_play(context, active, store)
        elif phase == Phase.OPP_SET_PLAY:
            _clear_normal_sticky(store)
            _act_opp_set_play(context, active, store)
        elif phase == Phase.READY:
            _clear_normal_sticky(store)
            _act_ready(context, active)
        elif phase == Phase.STOPPED:
            _clear_normal_sticky(store)
            for p in active:
                p.action = "stopped"
                p.stop()

        # Draw every teammate last, including penalized, unready, and stopped
        # players, so markers remain visible during states such as SET.
        for p in players:
            _draw_teammate_marker(p)


def _clear_normal_sticky(store) -> None:
    store.normal_attacker = None
    store.attack_near = None
    store.defense_presser = None
    store.defense_cover = None
    store.defense_balance = None
    store.ball_owner = _BALL_LOOSE
    store.loss_at = -math.inf
    store.loss_player_id = None


def _player_dist_to_position(p: Player, target: tuple[float, float]) -> float:
    """Return infinity when the player's position is unknown."""
    pose = p.pose
    if pose is None:
        return math.inf
    return dist(pose.x, pose.y, *target)


def _player_dist_to_ball(context: Context, p: Player) -> float:
    """Return a player's distance to the ball's current position."""
    ball = context.ball
    return (
        _player_dist_to_position(p, (ball.x, ball.y)) + _fallen_time_cost(p)
        if ball is not None else math.inf
    )


def _fallen_time_cost(p: Player) -> float:
    return FALLEN_COST if p.is_fallen else 0.0


def _select_closest_attacker(
    context: Context,
    players: list[Player],
    preferred_id: int | None = None,
) -> Player:
    """Select the player with the lowest effective distance to the ball.

    ``players`` must be nonempty, ready, and have known poses. Normal play and
    kickoffs share this selection logic.
    """
    return _select_closest_to_ball(
        context, players, preferred_id, ATTACKER_KEEP_DIST_MARGIN_M,
    )


def _select_closest_to_ball(
    context: Context,
    players: list[Player],
    preferred_id: int | None = None,
    keep_margin_m: float = ROLE_KEEP_DIST_MARGIN_M,
) -> Player:
    """Select the nearest player to the ball with role-switch hysteresis."""
    ranked = [(p, _player_dist_to_ball(context, p)) for p in players]
    best, best_dist = min(ranked, key=lambda item: item[1])
    preferred = next((item for item in ranked if item[0].id == preferred_id), None)
    if preferred is not None and preferred[1] <= best_dist + keep_margin_m:
        return preferred[0]
    return best


def _select_closest_to_position(
    players: list[Player],
    target: tuple[float, float],
    preferred_id: int | None = None,
) -> Player:
    """Select the nearest player to a team target with role hysteresis."""
    ranked = [(_player_dist_to_position(p, target), p) for p in players]
    best_dist, best = min(ranked, key=lambda item: item[0])
    preferred = next(
        (item for item in ranked if item[1].id == preferred_id), None,
    )
    if (
        preferred is not None
        and preferred[0] <= best_dist + ROLE_KEEP_DIST_MARGIN_M
    ):
        return preferred[1]
    return best


def _nearest_opponent_to_ball(
    context: Context,
) -> tuple[RobotState | None, float]:
    """Return the closest visible opponent and its distance to the ball."""
    ball = context.ball
    if ball is None:
        return None, math.inf
    best: RobotState | None = None
    best_dist = math.inf
    for opponent in context.opponents.values():
        pose = opponent.pose
        if pose is None:
            continue
        d = dist(pose.x, pose.y, ball.x, ball.y)
        if d < best_dist:
            best, best_dist = opponent, d
    return best, best_dist


def _dangerous_runner(
    context: Context, ball,
) -> RobotState | None:
    """Return the most goal-side opponent when it is ahead of the ball threat."""
    best: RobotState | None = None
    best_x = math.inf
    for opponent in context.opponents.values():
        pose = opponent.pose
        if pose is None:
            continue
        if pose.x > ball.x - DEFEND_MARK_LEAD_M:
            continue
        if pose.x < best_x:
            best, best_x = opponent, pose.x
    return best


def _classify_ball_owner(context: Context, players: list[Player]) -> str:
    """Classify clear possession, otherwise treat the ball as contested."""
    ball = context.ball
    if ball is None or not players:
        return _BALL_LOOSE

    our_player = _select_closest_attacker(context, players, None)
    our_dist = _player_dist_to_position(our_player, (ball.x, ball.y))
    _, opponent_dist = _nearest_opponent_to_ball(context)

    if (
        our_dist <= BALL_OWNED_DIST_M
        and our_dist + BALL_CONTROL_MARGIN_M <= opponent_dist
    ):
        return _BALL_OURS
    if (
        opponent_dist <= BALL_OWNED_DIST_M
        and opponent_dist + BALL_CONTROL_MARGIN_M <= our_dist
    ):
        return _BALL_OPPONENTS
    return _BALL_LOOSE


def _update_ball_owner(context: Context, owner: str, store) -> None:
    """Track possession changes for the short counter-press window."""
    previous = getattr(store, "ball_owner", _BALL_LOOSE)
    if previous == _BALL_OURS and owner != _BALL_OURS:
        store.loss_at = context.now
        store.loss_player_id = getattr(store, "normal_attacker", None)
    if owner == _BALL_OURS:
        store.loss_at = -math.inf
        store.loss_player_id = None
    store.ball_owner = owner


def _team_score(context: Context, team_id: int) -> int | None:
    game = context.game
    if game is None:
        return None
    team = game.get_team_state(team_id)
    return None if team is None else team.score


def _score_difference(context: Context) -> int:
    """Return our score minus the opponent score."""
    game = context.game
    if game is None:
        return 0
    ours = _team_score(context, context.team_id)
    theirs = None
    for team in game.teams:
        if team.team_number != context.team_id:
            theirs = team.score
            break
    if ours is None or theirs is None:
        return 0
    return ours - theirs


def _press_engage_line(context: Context) -> float:
    """Return the x line below which the default midfield block may engage."""
    line = DEFEND_ENGAGE_BALL_X_M
    difference = _score_difference(context)
    if difference > 0:
        line += SCORE_LEAD_PRESS_SHIFT_X_M
    elif difference < 0:
        line += SCORE_TRAIL_PRESS_SHIFT_X_M
        game = context.game
        if (
            game is not None
            and 0 < game.secs_remaining <= SCORE_LATE_REMAINING_S
        ):
            line += DEFEND_LATE_PUSH_BONUS_M
    return line


def _counter_presser(
    context: Context, players: list[Player], store,
) -> Player | None:
    """Return the player who may counter-press immediately after a loss."""
    if len(players) < 2:
        return None
    ball = context.ball
    lost_at = getattr(store, "loss_at", -math.inf)
    if ball is None or context.now - lost_at > COUNTER_PRESS_WINDOW_S:
        return None
    player_id = getattr(store, "loss_player_id", None)
    player = next((p for p in players if p.id == player_id), None)
    if player is None:
        return None
    if _player_dist_to_position(player, (ball.x, ball.y)) > COUNTER_PRESS_MAX_DIST_M:
        return None
    return player


def _opponent_facing_away(context: Context) -> bool:
    """Return whether the nearest opponent has its back to our goal."""
    opponent, _ = _nearest_opponent_to_ball(context)
    if opponent is None:
        return False
    pose = opponent.pose
    if pose is None:
        return False
    gx, gy = own_goal(context)
    toward_goal = angle_to(pose.x, pose.y, gx, gy)
    return math.cos(pose.theta - toward_goal) <= PRESS_BACK_TO_GOAL_COS


def _press_triggered(context: Context) -> bool:
    """Return whether a specific cue justifies stepping up for the ball."""
    ball = context.ball
    if ball is None:
        return False
    _, opponent_dist = _nearest_opponent_to_ball(context)
    loose_touch = opponent_dist >= PRESS_LOOSE_TOUCH_M
    sideline_trap = abs(ball.y) >= PRESS_SIDELINE_Y_M
    return loose_touch or sideline_trap or _opponent_facing_away(context)


def _midblock_position(
    context: Context, ball,
) -> tuple[float, float]:
    """Hold a compact midfield slot when the opponent ball is still high."""
    target_x = min(ball.x - DEFEND_BLOCK_DEPTH_M, 0.5)
    if _score_difference(context) > 0:
        target_x -= 0.6
    target_x = max(target_x, -context.field.length / 2.0 + 2.0)
    target_y = ball.y * 0.35
    return clamp_inside_field(context, target_x, target_y, margin=0.7)


def _cover_position(
    context: Context,
    anchor: tuple[float, float],
    ball,
) -> tuple[float, float]:
    """Return a central cover slot behind the first defender."""
    ax, ay = anchor
    gx, gy = own_goal(context)
    dx, dy = gx - ax, gy - ay
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return clamp_inside_field(context, gx, gy, margin=0.7)
    ux, uy = dx / d, dy / d
    along = min(DEFEND_COVER_DIST_M, d)
    tx = ax + ux * along
    ty = ay + uy * along

    # Shade toward the centre without collapsing onto the first defender.
    if abs(ty) > 1e-6:
        ty -= math.copysign(
            min(abs(ty), DEFEND_COVER_LATERAL_M), ty,
        )
    return clamp_inside_field(context, tx, ty, margin=0.7)


def _mark_position(
    context: Context, opponent: RobotState,
) -> tuple[float, float]:
    """Stand goal-side of a dangerous runner and shade toward the centre."""
    pose = opponent.pose
    if pose is None:
        return clamp_inside_field(context, *own_goal(context), margin=0.6)
    gx, gy = own_goal(context)
    dx, dy = gx - pose.x, gy - pose.y
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return clamp_inside_field(context, gx, gy, margin=0.6)
    ux, uy = dx / d, dy / d
    tx = pose.x + ux * DEFEND_MARK_DIST_M
    ty = pose.y + uy * DEFEND_MARK_DIST_M
    if abs(ty) > 1e-6:
        ty -= math.copysign(min(abs(ty), DEFEND_COVER_LATERAL_M), ty)
    return clamp_inside_field(context, tx, ty, margin=0.6)


def _balance_position(
    context: Context,
    anchor: tuple[float, float],
    ball,
) -> tuple[float, float]:
    """Protect the far post and the space behind the central cover."""
    if ball.x <= GUARD_THREAT_ENTER_X:
        gx, _ = own_goal(context)
        goal_half = context.field.goal_width * 0.5
        target_y = ball.y * DEFEND_BALANCE_BALL_TRACK
        target_y = max(-goal_half * 0.72, min(goal_half * 0.72, target_y))
        return clamp_inside_field(
            context,
            gx + DEFEND_GOAL_LINE_OFFSET_M,
            target_y,
            margin=0.6,
        )

    ax, _ = anchor
    target_x = ax - DEFEND_BALANCE_SETBACK_M
    min_x = -context.field.length / 2.0 + 1.0
    target_x = max(target_x, min_x)

    ball_side = 1.0 if ball.y >= 0.0 else -1.0
    if abs(ball.y) < 0.3:
        ball_side = 1.0 if anchor[1] >= 0.0 else -1.0
    weak_side = -ball_side
    target_y = weak_side * (context.field.goal_width * 0.5 + 0.35)
    return clamp_inside_field(context, target_x, target_y, margin=0.7)


def _attack_near_position(
    context: Context, ball,
) -> tuple[float, float]:
    """Short outlet behind the ball and toward the central lane."""
    target_x = ball.x - ATTACK_NEAR_SUPPORT_DIST_M
    target_y = ball.y * 0.55
    return clamp_inside_field(context, target_x, target_y, margin=0.8)


def _attack_far_position(
    context: Context, ball, far_player: Player,
) -> tuple[float, float]:
    """Weak-side run beyond the ball, attacking the far post lane."""
    target_x = ball.x + ATTACK_FAR_FORWARD_M
    ball_side = 1.0 if ball.y >= 0.0 else -1.0
    if abs(ball.y) < 0.3:
        pose = far_player.pose
        ball_side = 1.0 if pose is not None and pose.y >= 0.0 else -1.0
    target_y = -ball_side * ATTACK_FAR_WIDTH_M
    return clamp_inside_field(context, target_x, target_y, margin=0.8)


def _act_normal(
    context: Context,
    players: list[Player],
    store,
    *,
    force_attack: bool = False,
) -> None:
    """Dispatch normal play by possession: attack, defend, or contest."""
    if not players:
        return

    ball = context.ball
    if ball is None:
        _act_no_ball_shape(context, players)
        return

    owner = _classify_ball_owner(context, players)
    _update_ball_owner(context, owner, store)
    if owner == _BALL_OPPONENTS and not force_attack:
        _act_defensive_triangle(context, players, ball, store)
    elif owner == _BALL_OURS or force_attack:
        _act_attacking_triangle(context, players, ball, store)
    else:
        _act_contested_triangle(context, players, ball, store)


def _act_no_ball_shape(context: Context, players: list[Player]) -> None:
    """Keep a compact triangle while the ball observation is unavailable."""
    gx, gy = own_goal(context)
    guard = min(players, key=lambda p: _player_dist_to_position(p, (gx, gy)))
    guard.guard()

    rest = [p for p in players if p is not guard]
    slots = [(-1.0, 2.0), (-1.4, -2.0)]
    for p, target in zip(rest, slots):
        p.action = "shape:hold"
        p.move_to_position(clamp_inside_field(context, *target, margin=0.8))


def _act_attacking_triangle(
    context: Context, players: list[Player], ball, store,
) -> None:
    """Keep one on-ball player, one protected outlet, and one far threat."""
    attacker = _select_closest_attacker(
        context, players, getattr(store, "normal_attacker", None),
    )
    store.normal_attacker = attacker.id
    attacker.action = "attack:on_ball"
    attacker.attack()

    rest = [p for p in players if p is not attacker]
    if not rest:
        return

    near = _select_closest_to_ball(
        context,
        rest,
        getattr(store, "attack_near", None),
        ROLE_KEEP_DIST_MARGIN_M,
    )
    store.attack_near = near.id
    near.action = "attack:near"
    near.move_to_position(_attack_near_position(context, ball))

    remaining = [p for p in rest if p is not near]
    for far in remaining:
        far.action = "attack:far"
        far.move_to_position(_attack_far_position(context, ball, far))


def _act_contested_triangle(
    context: Context, players: list[Player], ball, store,
) -> None:
    """Win a loose ball without abandoning the goal-side structure."""
    challenger = _select_closest_to_ball(
        context,
        players,
        getattr(store, "normal_attacker", None),
        ATTACKER_KEEP_DIST_MARGIN_M,
    )
    store.normal_attacker = challenger.id
    challenger.action = "contest:ball"
    challenger.attack()

    rest = [p for p in players if p is not challenger]
    if not rest:
        return

    anchor = (ball.x, ball.y)
    dangerous = _dangerous_runner(context, ball)
    if dangerous is not None:
        cover_target = _mark_position(context, dangerous)
    else:
        cover_target = _cover_position(context, anchor, ball)
    cover = _select_closest_to_position(
        rest,
        cover_target,
        getattr(store, "defense_cover", None),
    )
    store.defense_cover = cover.id
    cover.action = "contest:cover"
    cover.move_to_position(cover_target)

    rest = [p for p in rest if p is not cover]
    if not rest:
        return

    balance_target = _balance_position(context, cover_target, ball)
    balance = _select_closest_to_position(
        rest,
        balance_target,
        getattr(store, "defense_balance", None),
    )
    store.defense_balance = balance.id
    balance.action = "contest:balance"
    balance.move_to_position(balance_target)


def _act_defensive_triangle(
    context: Context, players: list[Player], ball, store,
) -> None:
    """Assign first defender, central cover, and far-side balance."""
    if ball.x < 0.0:
        gx, gy = own_goal(context)
        goal_defender = _select_closest_to_position(
            players,
            (gx, gy),
            getattr(store, "defense_balance", None),
        )
        store.defense_balance = goal_defender.id
        goal_defender.action = "defend:goal"
        goal_defender.guard()

        challengers = [p for p in players if p is not goal_defender]
        if not challengers:
            return

        presser = _select_closest_to_ball(
            context,
            challengers,
            getattr(store, "defense_presser", None),
            ROLE_KEEP_DIST_MARGIN_M,
        )
        store.defense_presser = presser.id
        presser.action = "defend:press"
        presser.press()

        challengers = [p for p in challengers if p is not presser]
        if not challengers:
            return

        blocker = _select_closest_to_ball(
            context,
            challengers,
            getattr(store, "defense_cover", None),
            ROLE_KEEP_DIST_MARGIN_M,
        )
        store.defense_cover = blocker.id
        blocker.action = "defend:block_press"
        blocker.press()
        return

    counter_presser = _counter_presser(context, players, store)
    if counter_presser is not None:
        presser = counter_presser
    else:
        presser = _select_closest_to_ball(
            context,
            players,
            getattr(store, "defense_presser", None),
            ROLE_KEEP_DIST_MARGIN_M,
        )
    store.defense_presser = presser.id

    engage = (
        counter_presser is not None
        or ball.x <= _press_engage_line(context)
        or _press_triggered(context)
        or _player_dist_to_position(presser, (ball.x, ball.y))
        <= DEFEND_PRESS_DIST_M
    )
    if engage:
        presser.action = (
            "defend:counter" if counter_presser is not None
            else "defend:press"
        )
        presser.press()
        anchor = (ball.x, ball.y)
    else:
        anchor = _midblock_position(context, ball)
        presser.action = "defend:block"
        presser.move_to_position(anchor)

    rest = [p for p in players if p is not presser]
    if not rest:
        return

    dangerous = _dangerous_runner(context, ball)
    if dangerous is not None:
        cover_target = _mark_position(context, dangerous)
    else:
        cover_target = _cover_position(context, anchor, ball)
    cover = _select_closest_to_position(
        rest,
        cover_target,
        getattr(store, "defense_cover", None),
    )
    store.defense_cover = cover.id
    cover.action = "defend:cover"
    cover.move_to_position(cover_target)

    rest = [p for p in rest if p is not cover]
    if not rest:
        return

    balance_target = _balance_position(context, cover_target, ball)
    balance = _select_closest_to_position(
        rest,
        balance_target,
        getattr(store, "defense_balance", None),
    )
    store.defense_balance = balance.id
    balance.action = "defend:balance"
    balance.move_to_position(balance_target)


def _act_our_kickoff(context: Context, players: list[Player], store) -> None:
    """Lock the nearest kickoff taker, assign a guard, and leave others in support."""
    if not players:
        return

    active_ids = {p.id for p in players}
    if store.prev_phase != Phase.OUR_KICKOFF or store.kickoff_taker not in active_ids:
        # Select a new taker when entering the kickoff phase.
        store.kickoff_taker = _select_closest_attacker(context, players).id

    attacker_id = store.kickoff_taker
    attacker = next((p for p in players if p.id == attacker_id), None)
    if attacker is None:
        return

    attacker.action = "kickoff"
    attacker.kick(0.1, KICK_POWER_OUR_KICKOFF)

    rest = [p for p in players if p is not attacker]
    if rest:
        gx, gy = own_goal(context)
        guard = min(rest, key=lambda p: _player_dist_to_position(p, (gx, gy)))
        guard.guard()
        rest = [p for p in rest if p is not guard]

    for p in rest:
        p.action = "stay"
        p.stop()


def _act_opp_kickoff(context: Context, players: list[Player]) -> None:
    """Guard with one player while the others wait outside the center circle."""
    if not players:
        return
    gx, gy = own_goal(context)
    guard = min(players, key=lambda p: _player_dist_to_position(p, (gx, gy)))
    guard.guard()

    rest = [p for p in players if p is not guard]
    r = context.field.circle_radius
    slots = [(-r - 0.5, 0.0), (-r - 2.0, 0.5)]
    for p, target in zip(rest, slots):
        p.action = "opp_kickoff:ready"
        p.walk_to(target, avoid_ball=True, avoid_robots=True)


def _act_our_set_play(context: Context, players: list[Player], store) -> None:
    """Dispatch our set play by type.

    Set plays that cannot score directly (indirect free kick, throw-in, goal
    kick) have their direct shot disabled, so the attacker plays a forward pass
    instead. Direct free kicks, corners, and penalties keep the aggressive shot.
    """
    set_play = get_set_play_type(context)
    if set_play in (
        SetPlay.INDIRECT_FREE_KICK,
        SetPlay.THROW_IN,
        SetPlay.GOAL_KICK,
    ):
        for p in players:
            p.allow_direct_shot = False
    _act_normal(context, players, store, force_attack=True)


def _act_opp_set_play(context: Context, players: list[Player], store) -> None:
    """Retreat directly from the ball; wait until the official phase changes."""
    ball = context.ball
    # Account for walk_to's arrival tolerance when targeting the 1.5 m boundary.
    radius = OPP_SET_PLAY_CLEARANCE_M + ARRIVE_DIST
    half_l = context.field.length / 2.0 - OPP_SET_PLAY_FIELD_MARGIN_M
    half_w = context.field.width / 2.0 - OPP_SET_PLAY_FIELD_MARGIN_M
    for p in players:
        pose = p.pose
        p.action = "opp_set:wait"
        if ball is None or pose is None:
            p.stop()
            continue
        if dist(pose.x, pose.y, ball.x, ball.y) >= OPP_SET_PLAY_CLEARANCE_M:
            p.stop()
            continue

        # The radial exit is shortest. At the field edge, also consider the
        # retreat circle's intersections with the in-field boundary.
        outward = angle_to(ball.x, ball.y, pose.x, pose.y)
        candidates = [(
            ball.x + math.cos(outward) * radius,
            ball.y + math.sin(outward) * radius,
        )]
        for x in (-half_l, half_l):
            remaining = radius ** 2 - (x - ball.x) ** 2
            if remaining >= 0.0:
                offset = math.sqrt(remaining)
                candidates.extend((x, ball.y + sign * offset) for sign in (-1.0, 1.0))
        for y in (-half_w, half_w):
            remaining = radius ** 2 - (y - ball.y) ** 2
            if remaining >= 0.0:
                offset = math.sqrt(remaining)
                candidates.extend((ball.x + sign * offset, y) for sign in (-1.0, 1.0))
        candidates = [
            t for t in candidates
            if t == clamp_inside_field(context, *t, margin=OPP_SET_PLAY_FIELD_MARGIN_M)
        ]
        if not candidates:
            p.stop()
            continue

        px, py = pose.x, pose.y
        target = min(candidates, key=lambda t: dist(px, py, *t))
        # Short omnidirectional steps follow the retreat line directly.
        # Do not invoke path avoidance: the movement is away from the ball.
        length = dist(pose.x, pose.y, *target)
        ratio = min(1.0, OMNI_DIST / (2.0 * length))
        step = (
            pose.x + (target[0] - pose.x) * ratio,
            pose.y + (target[1] - pose.y) * ratio,
        )
        p.action = "opp_set:retreat"
        p.walk_to(step, face=angle_to(pose.x, pose.y, ball.x, ball.y))


def _act_ready(context: Context, players: list[Player]) -> None:
    """Move each player to a ready position."""
    game = context.game
    our_kickoff = game is not None and game.kicking_team == context.team_id
    field = context.field
    # Our kickoff priority: center-circle edge, goal-area center, then left side.
    if our_kickoff:
        if len(players) >= 1:
            p1 = players[0]
            p1.action = "ready"
            p1.walk_to(
                (-field.circle_radius, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
        if len(players) >= 2:
            p2 = players[1]
            p2.action = "ready"
            p2.walk_to(
                (-field.length / 2.0 + field.goal_area_length, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
        if len(players) >= 3:
            p3 = players[2]
            p3.action = "ready"
            p3.walk_to(
                (-0.5, field.circle_radius + 2),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
    # Opponent kickoff priority: outside center circle, goal area, then penalty area.
    else:
        if len(players) >= 1:
            p1 = players[0]
            p1.action = "ready"
            p1.walk_to(
                (-field.circle_radius - 0.5, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
        if len(players) >= 2:
            p2 = players[1]
            p2.action = "ready"
            p2.walk_to(
                (-field.length / 2.0 + field.goal_area_length, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
        if len(players) >= 3:
            p3 = players[2]
            p3.action = "ready"
            p3.walk_to(
                (-field.length / 2.0 + field.penalty_area_length, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )



# ======================================================================
# Match visualization: ball position and each player's distance to the ball
# ======================================================================

def _draw_teammate_marker(p: Player) -> None:
    """Draw teammates in red, using cubes while kicking and spheres otherwise.

    This runs for every player each frame regardless of phase, penalty, or
    readiness. The label contains the player ID and current high-level action
    (``p.action``), with ``[KICK]`` appended while kicking. Shape also indicates
    whether the player is in the kicking state.
    """
    from .framework import debugdraw

    pose = p.pose
    if pose is None:
        return
    red = (1.0, 0.2, 0.2)
    if p.is_kicking:
        debugdraw.cube(pose.x, pose.y, rgb=red, scale=0.38, ns="teammate")
    else:
        debugdraw.point(pose.x, pose.y, rgb=red, scale=0.3, ns="teammate")
    kick_tag = " [KICK]" if p.is_kicking else ""
    label = f"{p.id}:{p.action}{kick_tag}"
    debugdraw.text(pose.x, pose.y, label, rgb=(1.0, 0.9, 0.6), ns="teammate_id")


def _analyze_and_draw(context: Context, players: list[Player], store) -> None:
    """Calculate player-to-ball distances and draw them each frame.

    This no longer depends on the analysis module; distances use the ball's
    current position.
    """
    from .framework import debugdraw

    ball = context.ball

    # Nothing can be drawn when the ball is not visible.
    if ball is None:
        return

    # 1. Draw the ball's current position as a green point.
    debugdraw.point(ball.x, ball.y, rgb=(0.0, 1.0, 0.0), scale=0.2, ns="ball_current")

    # 2. Draw distance labels: red for teammates and blue for opponents.
    for p in players:
        pose = p.pose
        if pose is None:
            continue
        d = dist(pose.x, pose.y, ball.x, ball.y)
        debugdraw.text(
            pose.x + 0.3, pose.y - 0.3, f"{d:.1f}m",
            rgb=(1.0, 0.6, 0.6), ns="dist_ours",
        )
    for r in context.opponents.values():
        pose = r.pose
        if pose is None:
            continue
        d = dist(pose.x, pose.y, ball.x, ball.y)
        debugdraw.text(
            pose.x + 0.3, pose.y - 0.3, f"{d:.1f}m",
            rgb=(0.6, 0.6, 1.0), ns="dist_opp",
        )
