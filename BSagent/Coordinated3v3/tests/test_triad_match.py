"""Run the production behavior tree against deterministic synthetic match frames."""

import math
import unittest
from random import Random

import py_trees

from src.behavior_tree import TeamStrategyTree
from src.behavior_tree.blackboard import BlackboardClient, BlackboardKeys
from src.play import PLAYBOOKS
from src.play.ball_motion import BallMotion
from src.play.triad_tactics import keeper_intercept_y, restart_touch_target
from src.runtime import SoccerKit
from src.soccer_framework import (
    BallState, GameControlState, GamePhase, GameState, KickIntent, MoveIntent, Penalty,
    PlayContext, Pose2D, RobotRuntimeStatus, RobotState, SetPlay, SoccerConfig, StopIntent,
)


class MatchTests(unittest.TestCase):
    def side(self, side=1):
        py_trees.blackboard.Blackboard.clear()
        self.kit = SoccerKit(SoccerConfig.from_env({"SOCCER_TEAM_ID": str(side)}))
        self.tree = TeamStrategyTree(self.kit, PLAYBOOKS.create_default(self.kit), self)
        self.kit.bind_robot_services(self)
        self.board = BlackboardClient(name="test")
        self.now = 10.0
        self.fallen = set()
        self.recoveries = []

    def get_snapshot(self):
        return self.snapshot

    def poll_runtime_status(self, pid, now):
        return RobotRuntimeStatus("walk", "fallen" if pid in self.fallen else "normal", True, now)

    def trigger_get_up(self, pid, now):
        self.recoveries.append(pid)
        return True

    def frame(self, *, state=GameState.PLAYING, phase=GamePhase.NORMAL, stopped=False, owner=255,
              set_play=SetPlay.NONE, ball=(0.8, 0.0), poses=((-0.8, 0.0), (-2.0, 1.8), (-6.0, 0.0)),
              penalized=(), stale=(), game_age=0.0):
        self.now += 1 / 30
        game = GameControlState(
            state=state, game_phase=phase, stopped=stopped, set_play=set_play, kicking_team=owner,
            secondary_time=8 if owner != 255 else 0, last_seen_at=self.now - game_age,
        )
        team = game.get_team_state(self.kit.config.team_id)
        for pid in penalized:
            team.players[pid - 1].penalty = Penalty.ILLEGAL_POSITIONING
        self.snapshot = PlayContext(
            game_state=game,
            teammates={
                pid: RobotState(pid, Pose2D(*pos), self.now - (0.8 if pid in stale else 0.0))
                for pid, pos in enumerate(poses, 1) if pos is not None
            },
            opponents={
                pid: RobotState(pid, Pose2D(*pos), self.now)
                for pid, pos in enumerate(((3.5, 1.3), (4.3, -1.1), (6.3, 0.2)), 1)
            },
            ball=BallState(*ball, last_seen_at=self.now) if ball is not None else None,
        )
        self.tree.tick(now=self.now)
        commands = self.tree.last_commands
        self.assertEqual(set(commands), {1, 2, 3})
        return commands

    def role(self, name):
        return self.board.read(BlackboardKeys.ROLES).players_of(name)

    def test_guard_stops_movement_and_get_up(self):
        self.side()
        self.fallen = {1}
        for case in (
            {"state": GameState.SET}, {"state": GameState.READY, "stopped": True},
            {"phase": GamePhase.TIMEOUT}, {"state": GameState.FINISHED},
            {"ball": None}, {"ball": (math.nan, 0.0)}, {"game_age": 3.0},
        ):
            with self.subTest(case=case):
                commands = self.frame(**case)
                self.assertTrue(all(isinstance(c.intent, StopIntent) for c in commands.values()))
        self.frame(penalized=(1,))
        self.assertEqual(self.recoveries, [])
        self.frame()
        self.assertEqual(self.recoveries, [1])

    def test_opponent_restart_never_kicks(self):
        for side in (1, 2):
            self.side(side)
            for kind in SetPlay:
                commands = self.frame(owner=3 - side, set_play=kind, ball=(0.0, 0.0),
                                      poses=((-0.22, 0.0), (-1.5, 1.8), (-6.1, 0.0)))
                self.assertFalse(any(isinstance(c.intent, KickIntent) for c in commands.values()))

    def test_kickoff_holds_other_players_and_replaces_penalized_taker(self):
        for side in (1, 2):
            self.side(side)
            commands = self.frame(owner=side, ball=(0.0, 0.0))
            self.assertEqual(self.role("chaser"), (1,))
            self.assertTrue(all(isinstance(commands[pid].intent, StopIntent) for pid in (2, 3)))
            self.frame(owner=side, ball=(0.0, 0.0), poses=((-1.5, 0), (-0.24, 0), (-6, 0)))
            self.assertEqual(self.role("chaser"), (1,))
            commands = self.frame(owner=side, ball=(0.0, 0.0), penalized=(1,))
            self.assertEqual(self.role("chaser"), (2,))
            self.assertTrue(all(isinstance(commands[pid].intent, StopIntent) for pid in (1, 3)))

    def test_first_touch_avoids_both_goal_mouths(self):
        self.side()
        for kind, ball in (
            (SetPlay.NONE, (0.0, 0.0)), (SetPlay.INDIRECT_FREE_KICK, (4.0, 0.25)),
            (SetPlay.THROW_IN, (3.0, 4.49)), (SetPlay.GOAL_KICK, (-6.8, 2.1)),
        ):
            self.frame(ball=ball, set_play=kind, owner=1)
            aim = restart_touch_target(self.kit, 1, self.snapshot)
            self.assertIsNotNone(aim)
            dx, dy = aim.x - ball[0], aim.y - ball[1]
            self.assertGreaterEqual(math.hypot(dx, dy), 0.6)
            self.assertLess(abs(aim.x), 7)
            self.assertLess(abs(aim.y), 4.5)
            for goal in (-7, 7):
                if dx and (goal - ball[0]) / dx > 0:
                    self.assertGreaterEqual(abs(ball[1] + dy * (goal - ball[0]) / dx), 1.3)

    def test_reassigns_keeper_on_stale_pose_and_accepts_fresh_return(self):
        self.side()
        poses = ((-5.2, -0.2), (-1.2, 0), (-6.0, 0))
        self.frame(poses=poses, stale=(3,))
        self.assertEqual(self.role("goalkeeper"), (1,))
        self.assertEqual(self.role("chaser"), (2,))
        self.frame(poses=poses)
        self.assertEqual(self.role("goalkeeper"), (3,))

    def test_sole_keeper_sweeps_near_threats_but_stays_goal_side(self):
        self.side()
        poses = (None, None, (-6.15, 0.0))
        close = self.frame(poses=poses, ball=(-6.0, 0.0))
        far = self.frame(poses=poses, ball=(3.5, 0.0))
        self.assertIsInstance(close[3].intent, KickIntent)
        self.assertNotIsInstance(far[3].intent, KickIntent)

    def test_approach_lines_up_behind_lateral_touch(self):
        self.side()
        stance = self.kit.motion.approach_target(BallState(0.5, 0.5), math.pi / 2)
        self.assertAlmostEqual(stance.x, 0.5)
        self.assertAlmostEqual(stance.y, 0.1)

    def test_predicts_only_shots_that_can_reach_goal(self):
        ball = BallState(-4.2, 0.8)
        self.assertEqual(keeper_intercept_y(ball, BallMotion(3, -1, 0, 0), -6.18, -7, 2.6), 0.8)
        self.assertIsNone(keeper_intercept_y(ball, BallMotion(2, 1, 0, 0), -6.18, -7, 2.6))
        self.assertIsNone(keeper_intercept_y(ball, BallMotion(1, -1, 0, 2), -6.18, -7, 2.6))
        self.assertIsNone(keeper_intercept_y(BallState(-4.2, 3), BallMotion(3, -1, 0, 0), -6.18, -7, 2.6))

    def test_mixed_frames_keep_commands_finite_and_bounded(self):
        rng = Random(20261122)
        for side in (1, 2):
            self.side(side)
            for index in range(360):
                state = rng.choice((GameState.READY, GameState.SET, GameState.PLAYING, GameState.PLAYING))
                kind = rng.choice(list(SetPlay)) if index % 3 else SetPlay.NONE
                owner = rng.choice((1, 2)) if kind != SetPlay.NONE or index % 5 == 0 else 255
                blocked = rng.choice(((), (), (1,), (2,), (3,)))
                stopped = index % 17 == 0
                commands = self.frame(
                    state=state, set_play=kind, owner=owner, stopped=stopped, penalized=blocked,
                    ball=(rng.uniform(-6.9, 6.9), rng.uniform(-4.49, 4.49)),
                    poses=tuple((rng.uniform(-6.9, 6.9), rng.uniform(-4.4, 4.4),
                                 rng.uniform(-math.pi, math.pi)) for _ in range(3)),
                )
                for pid, command in commands.items():
                    if pid in blocked or state == GameState.SET or stopped:
                        self.assertIsInstance(command.intent, StopIntent)
                    if isinstance(command.intent, MoveIntent):
                        self.assertTrue(all(math.isfinite(v) for v in
                                            (command.intent.vx, command.intent.vy, command.intent.vyaw)))
                        self.assertLessEqual(abs(command.intent.vx), 1.1)
                        self.assertLessEqual(abs(command.intent.vyaw), 1.0)
