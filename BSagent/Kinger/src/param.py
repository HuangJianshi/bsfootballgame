"""Central configuration point for strategy tuning.

This module contains tunable strategy, movement, kicking, and obstacle
avoidance parameters. Existing values were moved here from their original
modules; make future parameter adjustments here first.
"""

from __future__ import annotations

import math


# ======================================================================
# Kick power
# ======================================================================

# Player.kick() clamps power to this range. Unsupported values may cause a fall.
KICK_POWER_MIN = 1.0
KICK_POWER_MAX = 10.0

# Kick power during normal play.
KICK_POWER_DEFAULT = 5.0
KICK_POWER_BACKFIELD = 5.0
KICK_POWER_OUR_KICKOFF = 5.0


# ======================================================================
# Player movement control
# ======================================================================

ARRIVE_DIST = 0.15             # Target arrival threshold (m)
OMNI_DIST = 1.0                # Use omnidirectional control below this distance
TURN_THRESHOLD = 0.5           # Turn in place above this heading error (rad)
MAX_LINEAR = 2.6               # Maximum commanded forward speed (m/s)
MAX_ANGULAR = 2.0              # Maximum commanded angular speed (rad/s)
LINEAR_GAIN = 1.5              # Translation proportional gain
ANGULAR_GAIN = 2.0             # Rotation proportional gain


# ======================================================================
# Player kicking and shot planning
# ======================================================================

KICK_ENTER_M = 2.0             # Enter kicking state below this ball distance
KICK_EXIT_M = 2.5              # Leave kicking state above this ball distance
CHASE_BEHIND_M = 0.35          # Distance behind the ball while chasing

# The visual kick only fires when the ball is inside a forward cone. When the
# ball is beside or behind the robot, turn in place to bring it into this
# half-angle first, then shoot with the same aim as before.
KICK_FRONT_CONE_RAD = math.radians(50)
# Turn until the ball is this much inside the cone, so the next frame does not
# immediately fall back out of it and start turning again.
KICK_FRONT_CONE_SETTLE_RAD = math.radians(10)
# Beyond this body-frame angle the ball is effectively behind the robot; pivoting
# in place would be slow, so fall back to walking around the ball instead.
KICK_TURN_MAX_RAD = math.radians(130)
# After pivoting, the shot direction must stay within this body-frame angle.
# Mechanical kick range is limited, so a pivot that would leave the aim far off
# the body axis is rejected in favour of walking around the ball.
KICK_AIM_MAX_RAD = math.radians(45)
# Do not pivot when the ball is this close; rotating the body would knock the
# ball. Below this distance the player keeps walking around the ball instead.
KICK_PIVOT_MIN_DIST_M = 0.6


# ======================================================================
# Aggressive attack: clearance, long ball, shooting
# ======================================================================
# Every aggressive action reuses Player.kick(direction, power): the SDK visual
# kick accepts a field-frame direction and power in [KICK_POWER_MIN,
# KICK_POWER_MAX], so a long ball or clearance is the same action with a
# farther target and a higher power. Cooldowns and lane checks keep the
# decisions from flipping every frame.

# Backfield clearance (大脚解围)
CLEARANCE_BALL_X_M = -1.0        # Ball is in our backfield below this x (m)
CLEARANCE_PRESSURE_M = 1.8       # Opponent distance counted as pressure (m)
CLEARANCE_MIN_PRESSERS = 1       # Pressing opponents needed to force a clearance
CLEARANCE_MIN_GAIN_X_M = 3.5     # Clearance must gain at least this much ground (m)
CLEARANCE_POWER = 9.0            # Kick power for a clearance
CLEARANCE_COOLDOWN_S = 1.2       # Minimum interval between clearances (s)
CLEARANCE_KEY = "clearance"

# Midfield long ball and through ball (长传/直塞)
LONG_BALL_BALL_X_M = -1.0        # Long balls are considered from this x onward (m)
LONG_BALL_MIN_GAIN_X_M = 3.0     # Forward target must be this far upfield (m)
LONG_BALL_CLEARANCE_M = 1.2      # Minimum opponent clearance on the lane (m)
LONG_BALL_POWER = 7.5            # Kick power for a long ball
LONG_BALL_COOLDOWN_S = 1.5       # Minimum interval between long balls (s)
LONG_BALL_KEY = "long_ball"

# Shooting
SHOT_RANGE_M = 4.5               # Shoot from this goal distance when lane is open (m)
SHOT_RANGE_CLOSE_M = 2.4         # Reduced lane requirement at close range (m)
SHOT_LANE_CLEARANCE_M = 0.9      # Minimum opponent clearance to the goal line (m)
SHOT_LANE_CLOSE_CLEARANCE_M = 0.45  # Close shots still need a usable lane (m)
SHOT_POWER = 10.0                # Always use full legal power for shots
SHOT_TARGET_SPREAD_M = 0.55      # Lateral goal-mouth aim points beside centre (m)
SHOT_RANGE_OPEN_GOAL_M = 6.2     # Shoot from this range when the goal is vacant
OPEN_GOAL_CLEAR_M = 1.25         # No opponent this close to goal counts as empty
SHOT_COOLDOWN_S = 0.8            # Minimum interval between shots (s)
SHOT_KEY = "shot"

# ======================================================================
# Passing: forward pass, lateral switch, back pass
# ======================================================================
# On-ball priority follows possession-first football: shoot, then a safe
# forward pass, then a lateral switch, then a backward/lateral keep-ball pass.
# Long balls and clearances are risk-control tools and are only used when no
# safe pass exists while under pressure.

PASS_MIN_GAIN_X_M = 1.2          # Forward pass must advance at least this much (m)
PASS_LANE_CLEARANCE_M = 1.0      # Minimum opponent clearance on the pass lane (m)
PASS_RECEIVER_SPACE_M = 1.2      # Receiver must have this much free space (m)
PASS_POWER_BASE = 3.5            # Minimum pass power
PASS_POWER_PER_M = 0.6           # Extra power per metre of pass distance
PASS_COOLDOWN_S = 0.5            # Minimum interval between passes (s)
PASS_FORWARD_KEY = "pass_forward"
PASS_LATERAL_KEY = "pass_lateral"
PASS_BACK_KEY = "pass_back"

LATERAL_MIN_GAIN_Y_M = 1.5       # Lateral switch must gain this much width (m)
BACK_PASS_MAX_PRESSURE = 1      # Back pass only below this many pressers
BACKFIELD_UNSAFE_PASS_X_M = -4.0  # No lateral/back pass this close to our goal (m)

# ======================================================================
# Defending: pressing and role depth
# ======================================================================
BALL_OWNED_DIST_M = 1.4          # Nearest player inside this range owns the ball (m)
BALL_CONTROL_MARGIN_M = 0.35     # Distance advantage needed for clear possession (m)
PRESS_MAX_DEFENDERS = 2          # Never send more than two players to the ball
DEFEND_PRESS_DIST_M = 4.0        # First defender engages inside this distance (m)
DEFEND_COVER_DIST_M = 2.2        # Cover slot distance behind the first defender (m)
DEFEND_COVER_LATERAL_M = 1.2     # Cover slot lateral offset toward the center (m)
DEFEND_BALANCE_X_M = -3.0        # Balance player holds at least this deep (m)
DEFEND_ENGAGE_BALL_X_M = 0.5     # Default midfield block starts below this x (m)
DEFEND_BLOCK_DEPTH_M = 4.0       # Defender holds this far goal-side of a high ball (m)
DEFEND_COMMIT_DIST_M = 1.6       # Commit to the ball inside this distance (m)
DEFEND_APPROACH_OFFSET_M = 0.8   # Challenge point goal-side of the ball (m)
DEFEND_APPROACH_CENTER_M = 0.45  # Challenge point offset toward the centre (m)
DEFEND_BALANCE_SETBACK_M = 2.3   # Balance player depth behind the cover (m)
DEFEND_GOAL_LINE_OFFSET_M = 0.75 # Emergency balance slot inside our goal line (m)
DEFEND_BALANCE_BALL_TRACK = 0.35 # Shift across goal toward the ball (fraction)
DEFEND_MARK_DIST_M = 1.25        # Cover distance from a dangerous runner (m)
DEFEND_MARK_LEAD_M = 0.55        # Runner must be this far goal-side of ball (m)
DEFEND_LATE_PUSH_BONUS_M = 1.0   # Extra press line advance when trailing late (m)
COUNTER_PRESS_WINDOW_S = 3.0     # Immediate counter-press window after a loss (s)
COUNTER_PRESS_MAX_DIST_M = 2.5   # Only the player this close joins counter-press (m)

PRESS_LOOSE_TOUCH_M = 1.1        # Opponent ball distance counted as a loose touch (m)
PRESS_SIDELINE_Y_M = 2.6         # Ball this wide counts as isolated on the wing (m)
PRESS_BACK_TO_GOAL_COS = -0.1    # Opponent facing away from our goal can trigger press

# ======================================================================
# Score-aware intensity
# ======================================================================
SCORE_LEAD_PRESS_SHIFT_X_M = -1.2  # Leading: engage line moves this much deeper
SCORE_TRAIL_PRESS_SHIFT_X_M = 1.5  # Trailing: engage line moves this much higher
SCORE_TRAIL_SHOT_RANGE_BONUS_M = 1.5  # Trailing: extra shooting range (m)
SCORE_LATE_REMAINING_S = 60        # Last minute can raise pressing intensity

# ======================================================================
# Player skill parameters: guarding and support
# ======================================================================

GUARD_FACE_BALL = True
GUARD_THREAT_ENTER_X = -1.0
GUARD_THREAT_EXIT_X = -0.7

SUPPORT_DIST_M = 3.0

# ======================================================================
# Normal-phase strategy
# ======================================================================

ATTACKER_KEEP_DIST_MARGIN_M = 0.3  # Prevent attacker-selection oscillation
ROLE_KEEP_DIST_MARGIN_M = 0.25     # Prevent support/cover role oscillation

# Attacking triangle. The near support stays goal-side of the ball so one
# player always protects the centre; the far support attacks the weak side.
ATTACK_NEAR_SUPPORT_DIST_M = 3.25  # Short outlet behind the ball (m)
ATTACK_NEAR_CENTRAL_M = 1.1        # Near support offset toward the centre (m)
ATTACK_FAR_FORWARD_M = 2.8         # Far support run beyond the ball (m)
ATTACK_FAR_WIDTH_M = 2.1           # Far support run toward the weak side (m)

FALLEN_COST = 10.0  # Distance penalty assigned to a fallen player (m)


# ======================================================================
# Kickoff and set-play strategy
# ======================================================================

# Kickoff
KICKOFF_STAGE_M = 2.0
KICKOFF_FRONT_MARGIN = 0.1
KICKOFF_LATERAL_TOL = 0.35

CENTER_LEAVE_DIST_M = 0.15 # Distance from center at which the ball counts as moved

OPP_SET_WALL_DIST_M = 2.0 # Blocking distance during an opponent kickoff

# Opponent set plays: retreat threshold and field boundary margin (m).
OPP_SET_PLAY_CLEARANCE_M = 1.5
OPP_SET_PLAY_FIELD_MARGIN_M = 0.3


# ======================================================================
# Positioning and avoidance
# ======================================================================

OPPONENT_RESTART_AVOID_M = 1.65
CIRCLE_MARGIN_M = 0.3


# ======================================================================
# Kick-target geometry
# ======================================================================

GOAL_TARGET_DEPTH_M = 0.25              # Kick target depth inside the goal (m)

# ======================================================================
# Obstacle geometry
# ======================================================================

BALL_OBSTACLE_RADIUS = 0.5              # Ball obstacle radius (m)
OPPONENT_RADIUS = 0.55                  # Opponent robot radius (m)
TEAMMATE_RADIUS = 0.48                  # Teammate robot radius (m)
SAFETY_MARGIN = 0.22                    # General safety margin (m)

GOAL_DEPTH = 0.6                        # Goal depth used for obstacle modeling (m)
POST_RADIUS = 0.18                      # Goal-post radius (m)
NET_RADIUS = 0.20                       # Goal-net sample radius (m)
NET_STEP = 0.35                         # Goal-net sampling interval (m)

START_IGNORE = 0.0                      # Ignore obstacles near the start (m)
TARGET_IGNORE = 0.0                     # Ignore obstacles near the target (m)

# ======================================================================
# Global path planner (A* grid planner)
# ======================================================================

USE_GLOBAL_PATH_PLANNER = True          # Fall back to local VFH when disabled
GLOBAL_GRID_RESOLUTION_M = 0.35         # Grid cell size; smaller is finer (m)
GLOBAL_FIELD_MARGIN_M = 0.25            # Extra field margin for boundary paths (m)
GLOBAL_OBSTACLE_MARGIN_M = 0.10         # Additional obstacle inflation radius (m)
GLOBAL_PATH_LOOKAHEAD_M = 0.9           # Lookahead distance along the path (m)

# ======================================================================
# Local path planner (VFH direction scan)
# ======================================================================

PLAN_LOOKAHEAD = 1.2                    # Forward probe length (m)
PLAN_CLEARANCE = 0.35                   # Minimum clearance for a candidate (m)
PLAN_STEP = math.radians(15)            # Candidate direction scan step (rad)
PLAN_MAX_OFFSET = math.radians(100)     # Maximum offset from target heading (rad)

# ======================================================================
# Visualization
# ======================================================================

KICK_TARGET_MARK_SIZE_M = 0.18
