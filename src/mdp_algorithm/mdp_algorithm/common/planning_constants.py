#!/usr/bin/env python3
"""
Physical constants for the collision-aware planner (occupancy_map.py,
hamiltonian.py, hybrid_astar.py), reconciled against this robot's actual
measured hardware instead of the teammate's mdp_algo defaults. All
distances here are in CENTIMETRES, matching this planning package's
internal units (see occupancy_map.py's module docstring) - convert to
metres only at the mdp_algorithm package boundary.
"""

import math

# Wheelbase - matches real_controller.yaml (0.1433m) and
# mdp_algorithm/pure_pursuit_follower.py (fixed to the same value
# 2026-09-03, see that file). The teammate's REAR_AXLE_TO_CENTER=9.5cm
# implied a different (~19cm) wheelbase-ish geometry - not used.
WHEELBASE_CM = 14.33

# Minimum turning radius the STM32 firmware can currently actually deliver.
#
# wheelbase / tan(steering_angle) is the standard Ackermann relation, and
# the tightest turn this vehicle can make is bounded by whichever side's
# firmware steering clamp is smaller. Hybrid A*'s L/S/R primitives assume
# one symmetric minR for both turn directions, so the more restrictive
# side must be used - a plan that never turns tighter than this radius is
# drivable on BOTH sides.
#
# UPDATED 2026-09-11: now a REAL wheel angle, measured with a protractor at
# the wheel. Every previous value of this constant (18.3deg, then 24deg) was
# an input to WHEELTEC's cubic PWM fit, which turned out not to correspond to
# any real deflection on this chassis - it overstated the true angle by
# roughly 1.5x. Those numbers were never steering angles, so they were never
# valid inputs to the Ackermann relation below. See
# docs/stm32/tuning.md#what-the-earlier-record-got-wrong.
#
# RE-MEASURED 2026-09-18, superseding the 35.0/29.5deg pair above: raw-pulse
# sweeps confirmed both mechanical limits directly (left 950->780us -> 850us
# chassis contact, right 2200->2450us -> 2400us confirmed mechanical stop,
# no longer "calibration ceiling, still tracking"), and protractor angles
# were re-read at both: LEFT +43.0deg (850us pulse), RIGHT -32.5deg (2400us).
# RIGHT remains the tighter/binding side (32.5 < 43.0), so it is still what
# bounds the plan - same reasoning as before, updated numbers.
#
# The right limit is now a CONFIRMED mechanical limit (see mdp_stm32 servo.h),
# not the "still tracking when measurement stopped" caveat that applied to
# the old 29.5deg figure - so this value should not need to shrink further.
_STEERING_CLAMP_DEG = 32.5

# Derived, not hand-set. See _STEERING_CLAMP_DEG's history above for how this
# constant's basis has evolved (guessed override -> 29.5deg-derived ->
# 32.5deg-derived); each revision has only tightened the achievable radius.
MIN_TURN_RADIUS_CM = round(WHEELBASE_CM / math.tan(math.radians(_STEERING_CLAMP_DEG)), 1)

# Half the car's own front-to-back footprint length beyond the rear axle,
# used by hybrid_astar.py/hamiltonian.py to collision-check the car's
# extremities (not just the rear-axle reference point) during search.
# NOT independently measured on this chassis - carried over from the
# teammate's mdp_algo default as a placeholder. Flagging rather than
# guessing a new number: measure this chassis's actual rear-axle-to-front
# (or rear-axle-to-center) distance before trusting collision checks near
# tight obstacle gaps.
REAR_AXLE_TO_CENTER_CM = 9.5


# --- Per-side turning radius (2026-09-25) ------------------------------------
# The steering is NOT symmetric: measured protractor limits are LEFT +43.0deg
# (chassis contact) and RIGHT -32.5deg (mechanical stop), see _STEERING_CLAMP_DEG
# above. wheelbase / tan(angle) gives the tightest rear-axle turning circle on
# each side. The planner uses one radius per side (footprint_astar.py) instead of
# the single, right-limited one, and plans with PLAN_RADIUS_MARGIN headroom so
# the follower still has steering authority (the servo has dead band and backlash,
# and which wheel each protractor reading came from was never recorded).
_STEERING_CLAMP_LEFT_DEG = 43.0
MIN_TURN_RADIUS_LEFT_CM = round(WHEELBASE_CM / math.tan(math.radians(_STEERING_CLAMP_LEFT_DEG)), 1)
MIN_TURN_RADIUS_RIGHT_CM = MIN_TURN_RADIUS_CM
PLAN_RADIUS_MARGIN = 1.15
PLAN_TURN_RADIUS_LEFT_CM = round(MIN_TURN_RADIUS_LEFT_CM * PLAN_RADIUS_MARGIN, 1)
PLAN_TURN_RADIUS_RIGHT_CM = round(MIN_TURN_RADIUS_RIGHT_CM * PLAN_RADIUS_MARGIN, 1)

# --- Body footprint, rear-axle frame (2026-09-25) -----------------------------
# From the URDF and base_link.STL (mdp_description): the chassis mesh spans
# x -10.07..+12.90 cm and y +-7.68 cm in base_link, whose rear axle is at
# x = -6.966 cm (rb_joint / lb_joint), so the chassis runs 3.1 cm behind to
# 19.87 cm ahead of the rear axle. The rear tyres (radius 3.32 cm, centre at
# the axle) reach 3.32 cm behind it and, at track +-7.975 cm plus half a tyre
# width, +-9.24 cm sideways. Camera, controller board and IMU sit inside this
# outline. Mesh values, not tape-measure: verify the nose distance on the car.
FOOTPRINT_REAR_CM = 3.3
FOOTPRINT_FRONT_CM = 19.9
FOOTPRINT_HALF_WIDTH_CM = 9.3

# Clearance kept between the body and an obstacle block / the table edge, on top
# of the body outline: covers dead-reckoning error and the few cm of path-tracking
# error. The checkpoint stand-off leaves 20 - 5 - 9.3 = 5.7 cm at the side.
OBSTACLE_PAD_CM = 3.0
TABLE_EDGE_MARGIN_CM = 2.0
