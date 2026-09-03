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
# firmware steering clamp is smaller. As of 2026-09-03
# (mdp_stm32/include/servo.h): SERVO_ANGLE_MAX_RIGHT_RAD = 18.3deg (the
# TIGHTER, confirmed-conservative side - real hardware limit is further
# out, not yet re-measured) vs SERVO_ANGLE_MAX_LEFT_RAD = 28.1deg. Hybrid
# A*'s L/S/R primitives assume one symmetric minR for both turn
# directions, so the more restrictive (right) side must be used - a plan
# that never turns tighter than this radius is drivable on BOTH sides
# today. Using the theoretical 32.5deg real-lock radius (~22.5cm) here
# would generate paths the firmware currently clamps mid-turn on the
# right, degrading path tracking rather than improving it.
#
# TODO: once mdp_stm32's right-side servo fine-sweep (flagged in
# docs/stm32/tuning.md) finds the real right-side lock and
# SERVO_ANGLE_MAX_RIGHT_RAD widens, recompute this and re-plan with a
# tighter (more capable) minR.
_STEERING_CLAMP_DEG = 18.3
MIN_TURN_RADIUS_CM = WHEELBASE_CM / math.tan(math.radians(_STEERING_CLAMP_DEG))  # ~= 43.3cm

# Half the car's own front-to-back footprint length beyond the rear axle,
# used by hybrid_astar.py/hamiltonian.py to collision-check the car's
# extremities (not just the rear-axle reference point) during search.
# NOT independently measured on this chassis - carried over from the
# teammate's mdp_algo default as a placeholder. Flagging rather than
# guessing a new number: measure this chassis's actual rear-axle-to-front
# (or rear-axle-to-center) distance before trusting collision checks near
# tight obstacle gaps.
REAR_AXLE_TO_CENTER_CM = 9.5
