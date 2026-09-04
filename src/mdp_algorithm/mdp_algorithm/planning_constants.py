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
# UPDATED 2026-09-05: the right-side fine-sweep this constant's old TODO
# was waiting on has since happened - mdp_stm32/include/servo.h now has
# SERVO_ANGLE_MAX_RIGHT_RAD confirmed at 24deg (real stall measured at
# 26deg, clamped 2deg back for safety margin), replacing the earlier
# provisional 18.3deg value this constant was still using. LEFT is 48deg
# (2deg back from confirmed-clean 50deg), so RIGHT (24deg) is still the
# tighter/binding side - unchanged reasoning, just an updated real number.
# 24deg -> minR ~= 32.2cm, down from ~43.3cm - a real, measured reduction,
# not an arbitrary tightening.
_STEERING_CLAMP_DEG = 24.0

# TEMPORARY OVERRIDE (2026-09-05, direct request - "put it to 25cm for now,
# I'll adjust later"): hardcoded to 25.0 instead of the hardware-derived
# ~32.2cm above. NOT currently drivable by the real servo clamp - 25cm
# needs ~29.8deg of steering, past the confirmed 24deg right-side limit
# (real stall measured at 26deg) - fine for planning-only testing (no
# hardware in the loop yet), but swap back to the derived value (or update
# _STEERING_CLAMP_DEG once the real limit is re-measured wider) before
# trusting a plan to actually drive on the real chassis.
MIN_TURN_RADIUS_CM = 25.0

# Half the car's own front-to-back footprint length beyond the rear axle,
# used by hybrid_astar.py/hamiltonian.py to collision-check the car's
# extremities (not just the rear-axle reference point) during search.
# NOT independently measured on this chassis - carried over from the
# teammate's mdp_algo default as a placeholder. Flagging rather than
# guessing a new number: measure this chassis's actual rear-axle-to-front
# (or rear-axle-to-center) distance before trusting collision checks near
# tight obstacle gaps.
REAR_AXLE_TO_CENTER_CM = 9.5
