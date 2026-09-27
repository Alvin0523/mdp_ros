"""The tablet's movement buttons (assessment C.3), shared by manual_drive
(task 0, no runner) and task1_runner (which owns /cmd_vel in task 1).

A button - f b fl fr bl br on /manual_drive - is a short burst: forward or
back, straight or at a fixed wheel angle, then stop. Parameters (live):
manual_speed_mps, manual_steer_deg, manual_burst_s - the `/**` section of
config/navigation.yaml.
"""
import math

from mdp_bringup.utils import config

BUTTONS = {'f': (1.0, 0.0), 'b': (-1.0, 0.0), 'fl': (1.0, 1.0), 'fr': (1.0, -1.0),
           'bl': (-1.0, 1.0), 'br': (-1.0, -1.0)}      # button -> (direction, steer side: + left)


def declare_params(node):
    config.declare(node, '/**')


def burst(node, button: str, wheelbase: float):
    """(linear_x, angular_z, seconds) for a button, or None if it is not one.
    Yaw rate by the bicycle model, so steering the same way turns the body the
    other way in reverse."""
    if button not in BUTTONS:
        return None
    direction, side = BUTTONS[button]
    speed = min(0.5, max(0.05, float(node.get_parameter('manual_speed_mps').value)))
    steer = math.radians(min(30.0, max(5.0, float(node.get_parameter('manual_steer_deg').value))))
    seconds = min(1.0, max(0.1, float(node.get_parameter('manual_burst_s').value)))
    v = direction * speed
    return v, v * math.tan(steer) / wheelbase * side, seconds
