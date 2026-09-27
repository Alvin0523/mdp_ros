"""`pixi run calib <what> ...` - the calibration and tuning drives, one command.

    calib straight 1.0      wheel size: drive 1.0 m, tape-measure what it really did
    calib rotate 90         heading (IMU / EKF): turn 90 deg, check with a protractor
    calib turn left         turning circle at full lock: half a circle, measure the diameter
    calib goto 5 8 E        planner + follower end to end: stop in cell (5,8) facing E

`calib <what> -h` shows each one's options. All drive the BARE car (task:=0,
`pixi run sim` / `pixi run real`): task1_runner / task2_runner own /cmd_vel, so
this refuses to start next to them.
"""
import importlib
import sys
import time

import rclpy

MODES = {
    'straight': 'straight',
    'rotate': 'rotate',
    'turn': 'turn',
    'goto': 'goto',
}
RUNNERS = ('task1_runner', 'task2_runner')
DISCOVERY_S = 4.0      # the ROS graph takes a few seconds to show the other nodes


def runner_running() -> str:
    """Name of a task runner on the ROS graph, or ''."""
    rclpy.init()
    node = rclpy.create_node('calib_check')
    try:
        end = time.monotonic() + DISCOVERY_S
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
            busy = [n for n in RUNNERS if n in node.get_node_names()]
            if busy:
                return busy[0]
        return ''
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main():
    args = sys.argv[1:]
    if not args or args[0] not in MODES:
        sys.exit(__doc__)
    mode, rest = args[0], args[1:]
    # around_obstacle runs several steps back to back and has checked already.
    check = '--no-runner-check' not in rest
    rest = [a for a in rest if a != '--no-runner-check']
    if check and '-h' not in rest and '--help' not in rest:
        busy = runner_running()
        if busy:
            sys.exit(f'{busy} is running and owns /cmd_vel - start the bare car (task:=0) '
                     f'to use calib {mode}')
    sys.argv = [f'calib {mode}'] + rest
    importlib.import_module(f'mdp_bringup.tools.calib.{MODES[mode]}').main()


if __name__ == '__main__':
    main()
