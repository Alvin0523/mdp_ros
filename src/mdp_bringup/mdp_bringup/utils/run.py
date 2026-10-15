"""Spin one node until Ctrl+C, with a clean shutdown - every mdp_bringup node's main().

When the launch stops (Ctrl+C), ROS shuts down underneath the running node, so
a callback or destroy_node() can fail half-way, and a second Ctrl+C (from the
terminal AND from launch) can land during the clean-up. Those are only the
shutdown and are dropped; an error while ROS is still up is re-raised.
Ctrl+C reaches rclpy a moment before it reports not-ok, so an error in that
moment (a message taken mid-shutdown) is given SHUTDOWN_GRACE_S to turn out
to be the shutdown.
"""
import signal
import time

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException

SHUTDOWN_GRACE_S = 0.5


def _shutting_down() -> bool:
    deadline = time.monotonic() + SHUTDOWN_GRACE_S
    while rclpy.ok():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


def wall_timer(node, period_s, callback):
    """A timer on the computer's steady clock, not ROS time - for control loops
    (like Nav2's). On sim time, a ROS-time timer sometimes never started when
    the node came up during Gazebo's start (task2_runner: no tick at all, seen
    2026-09-28 in ~1 run in 5 with YOLO running). What the callback does with
    time still uses node.get_clock(), i.e. sim time in sim."""
    return node.create_timer(period_s, callback, clock=Clock(clock_type=ClockType.STEADY_TIME))


def run(make_node, args=None, events=False):
    """events=True: rclpy's EventsExecutor instead of the default one. Per message
    it costs a third of the CPU: a node taking /ultrasonic, /ir, /ir2 (94 Hz each)
    and /odometry/filtered (40 Hz), 350 msg/s, used 42-51% of a Pi core, 15-22%
    with it (2026-10-15). Still experimental in Jazzy: opt in per node."""
    rclpy.init(args=args)
    node = make_node()
    try:
        if events:
            from rclpy.experimental import EventsExecutor
            executor = EventsExecutor()
            executor.add_node(node)
            executor.spin()
        else:
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if not _shutting_down():
            raise
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # already stopping: ignore a second Ctrl+C
        try:
            node.destroy_node()
        except (Exception, KeyboardInterrupt):   # Ctrl+C reaches a node twice (terminal + launch)
            pass
        rclpy.try_shutdown()
