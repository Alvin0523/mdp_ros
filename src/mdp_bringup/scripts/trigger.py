#!/usr/bin/env python3
"""
Call a std_srvs/Trigger service on the runner and print the reply.

    ros2 run mdp_bringup trigger.py /stop_run      (pixi run stop)
    ros2 run mdp_bringup trigger.py /reset_run     (pixi run reset)

Same pattern as go.py (which stays as the /start_run shortcut): a service, not
a topic, so the caller sees accepted / rejected-with-reason. Exits non-zero if
the service is missing or replies success=False.
"""
import sys

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


def call(node: Node, service: str) -> bool:
    client = node.create_client(Trigger, service)
    node.get_logger().info(f"Waiting for {service} ...")
    if not client.wait_for_service(timeout_sec=10.0):
        node.get_logger().error(f"{service} not available - is the runner up? (pixi run real1)")
        return False
    future = client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
    res = future.result()
    if res is None:
        node.get_logger().error(f"No response from {service} (timeout).")
        return False
    log = node.get_logger().info if res.success else node.get_logger().warn
    log(f"{service}: {res.message}")
    return res.success


def main(args=None):
    argv = rclpy.utilities.remove_ros_args(args if args is not None else sys.argv)[1:]
    if len(argv) != 1:
        print("usage: trigger.py <service>   e.g. trigger.py /stop_run", file=sys.stderr)
        sys.exit(2)
    rclpy.init(args=args)
    node = Node('trigger_client')
    ok = False
    try:
        ok = call(node, argv[0])
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
