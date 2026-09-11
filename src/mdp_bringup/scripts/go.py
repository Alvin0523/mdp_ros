#!/usr/bin/env python3
"""
`go` trigger - tell the runner to START DRIVING.

Calls the runner's /start_run service (std_srvs/Trigger). This is the dev/
bench stand-in for the tablet's "start now" Bluetooth command: in the real
run the tablet sends it once the supervisor is ready; here you run
`pixi run go`.

Why a service (not a topic): starting the run is a one-shot COMMAND that
needs an acknowledgement (accepted / rejected-with-reason), which a service
gives you and a topic does not. The runner only accepts it after setup +
planning (state WAITING_FOR_GO); calling it too early returns success=False
with a reason (e.g. "still planning"), which this script prints and exits
non-zero on, so you know it didn't take.
"""
import sys

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

SERVICE = '/start_run'


class GoClient(Node):
    def __init__(self):
        super().__init__('go_trigger')
        self.cli = self.create_client(Trigger, SERVICE)

    def call(self) -> bool:
        self.get_logger().info(f"Waiting for {SERVICE} service ...")
        if not self.cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                f"{SERVICE} not available - is the runner up? "
                f"(pixi run real / the task1 runner)")
            return False
        future = self.cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        res = future.result()
        if res is None:
            self.get_logger().error("No response from /start_run (timeout).")
            return False
        if res.success:
            self.get_logger().info(f"GO accepted: {res.message}")
        else:
            self.get_logger().warn(f"GO rejected: {res.message}")
        return res.success


def main(args=None):
    rclpy.init(args=args)
    node = GoClient()
    ok = False
    try:
        ok = node.call()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
