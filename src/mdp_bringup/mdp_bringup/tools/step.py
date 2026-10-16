"""
Task 1 / 2 step-through (the car launched with step:=true): SPACE = /step - the first
starts the run, each next one drives on to the next checkpoint (task 1: after the
stop's IR fix and scan; task 2 at checkpoint 1 with something missing: look again). s = E-STOP (/stop_run), q / Ctrl+C = quit.

    pixi run step

The runner prints what it saw and planned at each checkpoint (its log, Foxglove).
"""
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


def ask(node: Node, client, name: str):
    if not client.wait_for_service(timeout_sec=3.0):
        print(f"\r{name} not available - is the car up with task:=1|2 step:=true?\r")
        return
    future = client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    res = future.result()
    print(f"\r{name}: {res.message if res else 'no reply (timeout)'}\r")


def main(args=None):
    rclpy.init(args=args)
    node = Node('step_keys')
    step = node.create_client(Trigger, '/step')
    stop = node.create_client(Trigger, '/stop_run')
    print("SPACE = next step   s = E-STOP   q = quit")
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            if not select.select([sys.stdin], [], [], 0.1)[0]:
                continue
            key = sys.stdin.read(1)
            if key == ' ':
                ask(node, step, '/step')
            elif key == 's':
                ask(node, stop, '/stop_run')
            elif key in ('q', '\x03'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
