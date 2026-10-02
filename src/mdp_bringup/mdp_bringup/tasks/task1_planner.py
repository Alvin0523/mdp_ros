"""Task 1 leg planner on the laptop (`pixi run laptop`): plans the Hybrid A*
paths between the car's checkpoints, the slow part of planning, and sends each
one back as soon as it is found.

task1_runner on the Pi (role:=pi, remote_planner) still works out the visiting
order and checkpoints itself (milliseconds), then sends

    /plan_legs/request  {"gen", "obstacles_cm": [[x, y, facing], ...],
                         "start": [x, y, theta], "checkpoints": [[x, y, theta], ...],
                         "params": {"<section>.<field>": value, ...}}

and this node answers on /plan_legs/result with {"gen", "ack": true} at once,
then {"gen", "idx", "start", "path": [[x, y, theta, gear], ...]} per leg ([] =
no path). The runner's own planner settings come with the request, so a live
`ros2 param set` on the Pi applies here too. A newer request (gen) abandons the
one being planned. Nothing is sent while the car drives: it follows the paths
on its own, and plans small re-plans itself. Without an ack within
remote_plan_timeout the runner plans every leg itself (the Pi alone).
"""
import json
import threading
import traceback

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from mdp_algorithm.planning.planner import build_costmap, plan_leg
from mdp_algorithm.utils import params as planner_params
from mdp_bringup.utils.run import run

REQUEST, RESULT = '/plan_legs/request', '/plan_legs/result'


class Task1Planner(Node):
    def __init__(self):
        super().__init__('task1_planner')
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)
        qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(String, RESULT, qos)
        self.create_subscription(String, REQUEST, self.on_request, qos)
        self.gen = None                  # the request being planned
        self.lock = threading.Lock()
        self.get_logger().info('PLANNER   ready - planning task 1 legs for the car')

    def send(self, data: dict):
        self.pub.publish(String(data=json.dumps(data)))

    def on_request(self, msg: String):
        req = json.loads(msg.data)
        self.gen = req['gen']
        self.send({'gen': req['gen'], 'ack': True})
        self.get_logger().info(f"PLANNER   {len(req['checkpoints'])} legs requested")
        threading.Thread(target=self.plan, args=(req,), daemon=True).start()

    def plan(self, req: dict):
        with self.lock:                  # one plan at a time; a newer gen stops this one
            try:
                planner_params.configure(planner_params.from_flat(req['params']))
                costmap = build_costmap([tuple(o) for o in req['obstacles_cm']])
                pose = tuple(req['start'])
                for idx, target in enumerate(req['checkpoints']):
                    if self.gen != req['gen']:
                        return
                    path = plan_leg(costmap, pose, tuple(target)) or []
                    if self.gen != req['gen']:
                        return
                    self.send({'gen': req['gen'], 'idx': idx, 'start': list(pose),
                               'path': [list(map(float, p[:3])) + [int(p[3])] for p in path]})
                    self.get_logger().info(f"PLANNER   leg {idx + 1}/{len(req['checkpoints'])} "
                                           f"{'sent' if path else 'NO PATH'}")
                    pose = tuple(target)
            except Exception:
                self.get_logger().error(f"planning crashed:\n{traceback.format_exc()}")


def main(args=None):
    run(Task1Planner, args=args)


if __name__ == '__main__':
    main()
