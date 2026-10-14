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

It also serves /setup_obstacles (`pixi run setup`, Foxglove SETUP): the layout
file HERE (config/tasks.yaml on the laptop) - task 1's set on /obstacle_setup, as
the tablet's DONE would, and task 2's known layout on /task2_setup. The runner on
the Pi then serves none.
"""
import json
import threading
import traceback

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from mdp_algorithm.planning.planner import build_costmap, plan_leg
from mdp_algorithm.utils import params as planner_params
from mdp_bringup.utils import obstacle_layout
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
        self.layout = self.declare_parameter('layout', '').value
        self.setup_pub = self.create_publisher(String, '/obstacle_setup', 10)
        self.task2_setup_pub = self.create_publisher(String, '/task2_setup', 10)
        self.create_service(Trigger, '/setup_obstacles', self.setup_obstacles)
        self.get_logger().info("PLANNER   ready - waiting for the car's plan requests")

    def setup_obstacles(self, request, response):
        """/setup_obstacles: this machine's layout file, as if the tablet sent it -
        task 1's set on /obstacle_setup AND task 2's on /task2_setup (the laptop
        does not know which task the car runs; its runner takes its own)."""
        sent = []
        try:
            obstacles = obstacle_layout.load(self.layout) if self.layout else []
        except (OSError, ValueError, KeyError) as exc:
            obstacles = []
            self.get_logger().warn(f"SETUP     task 1 {self.layout}: {exc}")
        if obstacles:
            self.setup_pub.publish(String(data=obstacle_layout.setup_string(obstacles)))
            self.get_logger().info(f"SETUP     task 1 sent {obstacle_layout.describe(obstacles)}")
            sent.append(f'task 1: {len(obstacles)} obstacles')
        try:
            task2 = obstacle_layout.task2_setup_string(self.layout) if self.layout else ''
        except (OSError, ValueError, KeyError) as exc:
            task2 = ''
            self.get_logger().warn(f"SETUP     task 2 {self.layout}: {exc}")
        if task2:
            self.task2_setup_pub.publish(String(data=task2))
            self.get_logger().info(f"SETUP     task 2 sent {task2}")
            sent.append('task 2 layout')
        response.success = bool(sent)
        response.message = (f"{', '.join(sent)} from {self.layout} (laptop)" if sent
                            else f'no layout in {self.layout or "(no layout)"}')
        return response

    def send(self, data: dict):
        self.pub.publish(String(data=json.dumps(data)))

    def on_request(self, msg: String):
        req = json.loads(msg.data)
        self.send({'gen': req['gen'], 'ack': True})
        if req.get('single'):
            # One leg planned again mid-run (task1_runner.replan_leg): beside any
            # full plan, which it must not stop.
            self.get_logger().info("PLANNER   1 leg requested (replan)")
            threading.Thread(target=self.plan_one, args=(req,), daemon=True).start()
            return
        self.gen = req['gen']
        self.get_logger().info(f"PLANNER   {len(req['checkpoints'])} legs requested")
        threading.Thread(target=self.plan, args=(req,), daemon=True).start()

    def plan_one(self, req: dict):
        try:
            costmap = build_costmap([tuple(o) for o in req['obstacles_cm']])
            path = plan_leg(costmap, tuple(req['start']), tuple(req['checkpoints'][0])) or []
            self.send({'gen': req['gen'], 'idx': 0, 'start': req['start'],
                       'path': [list(map(float, p[:3])) + [int(p[3])] for p in path]})
            self.get_logger().info(f"PLANNER   replan {'sent' if path else 'NO PATH'}")
        except Exception:
            self.get_logger().error(f"replanning crashed:\n{traceback.format_exc()}")

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
