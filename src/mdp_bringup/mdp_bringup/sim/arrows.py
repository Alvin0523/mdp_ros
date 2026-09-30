"""Task 2 (fake_arrows:=true): a YOLO that always reads the arrows (part of sim_helpers).

Gazebo's camera is too coarse for YOLO to read the arrows from as far as the
real run needs (from home for arrow 1). This publishes the arrows of the sim
layout (config/tasks.yaml task2 sim: arrow_1 / arrow_2) on /yolo_result, as a
working YOLO would, while task2_runner looks for them:

  arrow 1   WAITING_FOR_GO, APPROACH_1
  arrow 2   TO_CHECKPOINT_1, CHECKPOINT_1   (the runner takes it once it is
            level with obstacle 1)
"""
from mdp_interfaces.msg import RunStatus
from std_msgs.msg import String
import yaml

from mdp_bringup.utils.run import wall_timer

YOLO_ID = {'LEFT': '39', 'RIGHT': '38'}


class Arrows:
    def __init__(self, node, layout):
        with open(layout) as f:
            sim = yaml.safe_load(f)['task2']['sim']
        self.ids = [YOLO_ID[str(sim['arrow_1']).upper()], YOLO_ID[str(sim['arrow_2']).upper()]]
        self.state = ''
        self.pub = node.create_publisher(String, '/yolo_result', 10)
        node.create_subscription(RunStatus, '/run_status', lambda m: setattr(self, 'state', m.state), 10)
        wall_timer(node, 0.1, self.tick)
        node.get_logger().info(f"fake YOLO: arrow 1 {sim['arrow_1']}, arrow 2 {sim['arrow_2']}")

    def tick(self):
        if self.state in ('WAITING_FOR_GO', 'APPROACH_1'):
            self.pub.publish(String(data=self.ids[0]))
        elif self.state in ('TO_CHECKPOINT_1', 'CHECKPOINT_1'):
            self.pub.publish(String(data=self.ids[1]))

