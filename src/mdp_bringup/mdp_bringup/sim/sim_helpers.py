"""Sim only: the stand-ins for what the real car has and Gazebo doesn't - one
node, started by mdp.launch.py with sim:=true.

  ultrasonic  always        Gazebo's lidar fan -> /ultrasonic, as the real bridge (ultrasonic.py)
  obstacles   task 1        tablet layout -> Gazebo's blocks replaced (obstacles.py)
  arrows      task 2 +      the sim layout's arrows on /yolo_result, a YOLO
              fake_arrows   that always reads them (arrows.py)
"""
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from mdp_bringup.sim.arrows import Arrows
from mdp_bringup.sim.obstacles import Obstacles
from mdp_bringup.sim.ultrasonic import Ultrasonic
from mdp_bringup.utils.run import run


class SimHelpers(Node):
    def __init__(self):
        super().__init__('sim_helpers')
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)
        task = str(self.declare_parameter('task', '0').value)
        world = self.declare_parameter('world', 'task1_arena').value
        fake_arrows = self.declare_parameter('fake_arrows', False).value
        layout = self.declare_parameter(
            'layout', f"{get_package_share_directory('mdp_bringup')}/config/tasks.yaml").value
        self.parts = [Ultrasonic(self)]
        if task == '1':
            self.parts.append(Obstacles(self, world, layout))
        if task == '2' and fake_arrows:
            self.parts.append(Arrows(self, layout))


def main(args=None):
    run(SimHelpers, args=args)


if __name__ == '__main__':
    main()
