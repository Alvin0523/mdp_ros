"""The tablet's movement buttons for the BARE car (task:=0, `pixi run real`).

/manual_drive (f b fl fr bl br) -> a short burst on /cmd_vel, then one zero -
see mdp_bringup/utils/manual.py. Started only for task:=0: in task 1 the runner
owns /cmd_vel and handles the buttons itself with the same helper.

Publishes only while a burst runs, so it does not fight dist / rotate / circle /
teleop. Tune live:  ros2 param set /manual_drive manual_speed_mps 0.25
"""
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from std_msgs.msg import String

from mdp_algorithm.utils import params as planner_params
from mdp_bringup.utils.run import run, wall_timer
from mdp_bringup.utils import manual

WHEELBASE_M = planner_params.car_from_urdf()['wheelbase']   # the URDF


class ManualDrive(Node):
    def __init__(self):
        super().__init__('manual_drive')
        manual.declare_params(self)
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(String, '/manual_drive', self.on_button, 10)
        wall_timer(self, 0.05, self.tick)
        self.cmd = (0.0, 0.0)
        self.until = 0.0
        self.active = False
        self.get_logger().info('manual drive ready: /manual_drive -> /cmd_vel (bare car)')

    def now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def on_button(self, msg: String):
        cmd = manual.burst(self, msg.data.strip().lower(), WHEELBASE_M)
        if cmd is None:
            self.get_logger().warn(f'unknown manual drive command {msg.data!r}')
            return
        v, w, seconds = cmd
        self.cmd, self.until, self.active = (v, w), self.now() + seconds, True

    def send(self, v: float, w: float):
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'base_link'
        m.twist.linear.x, m.twist.angular.z = v, w
        self.pub.publish(m)

    def tick(self):
        if self.until > self.now():
            self.send(*self.cmd)
        elif self.active:
            self.active = False
            self.send(0.0, 0.0)


def main():
    run(ManualDrive)


if __name__ == '__main__':
    main()
