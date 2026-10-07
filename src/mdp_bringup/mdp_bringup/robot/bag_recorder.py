"""Record a bag with one button - Foxglove's REC, the tablet-free way to record a run.

    /bag/toggle     (std_srvs/Trigger, `pixi run bag`)  not recording: start a new bag,
                    <bag_dir>/rosbag2_<date>_<time>, every topic except raw camera frames
                    (the JPEG camera and YOLO images are in);
                    recording: stop and close it
    /bag/recording  (std_msgs/Bool, latched)  true while recording - Foxglove's REC light

Runs `ros2 bag record` as a child process; stop sends it Ctrl+C so it finishes
the file properly. A recording still going when this node stops is closed too.
`pixi run bag-all` still records everything, images included, from a terminal.
"""
import os
import signal
import subprocess
import time

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from mdp_bringup.utils.run import run

# Raw frames only (~0.9 MB each). The JPEG copies stay in: /image_raw/compressed (what YOLO
# reads) and /yolo_result/image_annotated (its boxes) - ~35 MB a minute, fine on the laptop.
EXCLUDE = '/image_raw|/camera/image_raw'


class BagRecorder(Node):
    def __init__(self):
        super().__init__('bag_recorder')
        default_dir = os.path.join(os.environ.get('PIXI_PROJECT_ROOT', os.getcwd()), 'bags')
        self.bag_dir = self.declare_parameter('bag_dir', default_dir).value
        self.proc = None
        self.path = None
        self.started = 0.0
        self.create_service(Trigger, '/bag/toggle', self.toggle)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.recording_pub = self.create_publisher(Bool, '/bag/recording', latched)
        self.recording_pub.publish(Bool(data=False))

    def recording(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def toggle(self, request, response):
        return self.stop(request, response) if self.recording() else self.start(request, response)

    def start(self, request, response):
        os.makedirs(self.bag_dir, exist_ok=True)
        self.path = os.path.join(self.bag_dir, time.strftime('rosbag2_%Y%m%d_%H%M%S'))
        # Own process group: Ctrl+C in the launch terminal must not cut the bag short.
        self.proc = subprocess.Popen(['ros2', 'bag', 'record', '-a', '--exclude-regex', EXCLUDE, '-o', self.path],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        self.started = time.monotonic()
        self.get_logger().info(f'REC       started -> {self.path}')
        self.recording_pub.publish(Bool(data=True))
        response.success, response.message = True, f'recording {self.path}'
        return response

    def stop(self, request=None, response=None):
        if not self.recording():
            if response is not None:
                response.success, response.message = False, 'not recording'
            return response
        os.killpg(self.proc.pid, signal.SIGINT)        # ros2 bag record closes the file on Ctrl+C
        try:
            self.proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
        seconds = time.monotonic() - self.started
        self.get_logger().info(f'REC       stopped after {seconds:.0f} s -> {self.path}')
        self.recording_pub.publish(Bool(data=False))
        if response is not None:
            response.success, response.message = True, f'saved {self.path} ({seconds:.0f} s)'
        return response

    def destroy_node(self):
        self.stop()
        super().destroy_node()


def main(args=None):
    run(BagRecorder, args=args)


if __name__ == '__main__':
    main()
