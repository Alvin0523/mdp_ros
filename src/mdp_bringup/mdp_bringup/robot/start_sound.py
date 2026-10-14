"""Play a sound on the laptop's speakers when a run starts (START from the tablet,
Foxglove or `pixi run start`) - `pixi run laptop` runs it.

Watches /run_status: the state turning from WAITING_FOR_START to driving (task 1 or 2) is START. The
sound is sounds/start.mp3 in the repo (parameter `sound`); any file ffplay can play.
Played in the background, so a long file never holds anything up.
"""
import os
import shutil
import subprocess

from rclpy.node import Node

from mdp_interfaces.msg import RunStatus
from mdp_bringup.utils.run import run

DRIVING = ('NAVIGATING_TO_TARGET', 'PAUSE_FOR_SCAN',   # task 1
           'TO_CHECKPOINT_1', 'APPROACH_1')              # task 2 (planned at home / measuring on the way)


class StartSound(Node):
    def __init__(self):
        super().__init__('start_sound')
        root = os.environ.get('PIXI_PROJECT_ROOT', os.getcwd())
        self.sound = self.declare_parameter('sound', os.path.join(root, 'sounds', 'start.mp3')).value
        self.player = shutil.which('ffplay')
        self.state = None
        self.create_subscription(RunStatus, '/run_status', self.on_status, 10)
        if not os.path.isfile(self.sound):
            self.get_logger().warn(f"SOUND     no {self.sound} - nothing will play at START")
        elif self.player is None:
            self.get_logger().warn("SOUND     ffplay not found - nothing will play at START")

    def on_status(self, msg: RunStatus):
        if self.state == 'WAITING_FOR_START' and msg.state in DRIVING:
            self.play()
        self.state = msg.state

    def play(self):
        if self.player is None or not os.path.isfile(self.sound):
            return
        subprocess.Popen([self.player, '-nodisp', '-autoexit', '-loglevel', 'quiet', self.sound],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(args=None):
    run(StartSound, args=args)


if __name__ == '__main__':
    main()
