"""Sim only (task 1): keep Gazebo's blocks the same as the planner's.

The arena starts with the tasks.yaml layout baked in (with the symbol images).
When a DIFFERENT set arrives on /obstacle_setup - the tablet, in sim - this
removes Gazebo's blocks and spawns the new ones, so the car drives among the
blocks it planned around. The same set as the file (obstacles:=yaml) changes
nothing.

Spawned blocks are plain, no symbol: the tablet sends no images, and a decal
spawned after world load renders black anyway (obstacle_layout.py).

Gazebo's own services, through the `gz` command line:
  /world/<world>/remove   gz.msgs.Entity
  /world/<world>/create   gz.msgs.EntityFactory
"""
import os
import subprocess
import tempfile
import threading

from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String

from mdp_bringup.utils import obstacle_layout
from mdp_bringup.utils.run import run


def _key(obstacles):
    """A set compared by position and image side (ids and symbols aside)."""
    return sorted((round(o.x, 2), round(o.y, 2), o.facing) for o in obstacles)


class SimObstacles(Node):
    def __init__(self):
        super().__init__('sim_obstacles')
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)
        self.world = self.declare_parameter('world', 'task1_arena').value
        layout = self.declare_parameter(
            'layout', f"{get_package_share_directory('mdp_bringup')}/config/tasks.yaml").value
        self.current = obstacle_layout.load(layout, 'task1')
        self.names = [f'obstacle_{o.id}' for o in self.current]   # what Gazebo has now
        self.generation = 0
        self.lock = threading.Lock()
        self.tmp = tempfile.mkdtemp(prefix='mdp_sim_obstacles_')
        self.create_subscription(String, '/obstacle_setup', self.on_setup, 10)

    def on_setup(self, msg: String):
        new = obstacle_layout.from_setup_string(msg.data)
        if not new or _key(new) == _key(self.current):
            return
        self.current = new
        # gz calls take a moment each: off the executor thread.
        threading.Thread(target=self.replace, args=(new,), daemon=True).start()

    def gz(self, service, reqtype, req) -> bool:
        cmd = ['gz', 'service', '-s', f'/world/{self.world}/{service}', '--reqtype', reqtype,
               '--reptype', 'gz.msgs.Boolean', '--timeout', '3000', '--req', req]
        out = subprocess.run(cmd, capture_output=True, text=True)
        ok = out.returncode == 0 and 'data: true' in out.stdout
        if not ok:
            self.get_logger().warn(f'gz {service} failed: {out.stdout.strip()} {out.stderr.strip()}')
        return ok

    def replace(self, obstacles):
        with self.lock:
            for name in self.names:
                self.gz('remove', 'gz.msgs.Entity', f'name: "{name}", type: MODEL')
            self.generation += 1
            self.names = []
            for o in obstacles:
                name = f'obstacle_{o.id}_{self.generation}'   # never reuse a just-removed name
                sdf = obstacle_layout.model_sdf(o, sdf_root=True).replace(
                    f'<model name="obstacle_{o.id}">', f'<model name="{name}">')
                # Kept, not deleted: Gazebo reads the file after the call returns.
                path = os.path.join(self.tmp, f'{name}.sdf')
                with open(path, 'w') as f:
                    f.write(sdf)
                if self.gz('create', 'gz.msgs.EntityFactory', f'sdf_filename: "{path}"'):
                    self.names.append(name)
            self.get_logger().info(f'Gazebo blocks replaced: {obstacle_layout.describe(obstacles)}')


def main(args=None):
    run(SimObstacles, args=args)


if __name__ == '__main__':
    main()
