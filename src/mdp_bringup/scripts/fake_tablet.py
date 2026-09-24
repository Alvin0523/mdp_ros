#!/usr/bin/env python3
"""Stand-in for the Android tablet on the Bluetooth serial link.

Creates a pseudo-terminal and symlinks it at a device path (default
/tmp/mdp_fake_tablet), which bluetooth_bridge_node opens exactly as it opens
/dev/rfcomm0 on the robot. Then it behaves like the tablet at setup time: once
the robot is talking (the first PLAN/STATUS line arrives, which also means
task1_runner is up), it sends the layout's OBSTACLE lines and DONE - the same
lines the tablet sends, in tablet cell coordinates. Everything the robot sends
back (ROBOT, TARGET, PLAN, STATUS, ...) is printed.

So `obstacles:=yaml` exercises the whole real path - tablet protocol, bridge,
cell-centre conversion, runner - with the layout YAML in place of a person.

Usage: fake_tablet.py [layout.yaml] [device_link]
Plain Python, no ROS node; launch-injected --ros-args are ignored.
"""

import os
import pty
import select
import sys
import tty

from ament_index_python.packages import get_package_share_directory

import obstacle_layout

DEFAULT_LAYOUT = os.path.join(get_package_share_directory('mdp_bringup'), 'config', 'test_obstacles.yaml')
DEFAULT_LINK = '/tmp/mdp_fake_tablet'


def log(text: str):
    print(f'[fake_tablet] {text}', flush=True)


def main():
    argv = sys.argv[1:]
    if '--ros-args' in argv:
        argv = argv[:argv.index('--ros-args')]
    layout_path = argv[0] if len(argv) > 0 else DEFAULT_LAYOUT
    link = argv[1] if len(argv) > 1 else DEFAULT_LINK

    setup_lines = obstacle_layout.tablet_lines(obstacle_layout.load(layout_path))

    master, slave = pty.openpty()
    # Raw from the start, so nothing is echoed back before the bridge opens it.
    # The slave stays open here too: with no slave fd open, reads on the master
    # return EIO whenever the bridge closes and reopens the link.
    tty.setraw(slave)
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.ttyname(slave), link)
    log(f'{link} -> {os.ttyname(slave)}; layout {layout_path}')

    sent = False
    buf = b''
    try:
        while True:
            ready, _, _ = select.select([master], [], [], 1.0)
            if not ready:
                continue
            buf += os.read(master, 1024)
            while b'\n' in buf:
                raw, buf = buf.split(b'\n', 1)
                line = raw.decode(errors='replace').strip()
                if not line:
                    continue
                log(f'robot -> tablet: {line}')
                if not sent and line.startswith(('PLAN:', 'STATUS:')):
                    for out in setup_lines:
                        log(f'tablet -> robot: {out}')
                        os.write(master, (out + '\n').encode())
                    sent = True
    except KeyboardInterrupt:
        pass
    finally:
        if os.path.islink(link):
            os.remove(link)


if __name__ == '__main__':
    main()
