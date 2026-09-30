"""The calibration log: every `calib` run adds one row to calibration_log.csv
(mdp_ros/, next to pixi.toml) - what the car thought against what the tape (or,
in sim, Gazebo's true pose) says. History only: nothing reads it back. Setting a
number from it (URDF / navigation.yaml) is done by hand.

Columns: time, where (sim/real), test, setting (what was asked), speed_mps,
car (the car's own reading), true (tape, or Gazebo in sim), error (car - true),
unit, notes (the extras of that test, e.g. roll-past, gyro drift).
"""
import csv
import math
import os
import sys
import time

COLUMNS = ['time', 'where', 'test', 'setting', 'speed_mps', 'car', 'true', 'error', 'unit', 'notes']
PATH = os.path.join(os.environ.get('PIXI_PROJECT_ROOT', os.getcwd()), 'calibration_log.csv')


def where(node) -> str:
    """'sim' when Gazebo's true pose is on the graph, else 'real'."""
    return 'sim' if any(n == '/sim/ground_truth' for n, _ in node.get_topic_names_and_types()) else 'real'


def ask(question: str):
    """A number typed in the terminal (the tape), or None: blank, or no terminal."""
    if not sys.stdin.isatty():
        return None
    while True:
        try:
            text = input(f'{question} (blank = skip): ').strip()
        except EOFError:
            return None
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            print('  a number please')


def _num(v, digits=1):
    return '' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{v:.{digits}f}'


def write(where_, test, setting, speed, car, true, unit, notes='', digits=1):
    """Append one row and print it."""
    error = car - true if car is not None and true is not None else None
    row = {'time': time.strftime('%Y-%m-%d %H:%M'), 'where': where_, 'test': test, 'setting': setting,
           'speed_mps': _num(speed, 2), 'car': _num(car, digits), 'true': _num(true, digits),
           'error': _num(error, digits), 'unit': unit, 'notes': notes}
    new = not os.path.exists(PATH)
    with open(PATH, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    shown = ', '.join(f'{k} {v}' for k, v in row.items() if v and k not in ('time',))
    print(f'LOGGED    {shown}\n          -> {PATH}')


class Truth:
    """Sim only: Gazebo's true pose of the car, (x, y, yaw) in the world (= map)
    frame, from /sim/ground_truth (transforms[0] = the car). None on the real car."""

    def __init__(self, node):
        from tf2_msgs.msg import TFMessage
        self.pose = None
        node.create_subscription(TFMessage, '/sim/ground_truth', self._on, 10)

    def _on(self, msg):
        if msg.transforms:
            t = msg.transforms[0].transform
            q = t.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.pose = (t.translation.x, t.translation.y, yaw)
