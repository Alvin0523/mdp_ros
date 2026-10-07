"""`calib ir`: fit each side IR sensor's distance curve, cm = a / (raw/4095)^b -
the ir1_curve / ir2_curve in config/bridges.yaml (serial bridge).

    pixi run calib ir                          IR1 then IR2, 10/15/20/25/30/40 cm
    pixi run calib ir --sensors 2              IR2 only
    pixi run calib ir --distances 10 20 30     other distances (cm)
    pixi run calib ir --wait 15                more time to move the block (s)

HOW: the car stands still with the stack up (pixi run pi / pi-solo). For every
sensor and distance it says where to put the block - a flat face square to the
beam, the distance measured from the SENSOR'S FACE - counts down --wait seconds
while you move it, then samples the raw reading (/ir/raw, /ir2/raw) for
--sample seconds. At the end, per sensor: the new curve, the error at each
distance with the curve now in bridges.yaml and with the new one, and the line
to paste into bridges.yaml. Nothing is changed for you; every point goes into
calibration_log.csv (tools/calib/log.py). Restart the serial bridge (pixi run pi)
after editing bridges.yaml.
"""
import argparse
import math
import statistics
import sys
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import UInt16

from mdp_bringup.tools.calib import log

TOPICS = {1: '/ir/raw', 2: '/ir2/raw'}
NAMES = {1: 'IR1 (/ir)', 2: 'IR2 (/ir2)'}
STM32_CURVE = (6.30, 1.226)       # the STM32's generic formula (ir.c)


def curve_cm(raw: float, curve) -> float:
    return curve[0] / (raw / 4095.0) ** curve[1] if raw > 0 else float('inf')


def fit(points):
    """(a, b) of cm = a / x^b through (raw, cm) points, x = raw/4095: a straight
    line in logs, log cm = log a - b log x, least squares."""
    xs = [math.log(raw / 4095.0) for raw, _ in points]
    ys = [math.log(cm) for _, cm in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return math.exp(my - slope * mx), -slope


def current_curves():
    """ir1_curve / ir2_curve now in the installed bridges.yaml (the STM32 formula if missing)."""
    try:
        path = f"{get_package_share_directory('mdp_bringup')}/config/bridges.yaml"
        with open(path) as f:
            p = yaml.safe_load(f)['serial_bridge_node']['ros__parameters']
        return {i: tuple(p.get(f'ir{i}_curve', STM32_CURVE)) for i in (1, 2)}
    except Exception:
        return {1: STM32_CURVE, 2: STM32_CURVE}


class Sampler:
    def __init__(self, node):
        self.node = node
        self.latest = {1: [], 2: []}
        self.counts = {1: 0, 2: 0}
        for i, topic in TOPICS.items():
            node.create_subscription(UInt16, topic, lambda m, i=i: self._on(i, m), 50)

    def _on(self, i, msg):
        self.counts[i] += 1
        self.latest[i].append(msg.data)

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.02)

    def sample(self, i, seconds):
        self.latest[i] = []
        self.spin_for(seconds)
        return list(self.latest[i])


def main():
    ap = argparse.ArgumentParser(description='Fit the side IR sensors\' distance curves: calib ir')
    ap.add_argument('--sensors', type=int, nargs='+', default=[1, 2], choices=[1, 2])
    ap.add_argument('--distances', type=float, nargs='+', default=[10, 15, 20, 25, 30, 40],
                    help='cm from the sensor face to the block (default 10 15 20 25 30 40)')
    ap.add_argument('--wait', type=float, default=10.0, help='s to move the block before each sample (default 10)')
    ap.add_argument('--sample', type=float, default=3.0, help='s of readings per distance (default 3)')
    a, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = rclpy.create_node('calib_ir')
    sampler = Sampler(node)
    sampler.spin_for(2.0)
    missing = [NAMES[i] for i in a.sensors if sampler.counts[i] == 0]
    if missing:
        print(f"No raw readings from {', '.join(missing)} ({', '.join(TOPICS[i] for i in a.sensors)}).\n"
              f"Is the car stack up (pixi run pi), and the serial bridge new enough to publish /ir/raw?")
        rclpy.shutdown()
        return 1
    where = log.where(node)
    old = current_curves()
    results = {}
    try:
        for i in a.sensors:
            print(f"\n=== {NAMES[i]} - curve now a={old[i][0]:.2f} b={old[i][1]:.3f}")
            points = []
            for d in a.distances:
                print(f"\n{NAMES[i]}: put the block at {d:g} cm from the sensor face (flat face, square on)")
                for left in range(int(a.wait), 0, -1):
                    print(f"  sampling in {left:2d} s ", end='\r', flush=True)
                    sampler.spin_for(1.0)
                raws = sampler.sample(i, a.sample)
                if not raws:
                    print(f"  no readings - skipped {d:g} cm")
                    continue
                raw = statistics.median(raws)
                spread = (max(raws) - min(raws)) / 2
                old_cm = curve_cm(raw, old[i])
                print(f"  raw {raw:.0f} (+-{spread:.0f}, {len(raws)} readings) -> {old_cm:.1f} cm with the curve now")
                points.append((raw, d))
                log.write(where, f'ir{i}', f'{d:g} cm', None, old_cm, d, 'cm',
                          f'raw median {raw:.0f}, +-{spread:.0f} over {len(raws)} readings')
            results[i] = points
    except KeyboardInterrupt:
        print('\nstopped - fitting what was measured')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

    print()
    for i, points in results.items():
        if len(points) < 3:
            print(f"{NAMES[i]}: {len(points)} point(s) - need 3+ to fit")
            continue
        new = fit(points)
        print(f"=== {NAMES[i]}: new curve a={new[0]:.2f} b={new[1]:.3f}")
        print(f"   {'true':>6} {'curve now':>10} {'new curve':>10}")
        for raw, d in points:
            print(f"   {d:6.1f} {curve_cm(raw, old[i]):10.1f} {curve_cm(raw, new):10.1f}")
        print(f"   -> config/bridges.yaml:  ir{i}_curve: [{new[0]:.2f}, {new[1]:.3f}]")
        log.write(where, f'ir{i} fit', ' '.join(f'{d:g}' for _, d in points) + ' cm', None, None, None, '',
                  f'curve a={new[0]:.3f} b={new[1]:.4f} (was {old[i][0]}, {old[i][1]})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
