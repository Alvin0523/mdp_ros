"""`calib ultrasonic`: how well the front ultrasonic reads a block at a known
distance - task 2 measures both obstacle distances with it.

    pixi run calib ultrasonic 60            block 60 cm in front of the sensor
    pixi run calib ultrasonic 60 --samples 100

HOW: stand a block (flat face to the car) straight ahead, measure the tape from
the FRONT FACE OF THE SENSOR to the block, pass that distance. The car does not
move. Takes --samples readings of /ultrasonic and logs the median, the spread
(the middle half of the readings) and the dropouts (no echo) to
calibration_log.csv (tools/calib/log.py). Try 30 / 60 / 100 / 150 cm, and a
block turned ~20 deg to see when the echo gets lost.
"""
import argparse
import math
import statistics

from rclpy.node import Node
from sensor_msgs.msg import Range

from mdp_bringup.tools.calib import log
from mdp_bringup.utils.run import run, wall_timer

TIMEOUT_S = 10.0


class Ultrasonic(Node):
    def __init__(self, distance_cm, samples):
        super().__init__('calib_ultrasonic')
        self.distance_cm, self.samples = distance_cm, samples
        self.readings = []           # cm; inf/nan = no echo
        self.create_subscription(Range, '/ultrasonic', self.on_range, 10)
        self.t0 = self.get_clock().now().nanoseconds / 1e9
        wall_timer(self, 0.1, self.tick)
        self.get_logger().info(f"ULTRASONIC  block at {distance_cm:.0f} cm - taking {samples} readings")

    def on_range(self, msg: Range):
        r = msg.range
        ok = math.isfinite(r) and msg.min_range <= r <= msg.max_range
        self.readings.append(r * 100.0 if ok else float('nan'))

    def tick(self):
        waited = self.get_clock().now().nanoseconds / 1e9 - self.t0
        if len(self.readings) < self.samples and waited < TIMEOUT_S:
            return
        if not self.readings:
            self.get_logger().error('no /ultrasonic readings - is the car (or the sim) up?')
            raise SystemExit
        good = sorted(r for r in self.readings if not math.isnan(r))
        dropouts = len(self.readings) - len(good)
        if not good:
            self.get_logger().error(f'{dropouts} readings, no echo in any')
            log.write(log.where(self), 'ultrasonic', f'{self.distance_cm:.0f} cm', None, None,
                      self.distance_cm, 'cm', f'no echo in {dropouts} readings')
            raise SystemExit
        median = statistics.median(good)
        q1, q3 = good[len(good) // 4], good[(3 * len(good)) // 4]
        notes = (f'spread (middle half) {q1:.1f}-{q3:.1f} cm; min {good[0]:.1f} max {good[-1]:.1f}; '
                 f'dropouts {dropouts}/{len(self.readings)}')
        log.write(log.where(self), 'ultrasonic', f'{self.distance_cm:.0f} cm', None, median,
                  self.distance_cm, 'cm', notes)
        raise SystemExit


def main(args=None):
    ap = argparse.ArgumentParser(description='Ultrasonic reading vs tape: calib ultrasonic CM')
    ap.add_argument('distance', type=float, help='cm, tape from the sensor face to the block')
    ap.add_argument('--samples', type=int, default=50, help='readings to take (default 50)')
    a, ros_args = ap.parse_known_args()
    try:
        run(lambda: Ultrasonic(a.distance, a.samples), args=ros_args)
    except SystemExit:
        pass


if __name__ == '__main__':
    main()
