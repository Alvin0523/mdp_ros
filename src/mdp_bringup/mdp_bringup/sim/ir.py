"""Gazebo's IR stand-ins -> /ir and /ir2, like the real car (part of sim_helpers).

The URDF gives each Sharp GP2Y0A21YK a 3-beam lidar at ir_link / ir2_link. The
nearest hit becomes a sensor_msgs/Range with the fields serial_bridge_node
publishes, and the real sensor's limits: the STM32 clamps to 10-80 cm (closer
than 10 cm the Sharp reads wrong, so the sim reads 10 cm there too), and the
reading is noisy - about 1% of the distance (a guess - TODO from calib).
"""
import math
import random

from sensor_msgs.msg import LaserScan, Range

MIN_M, MAX_M = 0.10, 0.80     # ir.c IR_DISTANCE_MIN_CM / IR_DISTANCE_MAX_CM
NOISE = 0.01                  # sigma, fraction of the distance


class SharpIr:
    def __init__(self, node, name):
        self.pub = node.create_publisher(Range, f'/{name}', 10)
        node.create_subscription(LaserScan, f'/{name}_scan', self.on_scan, 10)

    def on_scan(self, scan: LaserScan):
        hits = [r for r in scan.ranges if math.isfinite(r) and scan.range_min <= r <= scan.range_max]
        r = min(hits) * (1.0 + random.gauss(0.0, NOISE)) if hits else MAX_M
        msg = Range(radiation_type=Range.INFRARED, field_of_view=0.1, min_range=MIN_M, max_range=MAX_M)
        msg.header.stamp = scan.header.stamp
        msg.header.frame_id = 'base_link'      # as the real bridge sends it
        msg.range = min(MAX_M, max(MIN_M, r))
        self.pub.publish(msg)
