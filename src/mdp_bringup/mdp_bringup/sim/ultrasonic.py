"""Gazebo's ultrasonic stand-in -> /ultrasonic, like the real car (part of sim_helpers).

The URDF gives the sim car a narrow lidar fan at ultrasonic_link (7 beams over
the HC-SR04's ~15 deg cone). The nearest hit becomes a sensor_msgs/Range with
the same fields serial_bridge_node publishes for the real sensor, so
task2_runner reads one topic in sim and on the car.
"""
import math

from sensor_msgs.msg import LaserScan, Range


class Ultrasonic:
    def __init__(self, node):
        self.pub = node.create_publisher(Range, '/ultrasonic', 10)
        node.create_subscription(LaserScan, '/ultrasonic_scan', self.on_scan, 10)

    def on_scan(self, scan: LaserScan):
        hits = [r for r in scan.ranges if math.isfinite(r) and scan.range_min <= r <= scan.range_max]
        msg = Range(radiation_type=Range.ULTRASOUND, field_of_view=0.26, min_range=0.02, max_range=4.0)
        msg.header.stamp = scan.header.stamp
        msg.header.frame_id = 'base_link'      # as the real bridge sends it
        msg.range = min(hits) if hits else float('inf')   # nothing in range: +inf (REP-117)
        self.pub.publish(msg)

