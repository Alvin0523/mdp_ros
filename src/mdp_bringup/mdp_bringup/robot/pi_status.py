"""The Pi's own load on /pi/status (mdp_interfaces/PiStatus), once a second.

Runs on the Pi (real car, role:=pi / solo): with a laptop the monitors run
there and cannot read the Pi's /proc. CPU per core from /proc/stat, memory
from /proc/meminfo, temperature from the thermal zone, throttling from
`vcgencmd get_throttled` (~2 ms a call, measured 2026-10-02). health_monitor
turns it into the "Pi" line on /diagnostics; Foxglove plots it.
"""
import subprocess

from mdp_interfaces.msg import PiStatus
from rclpy.node import Node

from mdp_bringup.utils.run import run, wall_timer

TEMP_FILE = '/sys/class/thermal/thermal_zone0/temp'
# get_throttled bits: 0 under-voltage now, 2 throttled now, 18 throttled since boot
UNDERVOLT_NOW, THROTTLED_NOW, THROTTLED_EVER = 1 << 0, 1 << 2, 1 << 18


def cpu_times():
    """[(busy, total)] for all cores, then each core, from /proc/stat."""
    out = []
    with open('/proc/stat') as f:
        for line in f:
            if not line.startswith('cpu'):
                break
            v = [int(x) for x in line.split()[1:]]
            idle = v[3] + v[4]    # idle + iowait
            out.append((sum(v) - idle, sum(v)))
    return out


def mem_percent() -> float:
    info = {}
    with open('/proc/meminfo') as f:
        for line in f:
            key, value = line.split(':')
            info[key] = int(value.split()[0])
    return 100.0 * (1.0 - info['MemAvailable'] / info['MemTotal'])


class PiStatusNode(Node):
    def __init__(self):
        super().__init__('pi_status')
        self.pub = self.create_publisher(PiStatus, '/pi/status', 10)
        self.last = cpu_times()
        wall_timer(self, 1.0, self.publish)

    def publish(self):
        now = cpu_times()
        load = [100.0 * (b1 - b0) / max(1, t1 - t0) for (b0, t0), (b1, t1) in zip(self.last, now)]
        self.last = now
        msg = PiStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.cpu_percent, msg.cpu_cores = load[0], load[1:]
        msg.mem_percent = mem_percent()
        with open(TEMP_FILE) as f:
            msg.temp_c = int(f.read()) / 1000.0
        try:
            out = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True, text=True, timeout=1.0)
            bits = int(out.stdout.strip().split('=')[1], 16)
            msg.throttled = bool(bits & THROTTLED_NOW)
            msg.throttled_since_boot = bool(bits & THROTTLED_EVER)
            msg.undervoltage = bool(bits & UNDERVOLT_NOW)
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            pass   # not a Pi / no vcgencmd: the rest is still right
        self.pub.publish(msg)


def main(args=None):
    run(PiStatusNode, args=args)


if __name__ == '__main__':
    main()
