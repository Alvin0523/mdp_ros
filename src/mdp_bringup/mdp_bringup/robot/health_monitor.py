"""One-glance system health on /diagnostics (diagnostic_msgs/DiagnosticArray).

Publishes once a second, one status per check, named "mdp/<check>":

    STM32 link     /hardware_bridge/link_ok  (real) - simulated in sim
    Motor switch   /estop                    (real) - simulated in sim
    Tablet link    /bluetooth_bridge/link_ok
    Pi             /pi/status: CPU, memory, temperature, throttling   (real)
    Wheel odometry /ackermann_steering_controller/odometry  data in the last 1 s
    IMU            /imu/data                 data in the last 1 s
    EKF            /odometry/filtered        data in the last 1 s
    Camera         camera_topic              has a publisher   (vision only)
    YOLO           yolo_detector node alive                    (vision only)
    Task runner    /run_status: state, plan, reset   (task 1 / 2; none in task 0)

Up/down only, kept cheap: the sensor topics are taken as raw bytes (never
deserialized), and the camera images are not received at all - its publisher
exits when rpicam-vid stops, so a publisher on the topic means the camera is up.
Topics only, no hardware: with a laptop (`pixi run laptop`) it runs there, off the Pi.

Level: OK / WARN / ERROR, or STALE when a topic has never been seen. View it in
Foxglove's "Diagnostics - Summary" / "Diagnostics - Detail" panels, or with
`ros2 topic echo /diagnostics`. The EKF and controller_manager publish their
own /diagnostics entries too; they show up alongside these.
"""
import time

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from rclpy.node import Node
from mdp_interfaces.msg import PiStatus, RunStatus
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
from mdp_bringup.utils.run import run, wall_timer

OK, WARN, ERROR, STALE = (DiagnosticStatus.OK, DiagnosticStatus.WARN,
                          DiagnosticStatus.ERROR, DiagnosticStatus.STALE)


class Latest:
    """Last value of a topic and when it arrived."""

    def __init__(self):
        self.value = None
        self.at = None

    def set(self, value):
        self.value = value
        self.at = time.monotonic()

    def age(self) -> float:
        return float('inf') if self.at is None else time.monotonic() - self.at


class HealthMonitor(Node):
    def __init__(self):
        super().__init__('health_monitor')
        self.sim = self.declare_parameter('sim', False).value
        self.vision = self.declare_parameter('vision', False).value
        self.camera_topic = self.declare_parameter('camera_topic', '/image_raw').value

        self.pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)

        self.stm32 = Latest()
        self.estop = Latest()
        self.tablet = Latest()
        self.create_subscription(Bool, '/hardware_bridge/link_ok', lambda m: self.stm32.set(m.data), 10)
        self.create_subscription(Bool, '/estop', lambda m: self.estop.set(m.data), 10)
        self.create_subscription(Bool, '/bluetooth_bridge/link_ok', lambda m: self.tablet.set(m.data), 10)

        # name: (last arrival, topic); raw=True: bytes only, nothing deserialized
        self.flows = {
            'Wheel odometry': (Latest(), '/ackermann_steering_controller/odometry'),
            'IMU': (Latest(), '/imu/data'),
            'EKF': (Latest(), '/odometry/filtered'),
        }
        for (seen, topic), msg_type in zip(self.flows.values(), (Odometry, Imu, Odometry)):
            self.create_subscription(msg_type, topic, seen.set, qos_profile_sensor_data, raw=True)

        self.run = Latest()
        self.create_subscription(RunStatus, '/run_status', self.run.set, 10)
        self.pi = Latest()
        if not self.sim:
            self.create_subscription(PiStatus, '/pi/status', self.pi.set, 10)

        wall_timer(self, 1.0, self.publish)

    @staticmethod
    def status(name: str, level, message: str, **values) -> DiagnosticStatus:
        s = DiagnosticStatus()
        s.name = f'mdp/{name}'
        s.hardware_id = 'mdp'
        s.level = level
        s.message = message
        s.values = [KeyValue(key=k, value=str(v)) for k, v in values.items()]
        return s

    def link(self, name: str, latest: Latest, ok_msg: str, bad_level, bad_msg: str, invert=False):
        if latest.at is None:
            return self.status(name, STALE, 'no messages yet')
        if latest.age() > 3.0:
            return self.status(name, ERROR, f'silent for {latest.age():.0f} s')
        good = (not latest.value) if invert else bool(latest.value)
        return self.status(name, OK if good else bad_level, ok_msg if good else bad_msg)

    def pi_check(self) -> DiagnosticStatus:
        """The Pi's load (pi_status). Hot = close to 80 C, where the Pi 4 starts
        slowing itself down (this one ran at ~75 C and had throttled, 2026-10-02)."""
        p = self.pi.value
        if p is None:
            return self.status('Pi', STALE, 'no /pi/status')
        if self.pi.age() > 3.0:
            return self.status('Pi', ERROR, f'silent for {self.pi.age():.0f} s')
        text = f'CPU {p.cpu_percent:.0f}%, mem {p.mem_percent:.0f}%, {p.temp_c:.0f} C'
        values = dict(cpu=f'{p.cpu_percent:.0f}', mem=f'{p.mem_percent:.0f}', temp=f'{p.temp_c:.1f}')
        if p.throttled or p.undervoltage:
            why = 'under-voltage' if p.undervoltage else 'throttled (hot)'
            return self.status('Pi', ERROR, f'{why} - {text}', **values)
        if p.temp_c >= 78.0 or p.cpu_percent >= 90.0:
            return self.status('Pi', WARN, f'busy/hot - {text}', **values)
        return self.status('Pi', OK, text, **values)

    def publish(self):
        out = DiagnosticArray()
        out.header.stamp = self.get_clock().now().to_msg()
        st = out.status

        if self.sim:
            st.append(self.status('STM32 link', OK, 'simulated (no STM32)'))
            st.append(self.status('Motor switch', OK, 'simulated'))
        else:
            st.append(self.link('STM32 link', self.stm32, 'connected', ERROR, 'link lost'))
            st.append(self.link('Motor switch', self.estop, 'ON (ready)', WARN,
                                'OFF - motors held at 0', invert=True))
        st.append(self.link('Tablet link', self.tablet, 'connected', WARN, 'not connected'))
        if not self.sim:
            st.append(self.pi_check())

        for name, (seen, topic) in self.flows.items():
            if seen.at is None:
                st.append(self.status(name, STALE, f'no messages on {topic}', topic=topic))
            elif seen.age() > 1.0:
                st.append(self.status(name, ERROR, f'no data for {seen.age():.0f} s on {topic}', topic=topic))
            else:
                st.append(self.status(name, OK, 'running', topic=topic))

        if self.vision:
            up = self.count_publishers(self.camera_topic) > 0
            st.append(self.status('Camera', OK if up else ERROR, 'up' if up else 'not publishing',
                                  topic=self.camera_topic))
            alive = 'yolo_detector' in self.get_node_names()
            st.append(self.status('YOLO', OK if alive else ERROR, 'running' if alive else 'not running'))

        m = self.run.value
        if m is None:
            st.append(self.status('Task runner', STALE, 'no /run_status (task 0, or no runner yet)'))
        elif self.run.age() > 3.0:
            st.append(self.status('Task runner', ERROR, f'silent for {self.run.age():.0f} s'))
        else:
            level = WARN if m.state == 'STOPPED' else OK
            text = f'{m.state}, plan {m.plan_state}, reset {"DONE" if m.reset_done else "WAITING"}'
            if m.obstacle:
                text += f', obstacle {m.obstacle} (leg {m.leg}/{m.leg_count})'
            st.append(self.status('Task runner', level, text, state=m.state, plan=m.plan_state,
                                  reset=m.reset_done, obstacle=m.obstacle))

        self.pub.publish(out)


def main():
    run(HealthMonitor)


if __name__ == '__main__':
    main()
