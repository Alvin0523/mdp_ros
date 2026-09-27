"""One-glance system health on /diagnostics (diagnostic_msgs/DiagnosticArray).

Publishes once a second, one status per check, named "mdp/<check>":

    STM32 link     /hardware_bridge/link_ok  (real) - simulated in sim
    Motor switch   /estop                    (real) - simulated in sim
    Tablet link    /bluetooth_bridge/link_ok
    Wheel odometry /ackermann_steering_controller/odometry  rate
    IMU            /imu/data                 rate
    EKF            /odometry/filtered        rate
    Camera         camera_topic              rate   (vision only)
    YOLO           yolo_detector node alive          (vision only)
    Task runner    /run_status: state, plan, reset   (task 1 only)

Level: OK / WARN / ERROR, or STALE when a topic has never been seen. View it in
Foxglove's "Diagnostics - Summary" / "Diagnostics - Detail" panels, or with
`ros2 topic echo /diagnostics`. The EKF and controller_manager publish their
own /diagnostics entries too; they show up alongside these.
"""
import time
from collections import deque

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Bool
from mdp_bringup.utils.run import run, wall_timer

OK, WARN, ERROR, STALE = (DiagnosticStatus.OK, DiagnosticStatus.WARN,
                          DiagnosticStatus.ERROR, DiagnosticStatus.STALE)


class Rate:
    """Messages per second over the last `window` seconds (wall clock)."""

    def __init__(self, window: float = 2.0):
        self.window = window
        self.stamps = deque()
        self.ever = False

    def tick(self, *_):
        self.ever = True
        self.stamps.append(time.monotonic())

    def hz(self) -> float:
        now = time.monotonic()
        while self.stamps and now - self.stamps[0] > self.window:
            self.stamps.popleft()
        return len(self.stamps) / self.window


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
        self.task = str(self.declare_parameter('task', '0').value)
        camera_topic = self.declare_parameter('camera_topic', '/image_raw').value

        self.pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)

        self.stm32 = Latest()
        self.estop = Latest()
        self.tablet = Latest()
        self.create_subscription(Bool, '/hardware_bridge/link_ok', lambda m: self.stm32.set(m.data), 10)
        self.create_subscription(Bool, '/estop', lambda m: self.estop.set(m.data), 10)
        self.create_subscription(Bool, '/bluetooth_bridge/link_ok', lambda m: self.tablet.set(m.data), 10)

        self.rates = {
            'Wheel odometry': (Rate(), '/ackermann_steering_controller/odometry', 20.0),
            'IMU': (Rate(), '/imu/data', 20.0),
            'EKF': (Rate(), '/odometry/filtered', 10.0),
        }
        self.create_subscription(Odometry, '/ackermann_steering_controller/odometry',
                                 self.rates['Wheel odometry'][0].tick, qos_profile_sensor_data)
        self.create_subscription(Imu, '/imu/data', self.rates['IMU'][0].tick, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odometry/filtered', self.rates['EKF'][0].tick, 10)
        if self.vision:
            self.rates['Camera'] = (Rate(), camera_topic, 5.0)
            self.create_subscription(Image, camera_topic, self.rates['Camera'][0].tick, qos_profile_sensor_data)

        self.run = Latest()
        if self.task == '1':
            from mdp_interfaces.msg import RunStatus
            self.create_subscription(RunStatus, '/run_status', self.run.set, 10)

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

        for name, (rate, topic, min_hz) in self.rates.items():
            hz = rate.hz()
            if not rate.ever:
                st.append(self.status(name, STALE, f'no messages on {topic}', topic=topic))
            elif hz < 0.5:
                st.append(self.status(name, ERROR, f'no data (stopped) on {topic}', topic=topic, hz=0))
            elif hz < min_hz:
                st.append(self.status(name, WARN, f'{hz:.0f} Hz (expected >= {min_hz:.0f})',
                                      topic=topic, hz=f'{hz:.1f}'))
            else:
                st.append(self.status(name, OK, f'{hz:.0f} Hz', topic=topic, hz=f'{hz:.1f}'))

        if self.vision:
            alive = 'yolo_detector' in self.get_node_names()
            st.append(self.status('YOLO', OK if alive else ERROR, 'running' if alive else 'not running'))

        if self.task == '1':
            m = self.run.value
            if m is None:
                st.append(self.status('Task runner', STALE, 'no /run_status yet'))
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
