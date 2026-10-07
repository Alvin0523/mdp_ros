"""`pixi run capture`: record the camera as an MP4, for Roboflow (it samples the
frames to label from a video).

    pixi run capture                          record until Ctrl+C
    pixi run capture --scan                   only while a task 1 run is at a scan stop
    pixi run capture --from-bag bags/rosbag2_...   a recorded run -> MP4
    pixi run capture --topic /camera/image_raw     sim (Gazebo's raw camera)

Run on the laptop next to `pixi run pi` (or in sim). It reads the camera's JPEG copy
(/image_raw/compressed - exactly what YOLO on the laptop sees) and pipes the frames
as they came into ffmpeg: H.264, CRF 17 (near lossless), the camera's frame rate.
Saved to datasets/videos/<date_time>[_scan].mp4 (not in git: large).
--scan keeps only the frames taken while /run_status is PAUSE_FOR_SCAN - the views
YOLO really has to read.
"""
import argparse
import os
import subprocess
import sys
import time

import rclpy
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

FPS = 15          # the Pi camera's rate (vision.yaml frame_rate); a sim run differs a little


def start_ffmpeg(path, fps):
    """ffmpeg reading JPEG frames on stdin, writing H.264 near lossless."""
    return subprocess.Popen(
        ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'image2pipe', '-c:v', 'mjpeg', '-framerate', str(fps),
         '-i', '-', '-c:v', 'libx264', '-crf', '17', '-preset', 'medium', '-pix_fmt', 'yuv420p', path],
        stdin=subprocess.PIPE)


def to_jpeg(msg, raw):
    """A frame as JPEG bytes: the compressed copy as it is, a raw (sim) frame encoded once."""
    if not raw:
        return bytes(msg.data)
    import cv2
    from cv_bridge import CvBridge
    ok, jpg = cv2.imencode('.jpg', CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8'),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    return jpg.tobytes() if ok else None


def from_bag(bag, topic, path, fps):
    """Every frame of `topic` in a recorded bag into one MP4."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in types:
        sys.exit(f"no {topic} in {bag} - topics with images: "
                 f"{', '.join(n for n, t in types.items() if 'Image' in t) or 'none'}")
    raw = types[topic] == 'sensor_msgs/msg/Image'
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    ff, n = start_ffmpeg(path, fps), 0
    while reader.has_next():
        _, data, _ = reader.read_next()
        jpg = to_jpeg(deserialize_message(data, Image if raw else CompressedImage), raw)
        if jpg:
            ff.stdin.write(jpg)
            n += 1
    ff.stdin.close()
    ff.wait()
    return n


def main():
    ap = argparse.ArgumentParser(description='Record the camera as an MP4 for Roboflow: pixi run capture')
    ap.add_argument('--topic', default='/image_raw/compressed',
                    help='camera topic (default /image_raw/compressed; sim: /camera/image_raw)')
    ap.add_argument('--scan', action='store_true', help='only frames at task 1 scan stops (PAUSE_FOR_SCAN)')
    ap.add_argument('--from-bag', default=None, help='turn a recorded bag into an MP4 instead')
    ap.add_argument('--fps', type=float, default=FPS, help=f'video frame rate (default {FPS}, the camera\'s)')
    ap.add_argument('--out', default=None, help='output .mp4 (default datasets/videos/<date_time>.mp4)')
    a, ros_args = ap.parse_known_args()

    root = os.environ.get('PIXI_PROJECT_ROOT', os.getcwd())
    name = (os.path.basename(a.from_bag.rstrip('/')) if a.from_bag else time.strftime('%Y%m%d_%H%M%S')) \
        + ('_scan' if a.scan else '')
    path = a.out or os.path.join(root, 'datasets', 'videos', name + '.mp4')
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if a.from_bag:
        n = from_bag(a.from_bag, a.topic, path, a.fps)
        print(f"{n} frames -> {path} ({n / a.fps:.0f} s)")
        return 0

    rclpy.init(args=ros_args)
    node = rclpy.create_node('capture')
    raw = not a.topic.endswith('/compressed')
    state = {'run': ''}
    ff, count = start_ffmpeg(path, a.fps), [0]

    def on_frame(msg):
        if a.scan and state['run'] != 'PAUSE_FOR_SCAN':
            return
        jpg = to_jpeg(msg, raw)
        if jpg:
            ff.stdin.write(jpg)
            count[0] += 1

    qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
    node.create_subscription(Image if raw else CompressedImage, a.topic, on_frame, qos)
    if a.scan:
        from mdp_interfaces.msg import RunStatus
        node.create_subscription(RunStatus, '/run_status', lambda m: state.update(run=m.state), 10)
    print(f"recording {a.topic}{' at scan stops only' if a.scan else ''} -> {path}  (Ctrl+C to stop)")
    last_report = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() - last_report >= 5.0:
                last_report = time.monotonic()
                print(f"  {count[0]} frames ({count[0] / a.fps:.0f} s of video)")
    except KeyboardInterrupt:
        pass
    finally:
        ff.stdin.close()
        ff.wait()
        node.destroy_node()
        rclpy.try_shutdown()
        print(f"\n{count[0]} frames -> {path} ({count[0] / a.fps:.0f} s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
