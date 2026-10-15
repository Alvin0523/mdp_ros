"""Tasks 1 and 2: a snapshot of what YOLO saw at every obstacle, and one picture of
them all - `pixi run laptop` runs it.

Keeps the newest annotated frame (/yolo_result/image_annotated) for each id
YOLO sent on /yolo_result. When the runner reports TARGET,<obstacle>,<id> to the
tablet (/bluetooth_tx), the newest frame of that id - the one the scan counted -
is saved with "OBSTACLE n -> X (ID id)" on top; UNKNOWN gets the newest frame.
A new folder each START: snapshots/<date_time>/obstacle_<n>.jpg and collage.jpg,
the collage rewritten after every obstacle (parameter `dir`).

Task 2: the runner names the frames itself on /task2_snapshot ({"obstacle": 1 | 2,
"stamp": the frame's time, "caption": ...}) - arrow 1's at START (a new folder,
snapshots/<date_time>_task2/), the one the bar's width came from at checkpoint 1.
The annotated frame with that stamp (or the newest) is saved the same way.
"""
import json
from collections import deque
import os
import time

import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from mdp_interfaces.msg import RunStatus
from mdp_bringup.utils import targets
from mdp_bringup.utils.run import run

DRIVING = ('NAVIGATING_TO_TARGET', 'PAUSE_FOR_SCAN')
TILE_W, TILE_H = 427, 320     # three across ~1280 px
COLS = 3
GREEN = (0, 255, 0)


def caption(obs, target_id) -> str:
    """'OBSTACLE 1 -> Z (ID 35)'; arrows by name (cv2 draws no unicode)."""
    try:
        tid = int(target_id)
    except ValueError:
        return f'OBSTACLE {obs} -> {target_id}'
    sym = targets.SYMBOLS.get(tid, str(tid))
    if not sym.isascii():
        sym = targets.NAMES[tid]
    return f'OBSTACLE {obs} -> {sym} (ID {tid})'


def put_caption(img, text, scale):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(img, (0, 0), (tw + 12, th + base + 12), GREEN, -1)
    cv2.putText(img, text, (6, th + 6), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2)


class ScanSnapshots(Node):
    def __init__(self):
        super().__init__('scan_snapshots')
        root = os.environ.get('PIXI_PROJECT_ROOT', os.getcwd())
        self.base = self.declare_parameter('dir', os.path.join(root, 'snapshots')).value
        self.frame = None              # newest annotated frame (JPEG bytes)
        self.by_id = {}                # id YOLO sent -> its newest annotated frame
        self.shots = {}                # obstacle number -> saved image
        self.folder = None
        self.state = None
        self.create_subscription(CompressedImage, '/yolo_result/image_annotated', self.on_frame, 5)
        self.create_subscription(String, '/yolo_result', self.on_result, 10)
        self.create_subscription(String, '/bluetooth_tx', self.on_tx, 10)
        self.create_subscription(RunStatus, '/run_status', self.on_status, 10)
        self.recent = deque(maxlen=90)  # (stamp s, JPEG) - ~6 s of annotated frames, for task 2
        self.create_subscription(String, '/task2_snapshot', self.on_task2, 10)

    def on_frame(self, msg: CompressedImage):
        self.frame = bytes(msg.data)
        self.recent.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, self.frame))

    def on_task2(self, msg: String):
        try:
            note = json.loads(msg.data)
            obs, stamp, text = str(note['obstacle']), float(note['stamp']), str(note['caption'])
        except (ValueError, KeyError, TypeError):
            return
        if obs == '1':
            self.folder, self.shots = None, {}
        near = min(self.recent, key=lambda f: abs(f[0] - stamp), default=None)
        jpeg = near[1] if near and abs(near[0] - stamp) < 0.2 else self.frame
        self.save(obs, jpeg, text, '_task2')

    def on_result(self, msg: String):
        # Published right after the frame it was picked on.
        if self.frame is not None:
            self.by_id[msg.data.strip()] = self.frame

    def on_status(self, msg: RunStatus):
        if self.state == 'WAITING_FOR_START' and msg.state in DRIVING:
            self.folder, self.shots, self.by_id = None, {}, {}
        self.state = msg.state

    def on_tx(self, msg: String):
        parts = msg.data.strip().split(',')
        if len(parts) != 3 or parts[0] != 'TARGET':
            return
        obs, target_id = parts[1], parts[2]
        jpeg = self.by_id.get(target_id, self.frame)
        self.by_id = {}
        self.save(obs, jpeg, caption(obs, target_id))

    def save(self, obs: str, jpeg, text: str, suffix: str = ''):
        if jpeg is None:
            self.get_logger().warn(f"SNAPSHOT  #{obs}: no YOLO image (is YOLO running?)")
            return
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        put_caption(img, text, 0.9 if len(text) < 30 else 0.6)
        if self.folder is None:
            self.folder = os.path.join(self.base, time.strftime('%Y%m%d_%H%M%S') + suffix)
            os.makedirs(self.folder, exist_ok=True)
        cv2.imwrite(os.path.join(self.folder, f'obstacle_{obs}.jpg'), img)
        self.shots[obs] = img
        path = os.path.join(self.folder, 'collage.jpg')
        cv2.imwrite(path, self.collage())
        self.get_logger().info(f"SNAPSHOT  #{obs} saved - {path}")

    def collage(self):
        order = sorted(self.shots, key=lambda o: (not o.isdigit(), int(o) if o.isdigit() else o))
        tiles = [cv2.resize(self.shots[o], (TILE_W, TILE_H)) for o in order]
        tiles += [np.full((TILE_H, TILE_W, 3), 30, np.uint8)] * (-len(tiles) % COLS)
        rows = [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
        return np.vstack(rows)


def main(args=None):
    run(ScanSnapshots, args=args)


if __name__ == '__main__':
    main()
