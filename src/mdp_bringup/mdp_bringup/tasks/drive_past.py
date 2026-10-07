"""Task 1: reading a block's image while driving past it (task1_runner). The
block being driven to, when the IR rule allows (task1_runner.pass_ok), is read on
the last stretch into its stop and passed without stopping; any other block read
on the way is only logged (SEEN) - the car still stops there.

Every YOLO frame while driving brings its boxes (/yolo_detections). A box counts
for a block only when it is where that block's image face must appear in the
image, and the right size for its distance:

  candidate face  not read yet; facing the camera (within MAX_ANGLE of square);
                  RANGE_M away; its projected centre in the middle band of the
                  image (BAND); no other block between the camera and it
  matching box    centre within POS_TOL_U_PX across / POS_TOL_V_PX up of the
                  face's projected centre, its
                  longer side SIZE_RATIO x the printed symbol's projected size
                  (SYMBOL_M square; YOLO boxes the symbol, a '1' is narrow but
                  tall), not cut by the image edge, confidence >= MIN_CONF
  unique          a box matching two faces, or a face matched by two boxes:
                  that frame is not used for them
  read            the same ID in READS frames spread over SPREAD_M of travel,
                  and no other ID matched that face at all (else: conflict,
                  never read on the way)

The pose is the estimate at the frame's own time, not when its result arrives.
Pure geometry, no ROS (tests: test/test_drive_past.py). Map frame, metres.
"""
import math

BLOCK_HALF = 0.05
FACE_Z = 0.05                       # the image's centre: half way up the 10 cm block
SYMBOL_M = 0.061                    # the printed symbol, square (models/symbols, the real print)
MAX_ANGLE = math.radians(30.0)
RANGE_M = (0.20, 0.60)
BAND = 0.25                         # the projected centre within the middle 50 % across
POS_TOL_U_PX = 80.0                 # ~4.5 cm along the face at 30 cm: the pose drifts 2-9 cm
                                    # along a leg between IR fixes (car, 2026-10-07)
POS_TOL_V_PX = 40.0                 # up/down: the camera's height is fixed, the pose does not move it
SIZE_RATIO = (0.6, 1.5)
EDGE_PX = 3.0
MIN_CONF = 0.5
READS = 3
SPREAD_M = 0.05


class Camera:
    """The camera on the car: (x, y, z, yaw) of camera_link on base_link (it looks
    along its +x), the image size and horizontal field of view."""

    def __init__(self, x, y, z, yaw, width=640, height=480, hfov=1.089):
        self.x, self.y, self.z, self.yaw = x, y, z, yaw
        self.w, self.h = width, height
        self.f = (width / 2.0) / math.tan(hfov / 2.0)      # px, square pixels

    def world(self, pose):
        """(x, y, yaw) of the camera in the map with the car at `pose`."""
        px, py, pth = pose
        c, s = math.cos(pth), math.sin(pth)
        return px + c * self.x - s * self.y, py + s * self.x + c * self.y, pth + self.yaw

    def project(self, pose, point):
        """(u, v, depth) of a map point (x, y, z) in the image, or None behind it."""
        cx, cy, cth = self.world(pose)
        dx, dy = point[0] - cx, point[1] - cy
        fwd = math.cos(cth) * dx + math.sin(cth) * dy
        left = -math.sin(cth) * dx + math.cos(cth) * dy
        if fwd <= 0.02:
            return None
        return self.w / 2.0 - self.f * left / fwd, self.h / 2.0 - self.f * (point[2] - self.z) / fwd, fwd


def _segment_hits_block(a, b, block):
    """Does the 2-D segment a-b pass through the 10 cm square `block` (x, y)?"""
    (ax, ay), (bx, by) = a, b
    x0, x1, y0, y1 = block[0] - BLOCK_HALF, block[0] + BLOCK_HALF, block[1] - BLOCK_HALF, block[1] + BLOCK_HALF
    t0, t1 = 0.0, 1.0
    for p, q in ((-(bx - ax), ax - x0), (bx - ax, x1 - ax), (-(by - ay), ay - y0), (by - ay, y1 - ay)):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def candidates(camera, pose, blocks, faces, unread):
    """Faces that could be in this frame: [(index, u, v, symbol size px)].
    blocks: (x, y) centres; faces: (nx, ny) outward normal of each image face."""
    cx, cy, _ = camera.world(pose)
    out = []
    for i in unread:
        (bx, by), (nx, ny) = blocks[i], faces[i]
        fx, fy = bx + nx * BLOCK_HALF, by + ny * BLOCK_HALF
        dx, dy = cx - fx, cy - fy
        dist = math.hypot(dx, dy)
        if not RANGE_M[0] <= dist <= RANGE_M[1] or (nx * dx + ny * dy) / dist < math.cos(MAX_ANGLE):
            continue
        centre = camera.project(pose, (fx, fy, FACE_Z))
        if centre is None or not BAND * camera.w <= centre[0] <= (1.0 - BAND) * camera.w:
            continue
        if any(j != i and _segment_hits_block((cx, cy), (fx, fy), blocks[j]) for j in range(len(blocks))):
            continue
        tx, ty = -ny, nx
        a = camera.project(pose, (fx + tx * SYMBOL_M / 2, fy + ty * SYMBOL_M / 2, FACE_Z))
        b = camera.project(pose, (fx - tx * SYMBOL_M / 2, fy - ty * SYMBOL_M / 2, FACE_Z))
        if a is None or b is None:
            continue
        out.append((i, centre[0], centre[1], abs(a[0] - b[0])))
    return out


def match(cands, boxes, width, height):
    """Unique box <-> face pairs: [(index, target_id)]. boxes: (id, conf, x1, y1, x2, y2)."""
    pairs = []
    for tid, conf, x1, y1, x2, y2 in boxes:
        if conf < MIN_CONF or x1 <= EDGE_PX or y1 <= EDGE_PX or x2 >= width - EDGE_PX or y2 >= height - EDGE_PX:
            continue
        u, v, size = (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(x2 - x1, y2 - y1)
        ok = [i for i, cu, cv, cw in cands
              if abs(u - cu) <= POS_TOL_U_PX and abs(v - cv) <= POS_TOL_V_PX
              and SIZE_RATIO[0] * cw <= size <= SIZE_RATIO[1] * cw]
        if len(ok) == 1:
            pairs.append((ok[0], tid))
    faces = [i for i, _ in pairs]
    return [(i, tid) for i, tid in pairs if faces.count(i) == 1]


class DrivePast:
    def __init__(self, camera):
        self.camera = camera
        self.reset([], [])

    def reset(self, blocks, faces):
        """A new run: blocks (x, y) and their image faces' normals (nx, ny)."""
        self.blocks, self.faces = blocks, faces
        self.seen = {}            # index -> [(id, x, y)] matched frames
        self.read = {}            # index -> id, read on the way
        self.conflict = set()

    def on_frame(self, pose, boxes, width, height, unread):
        """One YOLO frame with the pose estimate at its time. Returns the blocks
        newly read on the way: [(index, id)]."""
        new = []
        cands = candidates(self.camera, pose, self.blocks, self.faces, [i for i in unread if i not in self.read])
        for i, tid in match(cands, boxes, width, height):
            hits = self.seen.setdefault(i, [])
            hits.append((tid, pose[0], pose[1]))
            if len({h[0] for h in hits}) > 1:
                self.conflict.add(i)
            if i in self.conflict or len(hits) < READS:
                continue
            spread = max(math.hypot(a[1] - b[1], a[2] - b[2]) for a in hits for b in hits)
            if spread >= SPREAD_M:
                self.read[i] = tid
                new.append((i, tid))
        return new
