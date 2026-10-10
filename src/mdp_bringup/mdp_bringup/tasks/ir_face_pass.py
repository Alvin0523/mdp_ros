"""Task 2: correct the car's position while it drives past a face, with the side IR
looking at it (task2_runner, IRs at the front wheels - one each side). Map frame.

A face runs along the course (x0..x1) at y = face_y and looks toward `out` (+1: +y).
While the IR's beam is on it, the reading against the distance the pose expects
gives how far OUT the car really is (median of the readings); the beam jumping
onto / off the face gives where ALONG it the car is (the corner's place is known).
At speed one reading is ~2 cm of travel (25 Hz at 0.5 m/s), so an edge lands
within about half that. Pure geometry, no ROS (tests: test/test_ir_face_pass.py).
"""
import math
from statistics import median

IR_MIN, IR_MAX = 0.06, 0.40    # m, readings used (task 2 passes closer than task 1's 10 cm)
HIT_TOL = 0.08                 # m, a reading this close to the expected range is "the face"
MAX_INCIDENCE = math.radians(25.0)   # beam this far off square to the face: not used
EDGE_INSET = 0.01              # m, across readings only this far inside the corners
IR_LAG_S = 0.0                 # s, the reading's delay (sim: none; real Sharp ~0.04 - to measure)
MIN_READINGS = 3
MAX_MAD = 0.01                 # m, across readings disagreeing more: not used
EDGE_SPREAD = 0.03             # m, edges disagreeing more: not used
MAX_FIX = 0.10                 # m, a bigger correction is a misreading: not used


class FacePass:
    def __init__(self, label, x0, x1, face_y, out, sensor):
        self.label = label
        self.x0, self.x1, self.face_y, self.out = min(x0, x1), max(x0, x1), face_y, out
        self.sensor = sensor               # (x, y, yaw) of the IR on base_link
        self.hits, self.edges = [], []     # (along, how much further out) per reading on the face
        self.prev = None                   # (on the face?, along) of the previous reading
        self.seen = False

    def beam(self, pose):
        """(expected range, along-face x of the hit) for the IR at `pose`, or None
        when it does not point at the face's line square enough."""
        x, y, yaw = pose
        sx, sy, syaw = self.sensor
        c, s = math.cos(yaw), math.sin(yaw)
        px, py = x + c * sx - s * sy, y + s * sx + c * sy
        b = yaw + syaw
        toward = -self.out * math.sin(b)            # cos of the incidence
        if toward < math.cos(MAX_INCIDENCE):
            return None
        t = (py - self.face_y) * self.out / toward
        if t <= 0.0:
            return None
        return t, px + t * math.cos(b)

    def on_reading(self, reading, pose, speed):
        bm = self.beam(pose)
        if bm is None:
            self.prev = None
            return
        expected, along = bm
        # On the face by the READING (the pose may be cm off along it - that is what
        # the edges measure); only near the face, so a wall further on is not it.
        on = (self.x0 - MAX_FIX <= along <= self.x1 + MAX_FIX and IR_MIN < reading < IR_MAX
              and abs(reading - expected) < HIT_TOL)
        if on:
            self.seen = True
            # Further than expected -> the car is further OUT than the pose says.
            self.hits.append((along, (reading - expected) * (-self.out * math.sin(pose[2] + self.sensor[2]))))
        if self.prev is not None and self.prev[0] != on:
            moved = along - self.prev[1]
            if abs(moved) > 1e-4:
                direction = math.copysign(1.0, moved)
                crossing = 0.5 * (along + self.prev[1]) - direction * speed * IR_LAG_S
                # Coming on moving +x: in over the x0 corner; going off moving +x: out over x1.
                corner = (self.x0 if direction > 0 else self.x1) if on else (self.x1 if direction > 0 else self.x0)
                if abs(crossing - corner) < MAX_FIX:
                    self.edges.append(crossing - corner)
        self.prev = (on, along)

    def correction(self):
        """(dx, dy, note) to add to the map pose; (0, 0, why not) when unsure."""
        notes, dx, dy = [], 0.0, 0.0
        # Both corners: one edge alone at speed can be a reading off (sim 2026-10-10: 7-10 cm).
        if len(self.edges) >= 2 and max(self.edges) - min(self.edges) <= EDGE_SPREAD:
            dx = -sum(self.edges) / len(self.edges)
            notes.append(f'along {dx * 100:+.1f} ({len(self.edges)} edge{"s" if len(self.edges) > 1 else ""})')
        else:
            notes.append('along ? (' + ('edges disagree' if len(self.edges) > 1 else
                                        f'{len(self.edges)} edge') + ')')
        # Inside the corners by where the edges put the car: with the pose cm off along,
        # the pose alone left 2 of the ~4 readings (sim offline 2026-10-10: 'out' refused).
        gaps = [g for a, g in self.hits if self.x0 + EDGE_INSET <= a + dx <= self.x1 - EDGE_INSET]
        if len(gaps) >= MIN_READINGS:
            med = median(gaps)
            if median(abs(g - med) for g in gaps) <= MAX_MAD:
                dy = self.out * med
                notes.append(f'out {med * 100:+.1f} cm ({len(gaps)} readings)')
            else:
                notes.append('out ? (noisy)')
        else:
            notes.append(f'out ? ({len(gaps)} readings)')
        if math.hypot(dx, dy) > MAX_FIX:
            return 0.0, 0.0, f'{math.hypot(dx, dy) * 100:.0f} cm too big ({" ".join(notes)})'
        return dx, dy, ' '.join(notes)
