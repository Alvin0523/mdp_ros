"""Task 1: correct the car's position against the block at a scan stop, with the
two left IRs (task1_runner.ir_stop_step, parameter ir_pose_fix). The heading is
good (gyro); the position drifts a few cm per leg. Map frame, heading kept:

  across the face  The median IR distance vs the distance the pose expects.
  along the face   A flat face reads the same anywhere along it; what tells the
                   position is WHICH IRs see it. They sit 4 cm either side of
                   base_link and the face is 10 cm wide, so with both on it the
                   car is within +-1 cm of the block centre; with one, it is off
                   towards that one, and the runner creeps that way (seen())
                   until the other comes on - its beam has just crossed the
                   block's edge, whose place is known. Edges count only standing
                   still or creeping: at speed the reading lags and the edge
                   lands cm off (sim 2026-10-03). Beams are traced with the
                   heading, so stopping up to 20 deg off square is fine.

A fix that disagrees with itself or is over MAX_FIX_M is not used: a misreading
must never move the car further off than no fix. Pure geometry, no ROS (tests:
test/test_ir_pose_fix.py).
"""
import math
from statistics import median

BLOCK_HALF = 0.05        # m, the task 1 block is 10 x 10 cm
IR_MIN, IR_MAX = 0.10, 0.40   # m, readings used: the Sharp is good from 10 to ~40 cm
HIT_TOL = 0.10           # m, a reading this close to the expected gap is "the face" - the car may
                         # stop that far off it (2026-10-07: 7 cm closer, IR2 read 11.5 cm for
                         # 18.8, outside 5 cm - "not the block"). The floor / a wall reads 33 cm+
FAR_M = 0.15             # m, a reading this much past the expected gap is "past the block";
                         # in between: neither, no edge from it
MAX_INCIDENCE = math.radians(25.0)   # beam this far off square to the face: ignored
MAX_FIX_M = 0.08         # m, a bigger correction is a misreading - not used
EDGE_SPREAD_M = 0.025    # m, edge crossings disagreeing by more: not used
CROSS_MAD_M = 0.01       # m, the gaps' median absolute deviation above this: not used
CROSS_MIN_READINGS = 5   # gap readings needed (the Sharp is noisy: median of many)
STEADY_READINGS = 5      # seen() changes an IR's 'on the face' only after this many readings
                         # in a row agree: on the car 2026-10-07 the front IR, just past the
                         # block's edge, read a background jumping 23-80 cm, and ONE reading
                         # inside HIT_TOL counted it on the face - the creep stopped after 0 cm
# The runner (task1_runner.ir_stop_step): the fix waits IR_FIX_DELAY_S after
# stopping (the car settled), then for STEADY_READINGS per IR - at most IR_FIX_MAX_S
# (the STM32 reads the IRs at 5 Hz for now). A creep goes at most IR_CREEP_MAX_M.
EDGE_MAX_SPEED = 0.07    # m/s, an edge crossed faster is not used: the Sharp's reading lags, and
                         # rolling on after ARRIVED (~0.1 m/s) the edge landed ~1.5 cm off (car,
                         # 2026-10-07). The search creep goes 0.05.
IR_FIX_DELAY_S = 0.4
IR_FIX_MAX_S = 1.5
IR_CREEP_MAX_M = 0.10    # one IR on the block: creep at most this far for the other
IR_APPROACH_M = 0.15     # this close to the stop (last stretch of the leg): the IR that meets the
                         # block first on it - creep speed; both on - stop there
IR_APPROACH_YAW = math.radians(8.0)   # ...only this square to the stop's heading: still turning
                         # in, the IRs saw the block and it stopped 19 deg off - the camera
                         # looked past it (sim 2026-10-07)
IR_SEARCH_MAX_M = 0.15   # neither on it: search at most this far


class Face:
    """The side of the block the camera looks at: centre and outward normal."""

    def __init__(self, bx, by, nx, ny):
        self.nx, self.ny = nx, ny
        self.tx, self.ty = -ny, nx                      # along the face
        self.cx, self.cy = bx + BLOCK_HALF * nx, by + BLOCK_HALF * ny

    def beam(self, sensor, pose):
        """Where `sensor` (x, y, yaw on base_link) at map `pose` points at the
        face's line: (expected range, along-face position of the hit, cos of the
        incidence) - or None when it does not point at the face."""
        x, y, yaw = pose
        sx, sy, syaw = sensor
        c, s = math.cos(yaw), math.sin(yaw)
        px, py = x + c * sx - s * sy, y + s * sx + c * sy
        dx, dy = math.cos(yaw + syaw), math.sin(yaw + syaw)
        cos_inc = -(dx * self.nx + dy * self.ny)
        gap = (px - self.cx) * self.nx + (py - self.cy) * self.ny   # sensor's height above the face
        if cos_inc < math.cos(MAX_INCIDENCE) or gap <= 0.0:
            return None
        r = gap / cos_inc
        hx, hy = px + r * dx, py + r * dy
        return r, (hx - self.cx) * self.tx + (hy - self.cy) * self.ty, cos_inc


class IrPoseFix:
    def __init__(self, sensors):
        self.sensors = sensors            # name -> (x, y, yaw) on base_link
        self.face = None
        self.reset(None)

    def reset(self, face):
        """A new leg toward `face` (None: no fix)."""
        self.face = face
        self.last = {}                    # name -> (on the face?, along) of its previous reading
        self.pending = {}                 # name -> (new state, along before, along after): edge to confirm
        self.edges = []                   # along errors (estimate - truth) from edge crossings
        self.gaps = []                    # across errors from readings at the stop
        self.stopped = False
        self.count = {}                   # name -> readings since at_stop()
        self.steady = {}                  # name -> on the face, after STEADY_READINGS in a row
        self.seen_any = False             # an IR was steadily on the face since reset (the approach)
        self.streak = {}                  # name -> (on the face?, readings in a row)

    def along_offset(self, pose):
        """How far base_link at `pose` is along the face from its centre (m),
        and how much of a straight metre forward goes along it."""
        x, y, yaw = pose
        f = self.face
        return ((x - f.cx) * f.tx + (y - f.cy) * f.ty), math.cos(yaw) * f.tx + math.sin(yaw) * f.ty

    def seen(self):
        """Which IRs see the face now: 'both', 'none', or 'front' / 'rear'
        (only that one - the car is off towards it)."""
        on = [name for name in self.sensors if self.steady.get(name, False)]
        if len(on) == len(self.sensors):
            return 'both'
        if not on:
            return 'none'
        front = max(self.sensors, key=lambda n: self.sensors[n][0])
        return 'front' if front in on else 'rear'

    def at_stop(self):
        """The car has stopped at the checkpoint: gaps from now on."""
        self.stopped, self.gaps, self.count = True, [], {}

    def ready(self, per_sensor=STEADY_READINGS) -> bool:
        """Every IR has given per_sensor readings since at_stop() - the
        Sharp gives ~26 a second, but the STM32 reads it at 5 Hz for now."""
        return all(self.count.get(name, 0) >= per_sensor for name in self.sensors)

    def on_reading(self, name, reading, pose, speed=0.0):
        """One IR reading (m) with the pose estimate and the car's speed (m/s) at that moment."""
        if self.face is None or name not in self.sensors:
            return
        if self.stopped:
            self.count[name] = self.count.get(name, 0) + 1
        beam = self.face.beam(self.sensors[name], pose)
        if beam is None:
            self.last.pop(name, None)
            return
        expected, along, cos_inc = beam
        on_face = IR_MIN < reading < IR_MAX and abs(reading - expected) < HIT_TOL
        if not on_face and reading < expected + FAR_M:
            return                        # neither the face nor clearly past it
        # An edge: the beam's on/off-the-face changes AND stays changed for
        # the next reading too - a single noisy reading is not an edge. It
        # lies between the last reading before and the first one after.
        prev, pend = self.last.get(name), self.pending.get(name)
        if pend is not None:
            self.pending.pop(name)
            moved = pend[2] - pend[1]
            if on_face == pend[0] and abs(moved) > 1e-4 and speed <= EDGE_MAX_SPEED:
                crossing = 0.5 * (pend[1] + pend[2])
                # Which edge: the way the beam crossed it, not which side of the
                # centre the pose puts it - with the pose over 5 cm off that is
                # the wrong edge (sim 2026-10-07: 7 cm off, "fixed" 4 cm further
                # off). Coming on moving -along: in over the + edge; going off: out over the - edge.
                edge = math.copysign(BLOCK_HALF, -moved if on_face else moved)
                if abs(crossing - edge) < MAX_FIX_M:
                    self.edges.append(crossing - edge)
        elif prev is not None and prev[0] != on_face:
            self.pending[name] = (on_face, prev[1], along)
        self.last[name] = (on_face, along)
        state, n = self.streak.get(name, (None, 0))
        n = n + 1 if state == on_face else 1
        self.streak[name] = (on_face, n)
        if n >= STEADY_READINGS:
            self.steady[name] = on_face
            self.seen_any = self.seen_any or on_face
        if self.stopped and on_face and abs(along) < BLOCK_HALF:
            # Across: really `reading` along the beam, the pose says `expected`.
            self.gaps.append((reading - expected) * cos_inc)

    def correction(self):
        """(dx, dy, note) to add to the map pose; (0, 0, why not) when unsure."""
        if self.face is None:
            return 0.0, 0.0, 'no face'
        notes, along, across = [], 0.0, 0.0
        if self.edges and max(self.edges) - min(self.edges) <= EDGE_SPREAD_M:
            along = -sum(self.edges) / len(self.edges)
            notes.append(f'along {along * 100:+.1f} (edge)')
        elif not self.edges and self.seen() == 'both':
            # No edge, but both beams are steadily on the face: keep them there.
            hits = [self.last[name][1] for name in self.sensors]
            if max(hits) > BLOCK_HALF:
                along = -(max(hits) - BLOCK_HALF)
            elif min(hits) < -BLOCK_HALF:
                along = -(min(hits) + BLOCK_HALF)
            notes.append(f'along {along * 100:+.1f} (both on)')
        else:
            notes.append(f'along ? ({"edges disagree" if self.edges else "no edge"})')
        # Across: the median gap error - a noisy or odd reading does not move it.
        mad = None
        if len(self.gaps) >= CROSS_MIN_READINGS:
            med = median(self.gaps)
            mad = median([abs(g - med) for g in self.gaps])
        if mad is not None and mad <= CROSS_MAD_M:
            across = med
            notes.append(f'across {across * 100:+.1f} cm')
        else:
            notes.append(f'across ? ({"noisy" if mad is not None else "few readings"})')
        f = self.face
        dx, dy = along * f.tx + across * f.nx, along * f.ty + across * f.ny
        if math.hypot(dx, dy) > MAX_FIX_M:
            return 0.0, 0.0, f'{math.hypot(dx, dy) * 100:.0f} cm too big ({" ".join(notes)})'
        return dx, dy, ' '.join(notes)
