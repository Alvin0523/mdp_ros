"""
Pure (ROS-free) helpers for task1_runner.py, kept separate so they can be unit
tested without rclpy. Installed next to task1_runner.py, which imports it as a
sibling module (same arrangement as pose_transform.py).
"""
import math
from typing import List, Optional, Tuple

_FACINGS = 'NESW'


def parse_setup(text: str, cells: bool = False) -> List[Tuple[int, float, float, str]]:
    """Obstacle-setup payload -> [(id, x_m, y_m, facing 'N'|'E'|'S'|'W')].

    Wire format: 'id:x,y,facing|id:x,y,facing|...'. The id is the tablet's own
    obstacle number, kept so TARGET replies can quote it.

    cells=False ('/obstacle_setup', dev tools/sim): x/y are the obstacle centre
    in metres, arena frame.
    cells=True ('/obstacle_cells', the tablet): x/y are integer grid cells
    0..19 (10cm cells, origin bottom-left); the obstacle is centred in its cell,
    i.e. centre = cell * 10 + 5 cm.

    Malformed items are skipped; a non-numeric id falls back to the item's
    1-based position.
    """
    items = []
    for position, raw in enumerate(text.strip().split('|'), start=1):
        if ':' not in raw:
            continue
        obs_id, data = raw.split(':', 1)
        parts = [p.strip() for p in data.split(',')]
        if len(parts) < 3:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if cells:
            if x != int(x) or y != int(y) or not (0 <= x <= 19 and 0 <= y <= 19):
                continue
            x_m, y_m = (x * 10.0 + 5.0) / 100.0, (y * 10.0 + 5.0) / 100.0
        else:
            x_m, y_m = x, y
        facing = parts[2][:1].upper()
        if facing not in _FACINGS:
            continue
        try:
            oid = int(obs_id.strip())
        except ValueError:
            oid = position
        items.append((oid, x_m, y_m, facing))
    return items


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def pose_at_start(pose: Tuple[float, float, float], start: Tuple[float, float, float],
                  tol_m: float, tol_rad: float) -> bool:
    """True when (x, y, yaw) is within tol_m of the start position AND within
    tol_rad of the start heading (yaw wrap handled)."""
    dist = math.hypot(pose[0] - start[0], pose[1] - start[1])
    return dist <= tol_m and abs(_wrap_pi(pose[2] - start[2])) <= tol_rad


def indicator_lines(state_name: str, obstacles_known: bool, planning_active: bool,
                    legs_done: bool, at_start: bool, target_obstacle: Optional[int],
                    estop: bool = False) -> List[str]:
    """The tablet lines (without the trailing newline) describing the runner.

    PLAN:/RESET: are only reported while idle - during a run the tablet keeps
    the last values, since RESET would otherwise flip to WAITING the moment
    the car drives away from the start. STATUS: and ESTOP: are always reported;
    ESTOP:ON = the STM32's motor switch is engaged (the firmware then refuses to
    drive), and the runner is never 'Ready' while it is.
    """
    running = state_name in ('NAVIGATING_TO_TARGET', 'PAUSE_FOR_SCAN')
    if state_name == 'NAVIGATING_TO_TARGET':
        status = f"Going to obstacle {target_obstacle}"
    elif state_name == 'PAUSE_FOR_SCAN':
        status = f"Scanning obstacle {target_obstacle}"
    elif state_name == 'FINISHED':
        status = 'Finished'
    elif state_name == 'STOPPED':
        status = 'Stopped'
    else:
        status = 'Ready' if (legs_done and at_start and not estop) else 'Not ready'

    estop_line = f"ESTOP:{'ON' if estop else 'OFF'}"
    if running:
        return [estop_line, f"STATUS:{status}"]

    if legs_done:
        plan = 'DONE'
    elif planning_active or state_name == 'PLANNING_PATH':
        plan = 'PLANNING'
    else:
        plan = 'WAITING'
    return [f"PLAN:{plan}", f"RESET:{'DONE' if at_start else 'WAITING'}", estop_line,
            f"STATUS:{status}"]
