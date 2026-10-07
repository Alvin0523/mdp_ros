"""Foxglove drawings for Task 1, all in the `map` (arena) frame, metres.

Pure message builders - task1_runner publishes what these return:
  /occupancy_grid       costmap_grid()        the planner's costmap
  /grid_markers         arena_markers()       10 cm cell lines, area and start-box outlines
  /obstacle_markers     obstacle_markers()    blocks, image face, number, cell label, walls,
                                              task 1's scan result above each block
  /checkpoint_markers   checkpoint_markers()  where the car stops (arrow), visit order, cell
  /path_markers         leg_markers()         the leg being driven + the point being chased
  /search_progress      search_progress()     poses Hybrid A* has explored so far
  /planned_path         route_path()          every leg planned so far, as one Path
"""

import math

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

FRAME = 'map'
ARENA_M = 2.0
CELL_M = 0.10
START_BOX_M = 0.40
OBSTACLE_M = 0.10

GRID_LINE = (0.003, (0.3, 0.4, 0.5, 0.5))          # width, rgba
OUTLINE = (0.015, (0.3, 0.75, 1.0, 0.95))
OBSTACLE_RGBA = (0.9, 0.5, 0.1, 0.9)
PATH = (0.02, (0.1, 0.6, 1.0, 0.9))
CHECKPOINT_RGBA = (0.2, 1.0, 0.4, 0.95)
CHECKPOINT_ARROW_M = 0.15
FACING = {'N': (0.0, 1.0), 'S': (0.0, -1.0), 'E': (1.0, 0.0), 'W': (-1.0, 0.0)}


def header(stamp) -> Header:
    return Header(stamp=stamp, frame_id=FRAME)


def cell(v_m: float) -> int:
    """Tablet cell index (10 cm) containing v_m."""
    return int(math.floor(v_m * 100.0 / 10.0 + 1e-6))


def direction(yaw: float) -> str:
    """The nearest of N/E/S/W to a heading (rad, 0 = E, counter-clockwise)."""
    return 'ENWS'[int(round(yaw / (math.pi / 2.0))) % 4]


def marker(stamp, ns, id, type, rgba, x=0.0, y=0.0, z=0.0, scale=(1.0, 1.0, 1.0),
           yaw=0.0, text='', points=None) -> Marker:
    m = Marker(header=header(stamp), ns=ns, id=id, type=type, action=Marker.ADD, text=text)
    m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), float(z)
    m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
    m.scale.x, m.scale.y, m.scale.z = (float(s) for s in scale)
    m.color.r, m.color.g, m.color.b, m.color.a = rgba
    if points is not None:
        m.points = [Point(x=float(px), y=float(py), z=float(pz)) for px, py, pz in points]
    return m


def costmap_grid(costmap, stamp) -> OccupancyGrid:
    """The planner's costmap with Nav2's OccupancyGrid values (100 lethal,
    99 inscribed, 1..98 inflation, 0 free) - Foxglove colour mode 'costmap'."""
    msg = OccupancyGrid(header=header(stamp))
    msg.info.resolution = costmap.resolution / 100.0
    msg.info.width, msg.info.height = costmap.nx, costmap.ny
    msg.info.origin.orientation.w = 1.0
    msg.data = costmap.occupancy_grid_data()
    return msg


def _outline(x0, y0, x1, y1, z):
    return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z), (x0, y0, z)]


def arena_markers(stamp, size=(ARENA_M, ARENA_M), start_box=(0.0, 0.0, START_BOX_M, START_BOX_M)) -> MarkerArray:
    """10 cm grid over the area (task 1: the 2 x 2 m table) and the outlines of
    the area and the start box (x0, y0, x1, y1)."""
    w, h = size
    lines = []
    for i in range(int(round(w / CELL_M)) + 1):
        lines += [(i * CELL_M, 0.0, 0.01), (i * CELL_M, h, 0.01)]
    for j in range(int(round(h / CELL_M)) + 1):
        lines += [(0.0, j * CELL_M, 0.01), (w, j * CELL_M, 0.01)]
    return MarkerArray(markers=[
        marker(stamp, 'grid_lines', 0, Marker.LINE_LIST, GRID_LINE[1],
               scale=(GRID_LINE[0], 1, 1), points=lines),
        marker(stamp, 'placement_zone_outline', 0, Marker.LINE_STRIP, OUTLINE[1],
               scale=(OUTLINE[0], 1, 1), points=_outline(0.0, 0.0, w, h, 0.015)),
        marker(stamp, 'start_box_outline', 0, Marker.LINE_STRIP, OUTLINE[1],
               scale=(OUTLINE[0], 1, 1), points=_outline(*start_box, 0.015)),
    ])


RESULT_RGBA = {True: (0.2, 1.0, 0.4, 1.0), False: (1.0, 0.3, 0.3, 1.0)}   # found / UNKNOWN


SEEN_RGBA = (0.3, 0.85, 1.0, 1.0)       # read while driving past (task 1, log only)


def obstacle_markers(obstacles, labels, stamp, sizes=None, walls=(), results=None, seen=None) -> MarkerArray:
    """obstacles: (x_m, y_m, facing) block centres; labels: tablet number of each;
    results: index -> (text, found) - the scan's answer ('W / 32') above the block;
    its image face turns green when found (red: UNKNOWN); the block's own number is
    on its top; seen: index -> text read while driving past, cyan at the face. No cell labels: the grid shows the cells and they cluttered the view;
    sizes: (x, y) of each block (default 10 x 10 cm); walls: grey rectangles
    (x0, y0, x1, y1). Starts with DELETEALL so a smaller new set leaves no stale
    blocks behind."""
    markers = [Marker(header=header(stamp), action=Marker.DELETEALL)]
    thick = 0.008
    for i, (x, y, facing) in enumerate(obstacles):
        sx, sy = sizes[i] if sizes else (OBSTACLE_M, OBSTACLE_M)
        fx, fy = FACING[facing]
        markers += [
            marker(stamp, 'obstacles', i, Marker.CUBE, OBSTACLE_RGBA, x, y, 0.05,
                   scale=(sx, sy, 0.10)),
            # The image face: a thin slab on that side of the block - red, green once
            # its image was read (results).
            marker(stamp, 'obstacle_facing', 300 + i, Marker.CUBE,
                   RESULT_RGBA[True] if (results or {}).get(i, ('', False))[1] else (1.0, 0.0, 0.0, 1.0),
                   x + fx * (sx / 2.0 + thick / 2.0), y + fy * (sy / 2.0 + thick / 2.0), 0.05,
                   scale=(thick if fx else sx, thick if fy else sy, 0.10)),
            marker(stamp, 'obstacle_ids', 200 + i, Marker.TEXT_VIEW_FACING, (0.0, 0.0, 0.0, 1.0),
                   x, y, 0.102, scale=(1, 1, 0.07), text=labels[i]),
        ]
    for i, (text, found) in (results or {}).items():
        x, y, _ = obstacles[i]      # above the block (its number is on its top)
        markers.append(marker(stamp, 'scan_results', 400 + i, Marker.TEXT_VIEW_FACING, RESULT_RGBA[found],
                              x, y, 0.19, scale=(1, 1, 0.06), text=text))
    for i, text in (seen or {}).items():
        # Read on the way: cyan, against the image face.
        x, y, facing = obstacles[i]
        fx, fy = FACING[facing]
        markers.append(marker(stamp, 'drive_past', 500 + i, Marker.TEXT_VIEW_FACING, SEEN_RGBA,
                              x + fx * (OBSTACLE_M / 2.0 + 0.04), y + fy * (OBSTACLE_M / 2.0 + 0.04), 0.06,
                              scale=(1, 1, 0.045), text=text))
    for i, (x0, y0, x1, y1) in enumerate(walls):
        markers.append(marker(stamp, 'walls', i, Marker.CUBE, (0.35, 0.35, 0.35, 0.9),
                              (x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.05, scale=(x1 - x0, y1 - y0, 0.10)))
    return MarkerArray(markers=markers)


PASS_RGBA = (0.3, 0.85, 1.0, 0.95)    # a checkpoint planned as a pass (task 1)


def checkpoint_markers(checkpoints, current, stamp, passes=()) -> MarkerArray:
    """An arrow (pose) per checkpoint and its visit order. `current`: the
    index being driven to / scanned (drawn bigger), or None. `passes`: indices
    planned to be driven through without stopping - cyan, '3*'."""
    markers = []
    for i, (x, y, theta) in enumerate(checkpoints):
        rgba = PASS_RGBA if i in passes else CHECKPOINT_RGBA
        scale = (CHECKPOINT_ARROW_M, 0.02, 0.02)
        if i == current:
            rgba = (0.1, 1.0, 1.0, 1.0) if i in passes else (0.1, 1.0, 0.1, 1.0)
            scale = (CHECKPOINT_ARROW_M * 1.5, 0.04, 0.04)
        markers += [
            marker(stamp, 'checkpoints', i, Marker.ARROW, rgba, x, y, 0.05, scale=scale, yaw=theta),
            marker(stamp, 'checkpoint_ids', 200 + i, Marker.TEXT_VIEW_FACING, rgba,
                   x, y, 0.06, scale=(1, 1, 0.045), text=f"{i + 1}{'*' if i in passes else ''}"),
        ]
    return MarkerArray(markers=markers)


# Foxglove draws a text marker's dark box at the text's own opacity (no way to
# drop the box), so the timer is half see-through and small to not hide the car.
TIMER_RGBA = {'idle': (0.7, 0.7, 0.7, 0.5), 'running': (1.0, 1.0, 1.0, 0.6), 'done': (0.2, 1.0, 0.4, 0.7)}
TIMER_TEXT_M = 0.12


def run_timer(stamp, seconds, phase) -> Marker:
    """The run time as big text riding above the car, for the 3D panel (/run_timer).
    phase: 'idle' (grey), 'running' (white), 'done' (green, the final time)."""
    m = marker(stamp, 'run_timer', 0, Marker.TEXT_VIEW_FACING, TIMER_RGBA[phase], 0.08, 0.0, 0.45,
               scale=(1, 1, TIMER_TEXT_M), text=f'{seconds:.1f} s')
    m.header.frame_id = 'base_link'
    return m


IR_FIX_RGBA = (1.0, 0.55, 0.0, 1.0)


def ir_fix_markers(stamp, n, before, after, text) -> list:
    """One IR position fix (task 1 scan stop) for the 3D view: an arrow from the
    pose before to the pose after (none when nothing moved) and how far, beside it."""
    out = [marker(stamp, 'ir_fix_text', n, Marker.TEXT_VIEW_FACING, IR_FIX_RGBA,
                  after[0], after[1], 0.10, scale=(1, 1, 0.035), text=text)]
    if math.hypot(after[0] - before[0], after[1] - before[1]) >= 0.005:
        out.append(marker(stamp, 'ir_fix_arrow', n, Marker.ARROW, IR_FIX_RGBA, scale=(0.008, 0.02, 0.02),
                          points=[(before[0], before[1], 0.05), (after[0], after[1], 0.05)]))
    return out


def leg_markers(path, target, car_xy, stamp) -> MarkerArray:
    """The leg being driven (thick line), and the point the follower is chasing
    with a line from the car: magenta, orange when it is a reverse point."""
    markers = [marker(stamp, 'current_path', 0, Marker.LINE_STRIP, PATH[1], scale=(PATH[0], 1, 1),
                      points=[(p[0], p[1], 0.03) for p in path])]
    if target is None:
        for id in (0, 1):
            markers.append(Marker(header=header(stamp), ns='chased_waypoint', id=id, action=Marker.DELETE))
    else:
        rgba = (1.0, 0.5, 0.0, 1.0) if target[2] < 0 else (1.0, 0.0, 1.0, 1.0)
        markers += [
            marker(stamp, 'chased_waypoint', 0, Marker.SPHERE, rgba, target[0], target[1], 0.05,
                   scale=(0.04, 0.04, 0.04)),
            marker(stamp, 'chased_waypoint', 1, Marker.LINE_STRIP, rgba, scale=(0.01, 1, 1),
                   points=[(car_xy[0], car_xy[1], 0.05), (target[0], target[1], 0.05)]),
        ]
    return MarkerArray(markers=markers)


def search_progress(points_m, stamp) -> MarkerArray:
    return MarkerArray(markers=[
        marker(stamp, 'search_progress', 0, Marker.POINTS, (1.0, 0.6, 0.0, 0.6),
               scale=(0.02, 0.02, 1), points=[(x, y, 0.02) for x, y in points_m])])


def route_path(leg_paths, stamp) -> Path:
    """Every leg planned so far as ONE Path (a Path panel shows only the latest
    message, so per-leg messages would show one leg at a time)."""
    msg = Path(header=header(stamp))
    for leg in leg_paths:
        for p in leg or []:
            pose = PoseStamped(header=msg.header)
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = float(p[0]), float(p[1]), 0.05
            msg.poses.append(pose)
    return msg
