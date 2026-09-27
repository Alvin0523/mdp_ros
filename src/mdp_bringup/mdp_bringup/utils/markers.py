"""Foxglove drawings for Task 1, all in the `map` (arena) frame, metres.

Pure message builders - task1_runner publishes what these return:
  /occupancy_grid       costmap_grid()        the planner's costmap
  /grid_markers         arena_markers()       10 cm cell lines, arena and start-box outlines
  /obstacle_markers     obstacle_markers()    blocks, image face, number, cell label
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
LABEL_RGBA = (1.0, 1.0, 1.0, 1.0)
LABEL_Z = 0.20
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
    msg.info.width = msg.info.height = costmap.size
    msg.info.origin.orientation.w = 1.0
    msg.data = costmap.occupancy_grid_data()
    return msg


def _outline(size_m, z):
    return [(0.0, 0.0, z), (size_m, 0.0, z), (size_m, size_m, z), (0.0, size_m, z), (0.0, 0.0, z)]


def arena_markers(stamp) -> MarkerArray:
    lines = []
    for i in range(int(round(ARENA_M / CELL_M)) + 1):
        v = i * CELL_M
        lines += [(v, 0.0, 0.01), (v, ARENA_M, 0.01), (0.0, v, 0.01), (ARENA_M, v, 0.01)]
    return MarkerArray(markers=[
        marker(stamp, 'grid_lines', 0, Marker.LINE_LIST, GRID_LINE[1],
               scale=(GRID_LINE[0], 1, 1), points=lines),
        marker(stamp, 'placement_zone_outline', 0, Marker.LINE_STRIP, OUTLINE[1],
               scale=(OUTLINE[0], 1, 1), points=_outline(ARENA_M, 0.015)),
        marker(stamp, 'start_box_outline', 0, Marker.LINE_STRIP, OUTLINE[1],
               scale=(OUTLINE[0], 1, 1), points=_outline(START_BOX_M, 0.015)),
    ])


def obstacle_markers(obstacles, labels, stamp) -> MarkerArray:
    """obstacles: (x_m, y_m, facing) block centres; labels: tablet number of each.
    Starts with DELETEALL so a smaller new set leaves no stale blocks behind."""
    markers = [Marker(header=header(stamp), action=Marker.DELETEALL)]
    half, thick = OBSTACLE_M / 2.0, 0.008
    for i, (x, y, facing) in enumerate(obstacles):
        fx, fy = FACING[facing]
        markers += [
            marker(stamp, 'obstacles', i, Marker.CUBE, OBSTACLE_RGBA, x, y, 0.05,
                   scale=(OBSTACLE_M, OBSTACLE_M, 0.10)),
            # The image face: a thin red slab on that side of the block.
            marker(stamp, 'obstacle_facing', 300 + i, Marker.CUBE, (1.0, 0.0, 0.0, 1.0),
                   x + fx * (half + thick / 2.0), y + fy * (half + thick / 2.0), 0.05,
                   scale=(thick if fx else OBSTACLE_M, thick if fy else OBSTACLE_M, 0.10)),
            marker(stamp, 'obstacle_ids', 200 + i, Marker.TEXT_VIEW_FACING, (0.0, 0.0, 0.0, 1.0),
                   x, y, 0.102, scale=(1, 1, 0.07), text=labels[i]),
            marker(stamp, 'obstacle_labels', 100 + i, Marker.TEXT_VIEW_FACING, LABEL_RGBA,
                   x, y, LABEL_Z, scale=(1, 1, 0.07), text=f'({cell(x)},{cell(y)})'),
        ]
    return MarkerArray(markers=markers)


def checkpoint_markers(checkpoints, current, stamp) -> MarkerArray:
    """An arrow (pose) per checkpoint, its visit order and cell. `current`: the
    index being driven to / scanned (drawn bigger and green), or None."""
    markers = []
    for i, (x, y, theta) in enumerate(checkpoints):
        rgba, scale = CHECKPOINT_RGBA, (CHECKPOINT_ARROW_M, 0.02, 0.02)
        if i == current:
            rgba, scale = (0.1, 1.0, 0.1, 1.0), (CHECKPOINT_ARROW_M * 1.5, 0.04, 0.04)
        markers += [
            marker(stamp, 'checkpoints', i, Marker.ARROW, rgba, x, y, 0.05, scale=scale, yaw=theta),
            marker(stamp, 'checkpoint_ids', 200 + i, Marker.TEXT_VIEW_FACING, CHECKPOINT_RGBA,
                   x, y, 0.102, scale=(1, 1, 0.07), text=str(i + 1)),
            marker(stamp, 'checkpoint_labels', 100 + i, Marker.TEXT_VIEW_FACING, CHECKPOINT_RGBA,
                   x, y, LABEL_Z, scale=(1, 1, 0.07), text=f'({cell(x)},{cell(y)})'),
        ]
    return MarkerArray(markers=markers)


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
