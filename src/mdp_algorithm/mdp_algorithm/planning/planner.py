#!/usr/bin/env python3
"""
Task 1 planner - the entry point task1_runner uses, in metres and radians.

  1. plan_visiting_order(): build the costmap, place one checkpoint per
     obstacle and choose the visiting order (visiting_order.py). Cheap -
     milliseconds - so the runner can show the route at once.
  2. plan_leg(): one leg's dense Hybrid A* path (hybrid_astar.py) from a pose
     to a checkpoint. The expensive call; the runner plans legs one after
     another in a background thread.

Settings come from mdp_bringup/config/navigation.yaml (utils/params.py). The
planning modules work in centimetres; this module converts at the boundary.
"""

from typing import Callable, List, Optional, Tuple

from ..utils import params as planner_params
from .costmap import Costmap, Obstacle, snap_to_cell_centre
from .hybrid_astar import HybridAStar
from .visiting_order import find_visiting_order

CM_PER_M = 100.0

Pose = Tuple[float, float, float]           # (x, y, theta) - metres / radians
ObstacleSpec = Tuple[float, float, str]     # (x_cm, y_cm, facing) - block centre, cm

# (x, y, theta, gear) - a path point plus the direction the car drives to reach
# it (utils.motion_primitives.Gear: FORWARD = 1, REVERSE = -1). The gear must
# reach the follower: without it, reverse segments were driven forward
# (2026-09-17, the car drove into an obstacle instead of backing up).
DensePose = Tuple[float, float, float, int]


def plan_visiting_order(obstacles_grid: List[ObstacleSpec], start_pose_m: Pose,
                        theta_offset: float = 0.0,
                        ) -> Tuple[List[int], List[Pose], List[int], Costmap]:
    """Visiting order and checkpoints for the obstacles.

    obstacles_grid: (x_cm, y_cm, facing) per obstacle, the block's centre
        (snapped to its cell's centre - the tablet only knows cells).
    start_pose_m: base_link start pose (x, y, theta), metres / radians.
    theta_offset: camera mounting angle relative to the car's heading (task 1:
        +pi/2, the camera looks left).

    Returns (visiting_order, checkpoints_m, unreachable, costmap):
        visiting_order: obstacle indices into obstacles_grid, in visit order;
        checkpoints_m: one (x, y, theta) per visited obstacle, same order;
        unreachable: indices of obstacles with no valid checkpoint (skipped);
        costmap: pass it to plan_leg() and publish it (/occupancy_grid).
    """
    obstacles = [Obstacle(x_cm=snap_to_cell_centre(x_cm), y_cm=snap_to_cell_centre(y_cm),
                          facing=facing, id=i)
                 for i, (x_cm, y_cm, facing) in enumerate(obstacles_grid)]
    costmap = Costmap(obstacles)
    radius = planner_params.ACTIVE.symmetric_turn_radius_cm
    start_cm = (start_pose_m[0] * CM_PER_M, start_pose_m[1] * CM_PER_M, start_pose_m[2])

    order, checkpoints, unreachable = find_visiting_order(costmap, start_cm, theta_offset, radius)
    return ([o.id for o in order],
            [(cp[0] / CM_PER_M, cp[1] / CM_PER_M, cp[2]) for cp in checkpoints],
            [o.id for o in unreachable],
            costmap)


def plan_leg(costmap: Costmap, start_pose_m: Pose, target_pose_m: Pose,
             progress_callback: Optional[Callable[[List[Tuple[float, float]]], None]] = None,
             progress_interval: int = 200) -> List[DensePose]:
    """One leg's path from start_pose_m to target_pose_m (metres / radians) on
    the costmap from plan_visiting_order().

    Returns a list of DensePose (x, y, theta, gear) in metres / radians, ready
    for PurePursuitController.set_path() - or [] if Hybrid A* found no path
    (the caller skips that obstacle).

    progress_callback: called every `progress_interval` expansions with the
        (x, y) points explored so far, in metres - task1_runner publishes them
        as live search progress.
    """
    cm_callback = None if progress_callback is None else (
        lambda points_cm: progress_callback([(x / CM_PER_M, y / CM_PER_M) for x, y in points_cm]))

    search = HybridAStar(
        costmap,
        x_0=start_pose_m[0] * CM_PER_M, y_0=start_pose_m[1] * CM_PER_M, theta_0=start_pose_m[2],
        x_f=target_pose_m[0] * CM_PER_M, y_f=target_pose_m[1] * CM_PER_M, theta_f=target_pose_m[2],
        progress_callback=cm_callback, progress_interval=progress_interval,
    )
    nodes, _ = search.find_path()
    if nodes is None:
        return []
    return [(n.x / CM_PER_M, n.y / CM_PER_M, n.theta, int(n.prevAction[0])) for n in nodes]
