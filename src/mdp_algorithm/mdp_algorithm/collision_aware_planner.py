#!/usr/bin/env python3
"""
Top-level entry point tying occupancy_map.py + hamiltonian.py +
hybrid_astar.py together: given a start pose and a list of obstacles, find
a visiting order and a collision-checked, kinematically-feasible dense path
to each one in turn.

This is the new module (not present in the teammate's mdp_algo package,
which instead ended its pipeline at pathcommands.py's discrete
"SF020"/"LF045" command strings for an open-loop executor board - see
task1_runner.py's module docstring for why that layer isn't used here).
Its job is purely to bridge the ported cm-based planning modules to this
package's metre/pure_pursuit_follower.set_path() boundary.
"""

from typing import Callable, List, Optional, Tuple

from .hamiltonian import Hamiltonian, obstacle_to_checkpoint
from .hybrid_astar import HybridAStar
from .occupancy_map import Obstacle, OccupancyMap
from .planning_constants import MIN_TURN_RADIUS_CM

CM_PER_M = 100.0

Pose = Tuple[float, float, float]           # (x, y, theta) - metres/radians
ObstacleSpec = Tuple[float, float, str]     # (x_cm, y_cm, facing) - continuous centre position, cm


def plan_visiting_order(obstacles_grid: List[ObstacleSpec], start_pose_m: Pose,
                         theta_offset: float = 0.0, min_turn_radius_cm: Optional[float] = None,
                         ) -> Tuple[List[int], List[Pose], List[int], OccupancyMap]:
    """Phase 1 of planning a route: which order to visit obstacles in, and
    the stand-off checkpoint for each - the Hamiltonian TSP solve only, NO
    Hybrid A* dense-path search. Cheap (a handful of Reeds-Shepp distance
    evaluations via hamiltonian.py's cached distance matrix - tens of
    milliseconds for 8 obstacles, not the multi-second-per-leg cost that's
    actually in plan_leg()).

    Split out from what used to be plan_route() so a caller (task1_runner.py)
    can get the full visiting order and start driving toward the FIRST
    target as soon as plan_leg() finishes ONE leg, instead of blocking on
    every leg's dense path before the robot moves at all - call this once,
    then plan_leg() one leg at a time as each is actually needed.

    Args:
        obstacles_grid: (x_cm, y_cm, facing) per obstacle - x_cm/y_cm is the
            CENTRE of that obstacle's 10x10cm footprint (continuous, not a
            grid-cell index - see occupancy_map.Obstacle), in the order the
            caller's obstacle list is indexed (facing in {'N','S','E','W'}).
        start_pose_m: starting (x, y, theta) in metres/radians.
        theta_offset: camera-to-body heading offset, radians (0 if the
            camera looks straight down +x of base_link).
        min_turn_radius_cm: overrides planning_constants.MIN_TURN_RADIUS_CM
            if given (e.g. after a servo re-calibration changes it) - only
            affects checkpoint ordering here (the Reeds-Shepp distance
            metric uses it too), NOT collision-checking (occ_map has none
            baked in - that's a Hybrid A* concern, see plan_leg()).

    Returns:
        visiting_order: obstacle indices (into obstacles_grid), in the
            order they'll be visited. Matches task1_runner's previous
            `self.visiting_order` contract (index list, not Obstacle
            objects) so `self.visiting_order[i] + 1` still gives a 1-based
            obstacle number for Bluetooth reporting.
        checkpoints_m: one (x, y, theta) per visited obstacle, same order as
            visiting_order - the stand-off pose (see hamiltonian.py's
            obstacle_to_checkpoint()) plan_leg() should be asked to reach
            for that leg.
        unreachable: obstacle indices with no valid scan checkpoint at all
            (Hamiltonian's reachability filter) - never appear in
            visiting_order.
        occ_map: the OccupancyMap built internally for this plan - pass this
            straight into plan_leg() (rebuilding it per-leg would be
            pointless, it's the same obstacle list) and/or use it for
            visualization (nav_msgs/OccupancyGrid) without re-building it
            from the same obstacle list.
    """
    minR = min_turn_radius_cm if min_turn_radius_cm is not None else MIN_TURN_RADIUS_CM

    obstacles = [Obstacle(x_cm=x_cm, y_cm=y_cm, facing=facing, id=i)
                 for i, (x_cm, y_cm, facing) in enumerate(obstacles_grid)]
    occ_map = OccupancyMap(obstacles)

    x0_cm = start_pose_m[0] * CM_PER_M
    y0_cm = start_pose_m[1] * CM_PER_M
    theta0 = start_pose_m[2]

    tsp = Hamiltonian(occ_map, obstacles, x0_cm, y0_cm, theta0,
                       theta_offset=theta_offset, metric='reeds-shepp', minR=minR)
    order_obstacles = tsp.find_brute_force_path()
    unreachable = [o.id for o in tsp.unreachable_obstacles]

    visiting_order = [o.id for o in order_obstacles]
    checkpoints_m: List[Pose] = []
    for obstacle in order_obstacles:
        checkpoint = obstacle_to_checkpoint(occ_map, obstacle, theta_offset)
        checkpoints_m.append((checkpoint[0] / CM_PER_M, checkpoint[1] / CM_PER_M, checkpoint[2]))

    return visiting_order, checkpoints_m, unreachable, occ_map


def plan_leg(occ_map: OccupancyMap, start_pose_m: Pose, target_pose_m: Pose,
             theta_offset: float = 0.0, step_cm: float = 5.0,
             min_turn_radius_cm: Optional[float] = None,
             progress_callback: Optional[Callable[[List[Tuple[float, float]]], None]] = None,
             progress_interval: int = 200) -> List[Pose]:
    """Phase 2 of planning a route: ONE leg's dense, collision-checked
    Hybrid A* path from start_pose_m to target_pose_m (both metres/radians),
    against an OccupancyMap already built by plan_visiting_order(). This is
    the expensive call (multiple seconds, was 50s+/leg before hamiltonian.py's
    distance-matrix caching fix removed a different, larger cost) - call it
    once per leg, right before the robot is actually about to drive that
    leg, not all of them up front, so the robot starts moving after the
    FIRST leg's search instead of after the whole route's.

    progress_callback: if given, called every `progress_interval` node
        expansions with the (x, y) points explored SO FAR, in METRES (not
        cm - converted here at this module's cm<->m boundary, same as
        everything else this module hands back) - see HybridAStar.find_path()'s
        own docstring. Lets a caller (task1_runner.py) publish live search
        progress instead of the search being an invisible black box for
        however long it takes.

    Returns a dense List[Pose] in METRES/radians, ready for
    pure_pursuit_follower.set_path() - or an empty list if Hybrid A* found
    no path to target_pose_m (shouldn't happen for a checkpoint the
    reachability filter already validated as collision-free, but map/start-
    pose edge cases could still starve the search - caller should treat an
    empty leg as "skip this obstacle" rather than crash).
    """
    minR = min_turn_radius_cm if min_turn_radius_cm is not None else MIN_TURN_RADIUS_CM

    cm_callback = None
    if progress_callback is not None:
        def cm_callback(points_cm: List[Tuple[float, float]]) -> None:
            progress_callback([(x / CM_PER_M, y / CM_PER_M) for x, y in points_cm])

    planner = HybridAStar(
        occ_map,
        x_0=start_pose_m[0] * CM_PER_M, y_0=start_pose_m[1] * CM_PER_M, theta_0=start_pose_m[2],
        x_f=target_pose_m[0] * CM_PER_M, y_f=target_pose_m[1] * CM_PER_M, theta_f=target_pose_m[2],
        theta_offset=theta_offset, L=step_cm, minR=minR, heuristic='hybriddiag',
        progress_callback=cm_callback, progress_interval=progress_interval,
    )
    nodes, _ = planner.find_path()
    if nodes is None:
        return []
    return [(n.x / CM_PER_M, n.y / CM_PER_M, n.theta) for n in nodes]


def plan_route(obstacles_grid: List[ObstacleSpec], start_pose_m: Pose,
                theta_offset: float = 0.0, step_cm: float = 5.0,
                min_turn_radius_cm: Optional[float] = None,
                ) -> Tuple[List[int], List[List[Pose]], List[int], OccupancyMap, List[Pose]]:
    """Plan a full multi-obstacle route in one blocking call - every leg's
    dense path, computed up front. Kept as a convenience wrapper around
    plan_visiting_order()+plan_leg() for callers that don't care about
    getting the first leg back early (task1_runner.py does, and uses the
    two split functions directly instead - see this module's other
    docstrings for why blocking on the whole route isn't what you want for
    a robot that should start moving ASAP)."""
    visiting_order, checkpoints_m, unreachable, occ_map = plan_visiting_order(
        obstacles_grid, start_pose_m, theta_offset, min_turn_radius_cm)

    leg_paths_m: List[List[Pose]] = []
    current_pose_m = start_pose_m
    for checkpoint_m in checkpoints_m:
        leg_paths_m.append(plan_leg(occ_map, current_pose_m, checkpoint_m,
                                     theta_offset, step_cm, min_turn_radius_cm))
        current_pose_m = checkpoint_m

    return visiting_order, leg_paths_m, unreachable, occ_map, checkpoints_m
