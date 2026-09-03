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

from typing import List, Optional, Tuple

from .hamiltonian import Hamiltonian, obstacle_to_checkpoint
from .hybrid_astar import HybridAStar
from .occupancy_map import Obstacle, OccupancyMap
from .planning_constants import MIN_TURN_RADIUS_CM

CM_PER_M = 100.0

Pose = Tuple[float, float, float]           # (x, y, theta) - metres/radians
ObstacleSpec = Tuple[int, int, str]         # (x_g, y_g, facing) - grid cells 0..39


def plan_route(obstacles_grid: List[ObstacleSpec], start_pose_m: Pose,
                theta_offset: float = 0.0, step_cm: float = 5.0,
                min_turn_radius_cm: Optional[float] = None,
                ) -> Tuple[List[int], List[List[Pose]], List[int]]:
    """Plan a full multi-obstacle route.

    Args:
        obstacles_grid: (x_g, y_g, facing) per obstacle, in the order the
            caller's obstacle list is indexed (facing in {'N','S','E','W'}).
        start_pose_m: starting (x, y, theta) in metres/radians.
        theta_offset: camera-to-body heading offset, radians (0 if the
            camera looks straight down +x of base_link).
        step_cm: Hybrid A* primitive step length, cm - smaller is a finer
            (slower) search; 5cm matches one grid cell.
        min_turn_radius_cm: overrides planning_constants.MIN_TURN_RADIUS_CM
            if given (e.g. after a servo re-calibration changes it).

    Returns:
        visiting_order: obstacle indices (into obstacles_grid), in the
            order they'll be visited. Matches task1_runner's previous
            `self.visiting_order` contract (index list, not Obstacle
            objects) so `self.visiting_order[i] + 1` still gives a 1-based
            obstacle number for Bluetooth reporting.
        leg_paths_m: one dense List[Pose] per visited obstacle, same order
            as visiting_order, each pose in METRES/radians - ready for
            pure_pursuit_follower.set_path(). An empty list for a leg means
            Hybrid A* found no path to that checkpoint (shouldn't happen
            for a checkpoint the reachability filter already validated as
            collision-free, but map/start-pose edge cases could still
            starve the search - caller should treat an empty leg as "skip
            this obstacle" rather than crash).
        unreachable: obstacle indices with no valid scan checkpoint at all
            (Hamiltonian's reachability filter) - never appear in
            visiting_order.
    """
    minR = min_turn_radius_cm if min_turn_radius_cm is not None else MIN_TURN_RADIUS_CM

    obstacles = [Obstacle(x_g=x_g, y_g=y_g, facing=facing, id=i)
                 for i, (x_g, y_g, facing) in enumerate(obstacles_grid)]
    occ_map = OccupancyMap(obstacles)

    x0_cm = start_pose_m[0] * CM_PER_M
    y0_cm = start_pose_m[1] * CM_PER_M
    theta0 = start_pose_m[2]

    tsp = Hamiltonian(occ_map, obstacles, x0_cm, y0_cm, theta0,
                       theta_offset=theta_offset, metric='reeds-shepp', minR=minR)
    order_obstacles = tsp.find_brute_force_path()
    unreachable = [o.id for o in tsp.unreachable_obstacles]

    visiting_order = [o.id for o in order_obstacles]
    leg_paths_m: List[List[Pose]] = []
    current_pose_cm = (x0_cm, y0_cm, theta0)

    for obstacle in order_obstacles:
        checkpoint = obstacle_to_checkpoint(occ_map, obstacle, theta_offset)
        planner = HybridAStar(
            occ_map,
            x_0=current_pose_cm[0], y_0=current_pose_cm[1], theta_0=current_pose_cm[2],
            x_f=checkpoint[0], y_f=checkpoint[1], theta_f=checkpoint[2],
            theta_offset=theta_offset, L=step_cm, minR=minR, heuristic='hybriddiag',
        )
        nodes, _ = planner.find_path()

        if nodes is None:
            leg_paths_m.append([])
        else:
            leg_paths_m.append([(n.x / CM_PER_M, n.y / CM_PER_M, n.theta) for n in nodes])

        current_pose_cm = (checkpoint[0], checkpoint[1], checkpoint[2])

    return visiting_order, leg_paths_m, unreachable
