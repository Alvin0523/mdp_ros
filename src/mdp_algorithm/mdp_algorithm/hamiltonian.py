#!/usr/bin/env python3
"""
Obstacle visit ordering (TSP over an occupancy-aware "which pose can
actually see this obstacle's image" checkpoint search), plus
obstacle-to-checkpoint conversion.

Ported from the teammate's mdp_algo package (pathfinding/hamiltonian.py) -
includes their reachability fix (obstacles with no valid collision-free
scan checkpoint are skipped, not crashed on). pygame/random-generation/
print-grid demo helpers dropped (display-only, not planning logic).
Deliberately NOT ported: find_shortest_time_hamiltonian()/path_travel_time()
- the full permutation-of-Hybrid-A*-edges "exact minimum-time route" variant.
It is correct in principle but runs a full HybridAStar search for every
ordered pair of obstacles before even picking a visit order (expensive on
Pi-class hardware for marginal benefit over the euclidean/reeds-shepp
nearest-neighbour ordering here) - left as a possible future upgrade, not
wired into collision_aware_planner.py.
"""

from itertools import permutations
from typing import List, Optional, Tuple

import numpy as np

from . import geometry_utils as utils
from . import reeds_shepp_curves as rs
from .occupancy_map import Obstacle, OccupancyMap, grid_to_coords
from .planning_constants import REAR_AXLE_TO_CENTER_CM

Checkpoint = Tuple[float, float, float, int]  # (x_cm, y_cm, theta_rad, obstacle_id)


class Hamiltonian:
    def __init__(self, map: OccupancyMap, obstacles: List[Obstacle],
                 x_start: float, y_start: float, theta_start: float,
                 theta_offset: float = 0.0, metric: str = 'euclidean', minR: float = 25) -> None:
        assert -np.pi < theta_start <= np.pi
        self.map = map
        self.obstacles = obstacles
        self.start = (x_start, y_start, theta_start)
        self.theta_offset = theta_offset
        self.metric = metric
        self.minR = minR
        self.unreachable_obstacles: List[Obstacle] = []

    def _reachable_obstacles(self) -> List[Obstacle]:
        """Obstacles with at least one valid, collision-free scan checkpoint."""
        reachable = []
        self.unreachable_obstacles = []
        for obstacle in self.obstacles:
            if obstacle_to_checkpoint(self.map, obstacle, self.theta_offset) is None:
                self.unreachable_obstacles.append(obstacle)
            else:
                reachable.append(obstacle)
        return reachable

    def _leg_distance(self, current_pos: Tuple[float, float, float], checkpoint: Checkpoint) -> float:
        if self.metric == 'reeds-shepp':
            return rs.get_optimal_path_length(current_pos, checkpoint, self.minR)
        return utils.l2(current_pos[0], current_pos[1], checkpoint[0], checkpoint[1])

    def find_brute_force_path(self) -> List[Obstacle]:
        """Exact TSP ordering across reachable obstacles - fine for the <=8
        obstacles this arena has (at most 8! = 40320 permutations, using a
        cheap euclidean/reeds-shepp distance metric, not a full Hybrid A*
        search per edge)."""
        reachable = self._reachable_obstacles()
        if not reachable:
            return []

        shortest_distance = float('inf')
        shortest_path: List[Obstacle] = []
        for obstacle_path in permutations(reachable):
            current_pos = self.start
            total_distance = 0.0
            for obstacle in obstacle_path:
                checkpoint = obstacle_to_checkpoint(self.map, obstacle, self.theta_offset)
                total_distance += self._leg_distance(current_pos, checkpoint)
                current_pos = checkpoint
            if total_distance < shortest_distance:
                shortest_distance = total_distance
                shortest_path = list(obstacle_path)
        return shortest_path

    def find_nearest_neighbor_path(self) -> List[Obstacle]:
        """Greedy nearest-checkpoint ordering - fast fallback for larger
        obstacle counts, not needed at <=8 but kept for parity with the
        teammate's original API."""
        current_pos = self.start
        path: List[Obstacle] = []
        obstacles = self._reachable_obstacles()

        while obstacles:
            nearest = None
            min_dist = float('inf')
            for obstacle in obstacles:
                checkpoint = obstacle_to_checkpoint(self.map, obstacle, self.theta_offset)
                dist = self._leg_distance(current_pos, checkpoint)
                if dist < min_dist:
                    min_dist = dist
                    nearest = obstacle
            if nearest is None:
                break
            path.append(nearest)
            obstacles.remove(nearest)
            current_pos = obstacle_to_checkpoint(self.map, nearest, self.theta_offset)

        return path


def obstacle_to_checkpoint(map: OccupancyMap, obstacle: Obstacle, theta_offset: float) -> Optional[Checkpoint]:
    """First valid (closest-first, by scan radius then angle offset from
    dead-ahead) collision-free stand-off pose from which the obstacle's
    image can be scanned, or None if no such pose exists (reachability
    fix)."""
    starting_x, starting_y = grid_to_coords(obstacle.x_g, obstacle.y_g)
    starting_x += _offset_x(obstacle.facing)
    starting_y += _offset_y(obstacle.facing)
    starting_image_to_pos_theta = _offset_theta(obstacle.facing, np.pi)

    theta_scan_list = [0, np.pi / 36, -np.pi / 36, np.pi / 18, -np.pi / 18, np.pi / 12, -np.pi / 12,
                        np.pi / 9, -np.pi / 9, np.pi / 7.2, -np.pi / 7.2, np.pi / 6, -np.pi / 6,
                        np.pi * 180 / 35, -np.pi * 180 / 35, np.pi / 4.5, -np.pi / 4.5, np.pi / 4, -np.pi / 4]
    r_scan_list = [20, 19, 21, 18, 22, 17, 23, 16, 24, 15, 25, 26, 27, 28, 29, 30]

    for r_scan in r_scan_list:
        for theta_scan in theta_scan_list:
            cur_image_to_pos_theta = utils.M(starting_image_to_pos_theta + theta_scan)
            image_x = starting_x + r_scan * np.cos(cur_image_to_pos_theta)
            image_y = starting_y + r_scan * np.sin(cur_image_to_pos_theta)
            theta = utils.M(cur_image_to_pos_theta - theta_offset)

            # Candidate REAR-AXLE checkpoint - what actually gets returned
            # and used for planning. The car's front bumper reaches toward
            # the scan position (image_x, image_y); the rear axle sits
            # REAR_AXLE_TO_CENTER_CM behind it along theta.
            rear_x = image_x - REAR_AXLE_TO_CENTER_CM * np.cos(theta)
            rear_y = image_y - REAR_AXLE_TO_CENTER_CM * np.sin(theta)

            # FIXED DURING PORT: validate the point actually being
            # returned (rear_x/rear_y) plus the front bumper
            # (image_x/image_y), covering the car's whole length. The
            # original validated only the pre-shift scan position (plus
            # +-half the offset around IT, not around the shifted point)
            # then returned a DIFFERENT, unvalidated point - which could
            # pass while landing the actual checkpoint back inside the
            # very obstacle's own 15cm inflation zone (confirmed:
            # r_scan - REAR_AXLE_TO_CENTER_CM can be well under 15cm).
            if not map.collide_with_point(rear_x, rear_y) and not map.collide_with_point(image_x, image_y):
                return (rear_x, rear_y, theta, obstacle.id)

    return None


def obstacle_to_checkpoint_all(map: OccupancyMap, obstacle: Obstacle, theta_offset: float) -> List[Checkpoint]:
    """All valid checkpoints for one obstacle, closest-first - kept for
    callers that want more than the single best candidate."""
    starting_x, starting_y = grid_to_coords(obstacle.x_g, obstacle.y_g)
    starting_x += _offset_x(obstacle.facing)
    starting_y += _offset_y(obstacle.facing)
    starting_image_to_pos_theta = _offset_theta(obstacle.facing, np.pi)

    valid_checkpoints: List[Checkpoint] = []
    theta_scan_list = [0, np.pi / 36, -np.pi / 36, np.pi / 18, -np.pi / 18, np.pi / 12, -np.pi / 12,
                        np.pi / 9, -np.pi / 9, np.pi / 7.2, -np.pi / 7.2, np.pi / 6, -np.pi / 6]
    r_scan_list = [20, 19, 21, 18, 22, 17, 23, 16, 24, 15, 25, 26, 27, 28, 29, 30]

    for r_scan in r_scan_list:
        for theta_scan in theta_scan_list:
            cur_image_to_pos_theta = utils.M(starting_image_to_pos_theta + theta_scan)
            image_x = starting_x + r_scan * np.cos(cur_image_to_pos_theta)
            image_y = starting_y + r_scan * np.sin(cur_image_to_pos_theta)
            theta = utils.M(cur_image_to_pos_theta - theta_offset)

            rear_x = image_x - REAR_AXLE_TO_CENTER_CM * np.cos(theta)
            rear_y = image_y - REAR_AXLE_TO_CENTER_CM * np.sin(theta)

            # See obstacle_to_checkpoint() above for why both the rear-axle
            # point actually returned and the front bumper are validated.
            if not map.collide_with_point(rear_x, rear_y) and not map.collide_with_point(image_x, image_y):
                valid_checkpoints.append((rear_x, rear_y, theta, obstacle.id))

    return valid_checkpoints


def _offset_x(facing: str) -> float:
    return {'N': 5.0, 'S': 5.0, 'E': 0.0, 'W': 10.0}[facing]


def _offset_y(facing: str) -> float:
    return {'N': 0.0, 'S': 10.0, 'E': 5.0, 'W': 5.0}[facing]


def _offset_theta(facing: str, theta_offset: float) -> float:
    return utils.M(utils.facing_to_rad(facing) + theta_offset)
