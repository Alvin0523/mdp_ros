#!/usr/bin/env python3
"""
Obstacle visit ordering (TSP over an occupancy-aware "which pose can
actually see this obstacle's image" checkpoint search), plus
obstacle-to-checkpoint conversion.

Ported from the teammate's mdp_algo package (pathfinding/hamiltonian.py) -
includes their reachability fix (obstacles with no valid collision-free
scan checkpoint are skipped, not crashed on). pygame/random-generation/
print-grid demo helpers dropped (display-only, not planning logic).

Two ordering families live here:
  - Distance-based (find_brute_force_path/find_nearest_neighbor_path): cheap,
    orders by straight-line or Reeds-Shepp distance between checkpoints. This
    is the default fast path used by collision_aware_planner.plan_visiting_order().
  - Time-based (find_shortest_time_order + path_travel_time): the exact
    minimum-TIME route. It runs a full HybridAStar search (cost_mode='time')
    for every ordered pair of {start, reachable checkpoints}, then a Held-Karp
    DP over those calibrated leg times picks the optimal visit order. This
    orders by ACTUAL collision-aware drive time (a "near" obstacle behind a
    wall is correctly costed as slow), not straight-line distance - at the
    price of N^2+N Hybrid A* searches up front before the order is known.
    Opt-in only (see collision_aware_planner.plan_visiting_order_min_time()),
    NOT the default, precisely because of that up-front cost on Pi-class
    hardware. The leg node-paths it computes are returned so the caller can
    reuse them instead of re-searching each leg.
"""

from itertools import permutations
from typing import List, Optional, Tuple

import numpy as np

from ..common import geometry_utils as utils
from . import reeds_shepp_curves as rs
from ..common.motion_primitives import Gear, Steering
from .occupancy_map import INFLATION_RADIUS_CM, Obstacle, OccupancyMap

Checkpoint = Tuple[float, float, float, int]  # (x_cm, y_cm, theta_rad, obstacle_id)


def _blocked_by_any_obstacle(map: OccupancyMap, x: float, y: float) -> bool:
    """Exact continuous-distance check - is (x, y) within INFLATION_RADIUS_CM
    of ANY obstacle's centre? Used for checkpoint validation INSTEAD of
    map.collide_with_point(), which snaps to a 10cm grid cell first.

    CONFIRMED BUG (2026-09-05, live run): obstacle_to_checkpoint() places the
    checkpoint EXACTLY INFLATION_RADIUS_CM from its own obstacle's centre by
    construction - but that exact point can fall into a grid cell whose
    discretised CENTRE is closer than the true radius (e.g. obstacle at
    (50,100)cm facing S -> checkpoint at exactly 20.00cm, landing in a cell
    whose centre is only 15.81cm away) - map.collide_with_point() then
    reports the checkpoint as blocked BY ITS OWN OBSTACLE'S inflation,
    wrongly marking it unreachable. This happened to 3 of 5 obstacles in
    task1_runner's own test run (obstacles facing S, W, S - not a
    coincidence about compass direction, just which ones happened to round
    the wrong way against the 10cm grid).

    This checks the real, continuous distance instead - consistent with how
    the checkpoint's position was computed AND how add_obstacles_to_grid()'s
    own inflation circle is defined. Strict `<` (not `<=`): a point exactly
    ON the boundary (like every checkpoint, by construction) is NOT
    blocked - only genuinely inside a keep-out circle counts."""
    return any(np.hypot(x - o.x_cm, y - o.y_cm) < INFLATION_RADIUS_CM for o in map.obstacles)


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
        self._checkpoint_cache: dict = {}   # obstacle.id -> Checkpoint, filled by _reachable_obstacles()

    def _reachable_obstacles(self) -> List[Obstacle]:
        """Obstacles with at least one valid, collision-free scan checkpoint.

        Caches each reachable obstacle's checkpoint - obstacle_to_checkpoint()
        is a deterministic function of (map, obstacle, theta_offset), none of
        which change after this call, so find_brute_force_path()/
        find_nearest_neighbor_path() reuse this instead of recomputing the
        same checkpoint (a ~300-iteration collision scan) over and over."""
        reachable = []
        self.unreachable_obstacles = []
        self._checkpoint_cache = {}
        for obstacle in self.obstacles:
            checkpoint = obstacle_to_checkpoint(self.map, obstacle, self.theta_offset)
            if checkpoint is None:
                self.unreachable_obstacles.append(obstacle)
            else:
                reachable.append(obstacle)
                self._checkpoint_cache[obstacle.id] = checkpoint
        return reachable

    def _leg_distance(self, current_pos: Tuple[float, float, float], checkpoint: Checkpoint) -> float:
        if self.metric == 'reeds-shepp':
            return rs.get_optimal_path_length(current_pos, checkpoint, self.minR)
        return utils.l2(current_pos[0], current_pos[1], checkpoint[0], checkpoint[1])

    def find_brute_force_path(self) -> List[Obstacle]:
        """Exact TSP ordering across reachable obstacles - fine for the <=8
        obstacles this arena has (at most 8! = 40320 permutations).

        Every obstacle has exactly ONE checkpoint (obstacle_to_checkpoint()
        picks the first valid one and that never depends on visit order), so
        every leg distance - start->checkpoint, checkpoint->checkpoint - is
        the same fixed number no matter which permutation is being scored.
        Recomputing it with the Reeds-Shepp solver inside the permutation
        loop (the previous version of this method) meant up to 8! * 8 ~=
        320k solver calls for identical (start_i, checkpoint_j) inputs.
        Precomputing all of them ONCE into a small (n+1) x n matrix - at
        most 9*8 = 72 solver calls - and having the permutation loop do
        array lookups instead cannot change which ordering comes out
        shortest (same numbers, just computed once), only how long it takes
        to find it."""
        reachable = self._reachable_obstacles()
        if not reachable:
            return []

        n = len(reachable)
        # positions[0] = start pose; positions[i+1] = reachable[i]'s checkpoint.
        positions = [self.start] + [self._checkpoint_cache[o.id] for o in reachable]
        dist = [[self._leg_distance(positions[i], positions[j + 1]) if i != j + 1 else 0.0
                 for j in range(n)] for i in range(n + 1)]

        shortest_distance = float('inf')
        shortest_order: Tuple[int, ...] = ()
        for order in permutations(range(n)):
            total_distance = dist[0][order[0]]
            for k in range(n - 1):
                total_distance += dist[order[k] + 1][order[k + 1]]
                if total_distance >= shortest_distance:
                    break   # already worse than the best found - stop scoring this permutation
            if total_distance < shortest_distance:
                shortest_distance = total_distance
                shortest_order = order
        return [reachable[i] for i in shortest_order]

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
                checkpoint = self._checkpoint_cache[obstacle.id]
                dist = self._leg_distance(current_pos, checkpoint)
                if dist < min_dist:
                    min_dist = dist
                    nearest = obstacle
            if nearest is None:
                break
            path.append(nearest)
            obstacles.remove(nearest)
            current_pos = self._checkpoint_cache[nearest.id]

        return path

    def find_shortest_time_order(
        self, L: float = 5.0, thetaBins: int = 24,
        forward_speed: float = 20.0, reverse_speed: float = 15.0,
        gear_change_time: float = 0.5, steering_change_time: float = 0.15,
        recognition_seconds: float = 1.0,
    ) -> Tuple[List[Obstacle], List[list], float, List[Obstacle]]:
        """Exact minimum-TIME visit order via Held-Karp DP over collision-aware
        Hybrid A* leg times (NOT straight-line distance - see module docstring).

        Runs one HybridAStar search (cost_mode='time') for every ordered pair
        of {start, reachable checkpoints}, costs each resulting path in
        calibrated seconds (path_travel_time), then a Held-Karp dynamic program
        (O(n^2 * 2^n), not O(n!)) finds the order minimising total drive time.
        Adds recognition_seconds per obstacle to the reported total (it's a
        constant, so it does not affect the chosen ORDER, only best_time).

        Args:
            L: arc length per Hybrid A* primitive step, cm (matches the
                collision_aware_planner.plan_leg() step so reused legs are
                identical to what plan_leg() would have produced).
            thetaBins: heading discretisation for Hybrid A* (must match
                plan_leg()'s default of 24 for reuse consistency).
            forward_speed/reverse_speed: cm/s, for the time cost model.
            gear_change_time/steering_change_time: seconds lost per gear /
                steering change.
            recognition_seconds: image-recognition dwell added per obstacle to
                best_time (order-invariant).

        Returns:
            order: reachable Obstacles in optimal visit order (subset of
                self.obstacles; unreachable ones excluded).
            leg_paths: one list of HybridAStar Nodes per leg (cm poses), same
                order as `order` - leg_paths[i] is start->order[0] for i==0,
                else order[i-1]->order[i]. Empty legs never occur here (only
                reachable, searched-successfully pairs are used).
            total_time: calibrated seconds for the whole route incl.
                recognition dwell, or inf if no full route exists.
            unreachable: obstacles with no valid scan checkpoint (same as
                self.unreachable_obstacles after this call).

        Lazy-imports HybridAStar to avoid an import cycle (hybrid_astar.py
        imports occupancy_map, which this module also imports)."""
        from .hybrid_astar import HybridAStar

        reachable = self._reachable_obstacles()
        if not reachable:
            return [], [], float('inf'), list(self.unreachable_obstacles)

        n = len(reachable)

        def _plan_leg(source: Tuple[float, float, float],
                      destination: Checkpoint) -> Tuple[Optional[list], float]:
            planner = HybridAStar(
                self.map,
                x_0=source[0], y_0=source[1], theta_0=source[2],
                x_f=destination[0], y_f=destination[1], theta_f=destination[2],
                theta_offset=self.theta_offset, L=L, minR=self.minR,
                heuristic='euclidean', thetaBins=thetaBins, cost_mode='time',
                forward_speed=forward_speed, reverse_speed=reverse_speed,
                gear_change_time=gear_change_time,
                steering_change_time=steering_change_time,
            )
            path, _ = planner.find_path()
            if path is None:
                return None, float('inf')
            return path, path_travel_time(
                path, L, forward_speed, reverse_speed,
                gear_change_time, steering_change_time)

        # Directed edge tables. Index convention: -1 == start; 0..n-1 map to
        # reachable[i]. Store BOTH cost and node-path so the caller can reuse
        # the exact leg Hybrid A* already computed (no second search).
        edge_cost = {}
        edge_path = {}
        for j, obstacle in enumerate(reachable):
            path, cost = _plan_leg(self.start, self._checkpoint_cache[obstacle.id])
            edge_cost[(-1, j)] = cost
            edge_path[(-1, j)] = path
        for i, src_obs in enumerate(reachable):
            for j, dst_obs in enumerate(reachable):
                if i == j:
                    continue
                path, cost = _plan_leg(self._checkpoint_cache[src_obs.id],
                                       self._checkpoint_cache[dst_obs.id])
                edge_cost[(i, j)] = cost
                edge_path[(i, j)] = path

        # Held-Karp DP. best[(mask, last)] = (min time to start->...->last
        # having visited exactly the obstacles in `mask`, ending at `last`,
        # parent_last_or_None). mask is a bitset over reachable indices.
        best: dict = {}
        for j in range(n):
            cost = edge_cost[(-1, j)]
            if np.isfinite(cost):
                best[(1 << j, j)] = (cost, None)

        for mask in range(1, 1 << n):
            for last in range(n):
                state = best.get((mask, last))
                if state is None:
                    continue
                base_time = state[0]
                for nxt in range(n):
                    if mask & (1 << nxt):
                        continue
                    step = edge_cost[(last, nxt)]
                    if not np.isfinite(step):
                        continue
                    key = (mask | (1 << nxt), nxt)
                    candidate = base_time + step
                    if key not in best or candidate < best[key][0]:
                        best[key] = (candidate, last)

        full_mask = (1 << n) - 1
        finishes = [(best[(full_mask, last)][0], last)
                    for last in range(n) if (full_mask, last) in best]
        if not finishes:
            # No complete route reachable (some pair had no Hybrid A* path).
            # Report every reachable obstacle as effectively unvisitable in a
            # single route; caller falls back to distance ordering.
            return [], [], float('inf'), list(self.unreachable_obstacles) + list(reachable)

        total_time, last = min(finishes)

        # Reconstruct the index order by walking parent pointers back.
        order_idx: List[int] = []
        mask = full_mask
        while last is not None:
            order_idx.append(last)
            prev = best[(mask, last)][1]
            mask ^= 1 << last
            last = prev
        order_idx.reverse()

        order = [reachable[i] for i in order_idx]
        leg_paths: List[list] = []
        prev_idx = -1
        for i in order_idx:
            leg_paths.append(edge_path[(prev_idx, i)])
            prev_idx = i

        total_time += recognition_seconds * n
        return order, leg_paths, total_time, list(self.unreachable_obstacles)


def path_travel_time(path: list, L: float, forward_speed: float, reverse_speed: float,
                     gear_change_time: float, steering_change_time: float) -> float:
    """Calibrated traversal time (seconds) for a Hybrid A* node path.

    Each node carries prevAction = (Gear, Steering) - the primitive that
    reached it. Straight-line time is L/speed (forward vs reverse speed per
    the gear); a gear change adds gear_change_time and a steering change adds
    steering_change_time. Seed the previous action as (FORWARD, STRAIGHT) so
    the first node's own gear/steering choice is charged a change if it
    differs from that resting state (matches the newalgo original)."""
    total = 0.0
    previous_action = (Gear.FORWARD, Steering.STRAIGHT)
    for node in path:
        action = node.prevAction
        speed = forward_speed if action[0] == Gear.FORWARD else reverse_speed
        total += L / speed
        if action[0] != previous_action[0]:
            total += gear_change_time
        if action[1] != previous_action[1]:
            total += steering_change_time
        previous_action = action
    return total


def obstacle_to_checkpoint(map: OccupancyMap, obstacle: Obstacle, theta_offset: float) -> Optional[Checkpoint]:
    """The checkpoint is a single, direct point - NOT a closest-first search
    over candidate radii/angles like the previous version of this function.
    Confirmed by direct example (2026-09-04): obstacle at (120,40)cm facing
    'W' -> checkpoint at (100,40)cm; obstacle at (50,100)cm facing 'S' ->
    checkpoint at (50,80)cm. Both are exactly:

        checkpoint = obstacle_centre + INFLATION_RADIUS_CM * facing_direction

    i.e. sitting exactly on the inflation circle's boundary, straight out
    from the obstacle's centre in the `facing` direction - no rear-axle
    projection, no half-obstacle-size offset, no scan search. Heading is a
    simple derived value: face back toward the obstacle (facing+pi),
    corrected by theta_offset for the camera's fixed mounting angle (the
    car only has ONE camera, mounted on one side - a fixed offset from body
    heading, not something to search over per obstacle).

    Only reachability is still checked (this point must be collision-free -
    e.g. not overlapping a second obstacle's own inflation circle); unlike
    the old search there is no fallback to a farther candidate if this one
    obstacle's own point is invalid - it can't be, by construction (a point
    exactly on ITS OWN OBSTACLE's inflation boundary is never inside that
    same obstacle's inflation zone), so a collision here only happens when a
    DIFFERENT, nearby obstacle's inflation circle overlaps this point."""
    facing_rad = utils.facing_to_rad(obstacle.facing)
    x = obstacle.x_cm + INFLATION_RADIUS_CM * np.cos(facing_rad)
    y = obstacle.y_cm + INFLATION_RADIUS_CM * np.sin(facing_rad)
    theta = utils.M(facing_rad + np.pi - theta_offset)

    if _blocked_by_any_obstacle(map, x, y):
        return None
    return (x, y, theta, obstacle.id)


def obstacle_to_checkpoint_all(map: OccupancyMap, obstacle: Obstacle, theta_offset: float) -> List[Checkpoint]:
    """Kept for API parity with callers that want a list - now there's only
    ever at most one checkpoint per obstacle (see obstacle_to_checkpoint()),
    so this just wraps it in a list."""
    checkpoint = obstacle_to_checkpoint(map, obstacle, theta_offset)
    return [checkpoint] if checkpoint is not None else []
