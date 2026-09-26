#!/usr/bin/env python3
"""
Hybrid A* for the Ackermann car with real body-footprint collision checking
(2026-09-25). Used by collision_aware_planner.plan_leg(). The older
hybrid_astar.HybridAStar is left as is - hamiltonian.py's time-optimal ordering
still uses it.

What differs from HybridAStar, and why:
  * COLLISION: the whole body rectangle (footprint.pose_collides) at every node,
    against the exact 10 x 10 cm blocks + a pad, instead of one point against
    10 cm inflated cells.
  * TURNING: one radius PER SIDE (left steers harder than right on this car),
    each with headroom, see planning_constants.PLAN_TURN_RADIUS_*.
  * PRUNING: one state per 5 cm cell and 5 degree heading bin (was 10 cm / 15 deg),
    kept in a dict. The old coarse bins could discard the one route that lines up
    with the 3.5 cm goal window, so the goal was sometimes never found.
  * HEURISTIC: max(Reeds-Shepp length, obstacle-aware 2D shortest path to the goal).
    Both never overestimate; the 2D one stops the search wandering into dead ends
    behind blocks.
  * ANALYTIC SHOT: from time to time, and always when close, try a Reeds-Shepp
    curve straight to the goal and accept it if the whole curve is collision-free.
    This is what actually lands the exact goal pose, and it ends the search early.
  * COST: path length, reverse steps cost `reverse_factor` x (the car reverses
    slower and the camera side is blind), plus gear-change and steering-change
    penalties (each gear change is a stop).

Units: centimetres and radians, the OccupancyMap frame. find_path() returns
(list of nodes without the start, None); each node has x, y, theta and
prevAction = (gear, steering) - the same contract as HybridAStar.find_path().
"""

import heapq
import math
import time
from typing import Callable, List, Optional, Tuple

from ..common import geometry_utils as utils
from ..common.motion_primitives import Gear, Steering
from ..common.planning_constants import (
    FOOTPRINT_HALF_WIDTH_CM, FOOTPRINT_REAR_CM, OBSTACLE_PAD_CM,
    PLAN_TURN_RADIUS_LEFT_CM, PLAN_TURN_RADIUS_RIGHT_CM, TABLE_EDGE_MARGIN_CM)
from . import reeds_shepp_curves as rs
from .footprint import pose_collides
from .occupancy_map import ARENA_SIZE_CM, OBSTACLE_SIZE_CM, OccupancyMap

GOAL_XY_TOL_CM = 3.5
GOAL_HEADING_TOL_RAD = math.pi / 9            # 20 degrees - the camera does not need more
_INF = float('inf')


class SNode:
    __slots__ = ('x', 'y', 'theta', 'prevAction', 'parent', 'g', 'f')

    def __init__(self, x, y, theta, prevAction, parent=None):
        self.x, self.y, self.theta = x, y, theta
        self.prevAction = prevAction
        self.parent = parent
        self.g = 0.0
        self.f = 0.0


class FootprintHybridAStar:
    def __init__(self, map: OccupancyMap, x_0: float, y_0: float, theta_0: float,
                 x_f: float, y_f: float, theta_f: float,
                 step_cm: float = 5.0,
                 r_left: float = PLAN_TURN_RADIUS_LEFT_CM,
                 r_right: float = PLAN_TURN_RADIUS_RIGHT_CM,
                 theta_bins: int = 72, xy_res_cm: float = 5.0,
                 steering_change_cost: float = 10.0, gear_change_cost: float = 20.0,
                 reverse_factor: float = 1.5,
                 shot_interval: int = 10, shot_range_cm: float = 90.0,
                 max_expansions: int = 60000, time_limit_s: float = 60.0,
                 progress_callback: Optional[Callable[[List[Tuple[float, float]]], None]] = None,
                 progress_interval: int = 200):
        self.map = map
        self.obstacles = map.obstacles
        self.x0, self.y0, self.th0 = x_0, y_0, theta_0
        self.xf, self.yf, self.thf = x_f, y_f, theta_f
        self.L = step_cm
        self.r_left, self.r_right = r_left, r_right
        self.r_min = min(r_left, r_right)          # smaller radius -> shorter RS length -> admissible
        self.r_shot = max(r_left, r_right)         # a curve this wide is drivable on both sides
        self.theta_bins = theta_bins
        self.xy_res = xy_res_cm
        self.steering_change_cost = steering_change_cost
        self.gear_change_cost = gear_change_cost
        self.reverse_factor = reverse_factor
        self.shot_interval = shot_interval
        self.shot_range = shot_range_cm
        self.max_expansions = max_expansions
        self.time_limit_s = time_limit_s
        self.progress_callback = progress_callback
        self.progress_interval = progress_interval
        self.expansions = 0
        self.found_by_shot = False

    # -- motion ------------------------------------------------------------
    def _move(self, x, y, th, gear, steer, ds, radius=None):
        """Advance the rear axle by arc length ds. Same kinematics as
        HybridAStar.calculate_next_node, but with a per-side radius."""
        if steer == Steering.STRAIGHT:
            return x + gear * ds * math.cos(th), y + gear * ds * math.sin(th), th
        r = radius if radius is not None else (self.r_left if steer == Steering.LEFT else self.r_right)
        xc = x + steer * r * math.sin(th)
        yc = y - steer * r * math.cos(th)
        a = gear * (-steer * ds / r)
        xa, ya = x - xc, y - yc
        ca, sa = math.cos(a), math.sin(a)
        return xc + xa * ca - ya * sa, yc + xa * sa + ya * ca, utils.M(th + a)

    def _key(self, x, y, th):
        tb = int(((th + math.pi) / (2.0 * math.pi)) * self.theta_bins) % self.theta_bins
        return int(x // self.xy_res), int(y // self.xy_res), tb

    def _at_goal(self, x, y, th):
        if abs(x - self.xf) > GOAL_XY_TOL_CM or abs(y - self.yf) > GOAL_XY_TOL_CM:
            return False
        d = abs(utils.M(th - self.thf))
        return d <= GOAL_HEADING_TOL_RAD

    # -- heuristic ---------------------------------------------------------
    def _build_2d_table(self):
        """Shortest obstacle-avoiding distance from every 5 cm cell to the goal
        (8-connected Dijkstra). A cell is blocked if a body centred there could
        not fit at ANY heading: within block half + body half-width + pad of a
        block (Chebyshev), or within a rear-overhang of the table edge. The goal
        cell and its neighbours are never blocked."""
        n = int(ARENA_SIZE_CM // self.xy_res)
        res = self.xy_res
        clear = OBSTACLE_SIZE_CM / 2.0 + FOOTPRINT_HALF_WIDTH_CM + OBSTACLE_PAD_CM
        edge = FOOTPRINT_REAR_CM + TABLE_EDGE_MARGIN_CM
        blocked = [[False] * n for _ in range(n)]
        for i in range(n):
            cx = (i + 0.5) * res
            for j in range(n):
                cy = (j + 0.5) * res
                if cx < edge or cx > ARENA_SIZE_CM - edge or cy < edge or cy > ARENA_SIZE_CM - edge:
                    blocked[i][j] = True
                    continue
                for o in self.obstacles:
                    if abs(cx - o.x_cm) < clear and abs(cy - o.y_cm) < clear:
                        blocked[i][j] = True
                        break
        gi = min(n - 1, max(0, int(self.xf // res)))
        gj = min(n - 1, max(0, int(self.yf // res)))
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if 0 <= gi + di < n and 0 <= gj + dj < n:
                    blocked[gi + di][gj + dj] = False
        dist = [[_INF] * n for _ in range(n)]
        dist[gi][gj] = 0.0
        heap = [(0.0, gi, gj)]
        diag = res * math.sqrt(2.0)
        while heap:
            d, i, j = heapq.heappop(heap)
            if d > dist[i][j]:
                continue
            for di, dj, w in ((1, 0, res), (-1, 0, res), (0, 1, res), (0, -1, res),
                              (1, 1, diag), (1, -1, diag), (-1, 1, diag), (-1, -1, diag)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n and not blocked[ni][nj] and d + w < dist[ni][nj]:
                    dist[ni][nj] = d + w
                    heapq.heappush(heap, (d + w, ni, nj))
        self._n2d = n
        return dist

    def _h(self, x, y, th):
        h_rs = rs.get_optimal_path_length((x, y, th), (self.xf, self.yf, self.thf), self.r_min)
        i = min(self._n2d - 1, max(0, int(x // self.xy_res)))
        j = min(self._n2d - 1, max(0, int(y // self.xy_res)))
        return max(h_rs, self._dist2d[i][j])

    # -- analytic shot -----------------------------------------------------
    def _shot(self, node: SNode) -> Optional[List[SNode]]:
        """A collision-free Reeds-Shepp curve from `node` to the goal, sampled every
        <= 2.5 cm, or None."""
        r = self.r_shot
        elements = rs.get_optimal_path((node.x / r, node.y / r, node.theta),
                                       (self.xf / r, self.yf / r, self.thf))
        if not elements:
            return None
        x, y, th = node.x, node.y, node.theta
        out: List[SNode] = []
        parent = node
        for e in elements:
            length = e.param * r
            if length < 1e-6:
                continue
            n = max(1, int(math.ceil(length / 2.5)))
            ds = length / n
            for _ in range(n):
                x, y, th = self._move(x, y, th, e.gear, e.steering, ds, radius=r)
                if pose_collides(self.obstacles, x, y, th):
                    return None
                nd = SNode(x, y, th, (int(e.gear), int(e.steering)), parent)
                out.append(nd)
                parent = nd
        if not out or not self._at_goal(out[-1].x, out[-1].y, out[-1].theta):
            return None
        return out

    # -- search ------------------------------------------------------------
    def find_path(self) -> Tuple[Optional[List[SNode]], None]:
        t_start = time.time()
        self._dist2d = self._build_2d_table()
        start = SNode(self.x0, self.y0, self.th0, (Gear.FORWARD, Steering.STRAIGHT))
        start.f = self._h(start.x, start.y, start.theta)
        if self._dist2d[min(self._n2d - 1, int(self.x0 // self.xy_res))][
                min(self._n2d - 1, int(self.y0 // self.xy_res))] == _INF:
            return None, None     # the start cell cannot reach the goal cell at all

        choices = [(g, s) for g in (Gear.FORWARD, Gear.REVERSE)
                   for s in (Steering.LEFT, Steering.STRAIGHT, Steering.RIGHT)]
        counter = 0
        heap = [(start.f, counter, start)]
        best_g = {self._key(start.x, start.y, start.theta): 0.0}
        explored: List[Tuple[float, float]] = []
        goal_node: Optional[SNode] = None

        while heap:
            _, _, node = heapq.heappop(heap)
            key = self._key(node.x, node.y, node.theta)
            if node.g > best_g.get(key, _INF) + 1e-9:
                continue                                  # a better route to this state exists
            self.expansions += 1
            if self.expansions > self.max_expansions or time.time() - t_start > self.time_limit_s:
                break

            if self._at_goal(node.x, node.y, node.theta):
                goal_node = node
                break

            if self.progress_callback is not None:
                explored.append((node.x, node.y))
                if self.expansions % self.progress_interval == 0:
                    self.progress_callback(explored)

            dist_goal = math.hypot(node.x - self.xf, node.y - self.yf)
            if dist_goal <= self.shot_range and (self.expansions % self.shot_interval == 0
                                                 or dist_goal < 40.0):
                shot = self._shot(node)
                if shot:
                    goal_node = shot[-1]
                    self.found_by_shot = True
                    break

            for gear, steer in choices:
                x, y, th = self._move(node.x, node.y, node.theta, gear, steer, self.L)
                if pose_collides(self.obstacles, x, y, th):
                    continue
                i = min(self._n2d - 1, max(0, int(x // self.xy_res)))
                j = min(self._n2d - 1, max(0, int(y // self.xy_res)))
                if self._dist2d[i][j] == _INF:
                    continue
                g = (node.g + self.L * (self.reverse_factor if gear == Gear.REVERSE else 1.0)
                     + self.gear_change_cost * abs(node.prevAction[0] - gear)
                     + self.steering_change_cost * abs(node.prevAction[1] - steer))
                child_key = self._key(x, y, th)
                if g >= best_g.get(child_key, _INF):
                    continue
                best_g[child_key] = g
                child = SNode(x, y, th, (gear, steer), node)
                child.g = g
                child.f = g + self._h(x, y, th)
                counter += 1
                heapq.heappush(heap, (child.f, counter, child))

        if self.progress_callback is not None:
            self.progress_callback(explored)
        if goal_node is None:
            return None, None
        path = []
        n = goal_node
        while n.parent is not None:
            path.append(n)
            n = n.parent
        path.reverse()
        return path, None
