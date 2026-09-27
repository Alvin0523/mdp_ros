#!/usr/bin/env python3
"""
Hybrid A* for the Ackermann car on the arena costmap (Nav2 Smac Hybrid-A*
style). Used by planner.plan_leg(). Settings: planner.* and robot.* in
mdp_bringup/config/navigation.yaml (utils/params.py).

  * MOTION: from every pose, forward/reverse x left/straight/right, each
    planner.step_size long; one turning radius PER SIDE (left steers harder
    than right on this car) - robot.minimum_turning_radius_* x
    planner.turning_radius_margin.
  * COLLISION: the padded footprint against the costmap (costmap.Costmap:
    blocks and off-arena lethal, inflation around them), as Nav2's
    GridCollisionChecker does it.
  * PRUNING: one state per planner.xy_search_resolution cell and heading bin
    (planner.angle_quantization_bins), kept in a dict.
  * HEURISTIC: max(Reeds-Shepp length, obstacle-aware 2D shortest path to the
    goal on the costmap). The 2D one stops the search wandering into dead ends
    behind blocks.
  * ANALYTIC SHOT: from time to time, and always when close, try a Reeds-Shepp
    curve straight to the goal; the first collision-free one ends the search.
    This is what lands the exact goal pose. As Smac's
    analytic_expansion_max_cost, a shot through a pose costlier than that is
    rejected, except within two turning radii of the goal (checkpoints sit next
    to a block).
  * COST: path length; reverse steps cost planner.reverse_penalty x; plus
    direction-change and steering-change penalties (each direction change is a
    stop). Every step is also scaled by (1 + cost_penalty * costmap cost / 252),
    Smac's cost_penalty, so the search keeps room from blocks when that is cheap.

Units: centimetres and radians (the planner's internal unit). find_path()
returns (list of nodes without the start, None); each node has x, y, theta and
prevAction = (gear, steering).
"""

import heapq
import math
import time
from typing import Callable, List, Optional, Tuple

from ..utils import geometry as utils
from ..utils import params as planner_params
from ..utils.motion_primitives import Gear, Steering
from . import reeds_shepp as rs
from .costmap import ARENA_SIZE_CM, INSCRIBED, LETHAL, MAX_NON_OBSTACLE, Costmap

_INF = float('inf')


class SNode:
    __slots__ = ('x', 'y', 'theta', 'prevAction', 'parent', 'g', 'f')

    def __init__(self, x, y, theta, prevAction, parent=None):
        self.x, self.y, self.theta = x, y, theta
        self.prevAction = prevAction
        self.parent = parent
        self.g = 0.0
        self.f = 0.0


class HybridAStar:
    """Settings default to planner.* / robot.* of navigation.yaml (utils/params.py
    ACTIVE, read here - not at import); any keyword given overrides that one
    setting. Arguments are in cm, the planner's unit."""

    def __init__(self, costmap: Costmap, x_0: float, y_0: float, theta_0: float,
                 x_f: float, y_f: float, theta_f: float,
                 step_cm: float = None, r_left: float = None, r_right: float = None,
                 theta_bins: int = None, xy_res_cm: float = None,
                 steering_change_cost: float = None, gear_change_cost: float = None,
                 reverse_factor: float = None,
                 shot_interval: int = None, shot_range_cm: float = None,
                 max_expansions: int = None, time_limit_s: float = None,
                 progress_callback: Optional[Callable[[List[Tuple[float, float]]], None]] = None,
                 progress_interval: int = 200,
                 params: planner_params.PlannerParams = None):
        p = params or planner_params.ACTIVE

        def pick(value, default):
            return default if value is None else value

        self.costmap = costmap
        self.x0, self.y0, self.th0 = x_0, y_0, theta_0
        self.xf, self.yf, self.thf = x_f, y_f, theta_f
        self.L = pick(step_cm, p.step_size * 100.0)
        self.r_left = pick(r_left, p.plan_turn_radius_left_cm)
        self.r_right = pick(r_right, p.plan_turn_radius_right_cm)
        self.r_min = min(self.r_left, self.r_right)   # smaller radius -> shorter RS length -> admissible
        self.r_shot = max(self.r_left, self.r_right)  # a curve this wide is drivable on both sides
        self.theta_bins = pick(theta_bins, p.angle_quantization_bins)
        self.xy_res = pick(xy_res_cm, p.xy_search_resolution * 100.0)
        self.steering_change_cost = pick(steering_change_cost, p.steering_change_penalty)
        self.gear_change_cost = pick(gear_change_cost, p.change_penalty)
        self.reverse_factor = pick(reverse_factor, p.reverse_penalty)
        self.shot_interval = pick(shot_interval, p.analytic_expansion_interval)
        self.shot_range = pick(shot_range_cm, p.analytic_expansion_max_length * 100.0)
        self.max_expansions = pick(max_expansions, p.max_iterations)
        self.time_limit_s = pick(time_limit_s, p.max_planning_time)
        self.cost_penalty = p.cost_penalty
        self.shot_max_cost = p.analytic_expansion_max_cost
        self.goal_xy_tol = p.goal_xy_tolerance * 100.0
        self.goal_heading_tol = p.goal_yaw_tolerance
        self.progress_callback = progress_callback
        self.progress_interval = progress_interval
        self.expansions = 0
        self.found_by_shot = False

    def _step_cost(self, length, gear, x, y, th, cost=None):
        """Cost of driving `length` cm into pose (x, y, th), or None if the
        footprint there collides. `cost`: the pose's costmap cost, if known."""
        if cost is None:
            cost = self.costmap.footprint_cost(x, y, th)
        if cost >= LETHAL:
            return None
        return (length * (self.reverse_factor if gear == Gear.REVERSE else 1.0)
                * (1.0 + self.cost_penalty * cost / MAX_NON_OBSTACLE))

    # -- motion ------------------------------------------------------------
    def _move(self, x, y, th, gear, steer, ds, radius=None):
        """Advance the rear axle by arc length ds along a straight line or an
        arc of the given side's turning radius (bicycle model)."""
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
        if abs(x - self.xf) > self.goal_xy_tol or abs(y - self.yf) > self.goal_xy_tol:
            return False
        d = abs(utils.M(th - self.thf))
        return d <= self.goal_heading_tol

    # -- heuristic ---------------------------------------------------------
    def _build_2d_table(self):
        """Shortest distance from every 5 cm cell to the goal (8-connected
        Dijkstra) on the costmap, Nav2's obstacle heuristic: a cell whose centre
        is INSCRIBED or LETHAL is blocked (base_link there puts the body on an
        obstacle), and each step is weighted by (1 + cost_penalty * cost / 252)
        like the search itself. The goal cell and its neighbours are never
        blocked."""
        n = int(ARENA_SIZE_CM // self.xy_res)
        res = self.xy_res
        cell_cost = [[self.costmap.cost_at((i + 0.5) * res, (j + 0.5) * res) for j in range(n)]
                     for i in range(n)]
        blocked = [[cell_cost[i][j] >= INSCRIBED for j in range(n)] for i in range(n)]
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
                if 0 <= ni < n and 0 <= nj < n and not blocked[ni][nj]:
                    nd = d + w * (1.0 + self.cost_penalty * cell_cost[ni][nj] / MAX_NON_OBSTACLE)
                    if nd < dist[ni][nj]:
                        dist[ni][nj] = nd
                        heapq.heappush(heap, (nd, ni, nj))
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
        <= 2.5 cm, or None: rejected if any pose collides, or costs more than
        planner.analytic_expansion_max_cost while still over two turning radii from the
        goal (Smac's analytic expansion limits). Its last node's g is the full
        path cost through it."""
        r = self.r_shot
        elements = rs.get_optimal_path((node.x / r, node.y / r, node.theta),
                                       (self.xf / r, self.yf / r, self.thf))
        if not elements:
            return None
        x, y, th = node.x, node.y, node.theta
        out: List[SNode] = []
        parent = node
        g = node.g
        prev = node.prevAction
        for e in elements:
            length = e.param * r
            if length < 1e-6:
                continue
            n = max(1, int(math.ceil(length / 2.5)))
            ds = length / n
            action = (int(e.gear), int(e.steering))
            g += (self.gear_change_cost * abs(prev[0] - action[0])
                  + self.steering_change_cost * abs(prev[1] - action[1]))
            prev = action
            for _ in range(n):
                x, y, th = self._move(x, y, th, e.gear, e.steering, ds, radius=r)
                cost = self.costmap.footprint_cost(x, y, th)
                if (cost > self.shot_max_cost
                        and math.hypot(x - self.xf, y - self.yf) > 2.0 * self.r_min):
                    return None
                step = self._step_cost(ds, e.gear, x, y, th, cost)
                if step is None:
                    return None
                g += step
                nd = SNode(x, y, th, action, parent)
                nd.g = g
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
                    goal_node = shot[-1]                  # Smac: first valid shot ends the search
                    self.found_by_shot = True
                    break

            for gear, steer in choices:
                x, y, th = self._move(node.x, node.y, node.theta, gear, steer, self.L)
                step = self._step_cost(self.L, gear, x, y, th)
                if step is None:
                    continue
                i = min(self._n2d - 1, max(0, int(x // self.xy_res)))
                j = min(self._n2d - 1, max(0, int(y // self.xy_res)))
                if self._dist2d[i][j] == _INF:
                    continue
                g = (node.g + step
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
