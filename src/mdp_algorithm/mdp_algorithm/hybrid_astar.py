#!/usr/bin/env python3
"""
Hybrid A* search: 6-primitive (forward/reverse x left/straight/right) grid
search with a Reeds-Shepp-heuristic-admissible cost, producing a
collision-checked, kinematically-feasible path between two poses.

Ported from the teammate's mdp_algo package (pathfinding/hybrid_astar.py) -
matplotlib import/plotting removed (this runs headless on the Pi), the
"to remove" matplotlib import the teammate had already flagged is gone,
and utils.normalise_theta() (an angle-wrap helper that looked error-prone
to hand-port faithfully - see geometry_utils.py) is replaced with the
simpler, verified M() wrap, which serves the same purpose here (keeping
accumulated heading in a canonical range across primitives).

All positions/lengths are in CENTIMETRES (see occupancy_map.py's module
docstring) - x_0/y_0/x_f/y_f are the REAR AXLE position, matching the
teammate's original convention.
"""

import heapq
import math
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import geometry_utils as utils
from . import reeds_shepp_curves as rs
from .motion_primitives import Gear, Steering
from .occupancy_map import GRID_SIZE, OccupancyMap, coords_to_grid
from .planning_constants import REAR_AXLE_TO_CENTER_CM


# How far off the checkpoint's exact heading is still "close enough" to
# count as arrived, for Node.__eq__'s goal check below. The camera has a
# real field of view - it doesn't need to be perfectly perpendicular to the
# obstacle's image to capture it, so the goal doesn't need to be that
# precise either. 20deg per direct user instruction (2026-09-04) - was
# 7.5deg (pi/24), which forced Hybrid A* to hunt for a near-exact heading
# match that was never actually required by the camera itself. A looser
# tolerance also gives the search more valid final poses to land on, which
# should help the per-leg search time too, not just goal precision.
GOAL_HEADING_TOLERANCE_RAD = math.pi / 9  # 20 degrees


class Node:
    def __init__(self, x: float, y: float, theta: float, prevAction, parent=None) -> None:
        self.x = x
        self.y = y
        self.theta = theta
        self.x_g, self.y_g, self.theta_g = self._discretize(x, y, theta)
        self.parent = parent
        self.prevAction = prevAction
        self.g = 0.0
        self.h = 0.0
        self.f = 0.0

    @staticmethod
    def _discretize(x, y, theta, thetaBins=24):
        # Was a duplicated inline copy of this formula (hardcoded 200/GRID_SIZE,
        # no margin offset) - found during the 2026-09-04 arena-padding change
        # (occupancy_map.py's module docstring) to silently drift out of sync:
        # GRID_SIZE now refers to the padded array, and any pose planned in
        # the new margin needs the offset coords_to_grid() applies, or this
        # would produce a negative/wrapped array index into openList/closedList.
        x_g, y_g = coords_to_grid(x, y)
        theta_g = int(((theta * 180 / math.pi + 180) // (360 / thetaBins)))
        return x_g, y_g, theta_g

    def __eq__(self, other):
        return (abs(self.x - other.x) <= 3.5 and abs(self.y - other.y) <= 3.5
                and (abs(self.theta - other.theta) <= GOAL_HEADING_TOLERANCE_RAD
                     or abs(abs(self.theta - other.theta) - 2 * math.pi) <= GOAL_HEADING_TOLERANCE_RAD))

    def __lt__(self, other):
        return self.f < other.f


class HybridAStar:
    def __init__(self, map: OccupancyMap, x_0: float, y_0: float, theta_0: float,
                 x_f: float, y_f: float, theta_f: float, theta_offset: float = 0.0,
                 steeringChangeCost: float = 10, gearChangeCost: float = 20,
                 L: float = 5, minR: float = 25, heuristic: str = 'hybriddiag',
                 simulate: bool = False, thetaBins: int = 24,
                 cost_mode: str = 'distance', forward_speed: float = 20.,
                 reverse_speed: float = 15., gear_change_time: float = .5,
                 steering_change_time: float = .15,
                 progress_callback: Optional[Callable[[List[Tuple[float, float]]], None]] = None,
                 progress_interval: int = 200):
        """
        Args:
            map: OccupancyMap to search/collide against.
            x_0/y_0/theta_0: starting rear-axle pose (cm, cm, rad).
            x_f/y_f/theta_f: goal rear-axle pose (cm, cm, rad).
            L: distance travelled per primitive step, cm.
            minR: minimum turning radius, cm - see planning_constants.py's
                MIN_TURN_RADIUS_CM for how this is set for this chassis.
            progress_callback: if given, called every `progress_interval`
                node expansions with the full list of (x, y) points explored
                SO FAR (cm, cheap - just the two floats per node, not the
                whole Node object) - lets a caller (task1_runner.py)
                publish live search progress to Foxglove instead of the
                search being a black box until it returns. None (default)
                adds no overhead - the list still gets built (cheap append),
                just never handed anywhere.
            progress_interval: how many node expansions between callback
                calls - 200 is frequent enough to feel "live" without the
                callback itself (a ROS publish) becoming the bottleneck.
        """
        self.map = map
        self.x, self.y, self.theta = x_0, y_0, theta_0
        self.x_f, self.y_f, self.theta_f = x_f, y_f, theta_f
        self.theta_offset = theta_offset
        self.steeringChangeCost = steeringChangeCost
        self.gearChangeCost = gearChangeCost
        self.L = L
        self.minR = minR
        self.heuristic = heuristic
        self.simulate = simulate
        self.thetaBins = thetaBins
        self.cost_mode = cost_mode
        self.forward_speed = forward_speed
        self.reverse_speed = reverse_speed
        self.gear_change_time = gear_change_time
        self.steering_change_time = steering_change_time
        self.progress_callback = progress_callback
        self.progress_interval = progress_interval

    def transition_cost(self, previous_action, action) -> float:
        if self.cost_mode == 'time':
            speed = self.forward_speed if action[0] == Gear.FORWARD else self.reverse_speed
            cost = self.L / speed
            if previous_action[0] != action[0]:
                cost += self.gear_change_time
            if previous_action[1] != action[1]:
                cost += self.steering_change_time
            return cost
        return (self.L + self.gearChangeCost * abs(previous_action[0] - action[0])
                + self.steeringChangeCost * abs(previous_action[1] - action[1]))

    def _heuristic(self, childNode: Node, endNode: Node) -> float:
        if self.cost_mode == 'time':
            return utils.l2(childNode.x, childNode.y, endNode.x, endNode.y) / max(self.forward_speed, self.reverse_speed)
        if self.heuristic == 'euclidean':
            return utils.l2(childNode.x, childNode.y, endNode.x, endNode.y)
        if self.heuristic == 'manhattan':
            return utils.l1(childNode.x, childNode.y, endNode.x, endNode.y)
        if self.heuristic == 'diag':
            return utils.diag_dist(childNode.x, childNode.y, endNode.x, endNode.y)
        if self.heuristic == 'reeds-shepp':
            return rs.get_optimal_path_length((childNode.x, childNode.y, childNode.theta),
                                               (endNode.x, endNode.y, endNode.theta), self.minR)
        if self.heuristic == 'hybridl2':
            return max(utils.l2(childNode.x, childNode.y, endNode.x, endNode.y),
                       rs.get_optimal_path_length((childNode.x, childNode.y, childNode.theta),
                                                   (endNode.x, endNode.y, endNode.theta), self.minR))
        if self.heuristic == 'hybridl1':
            return min(utils.l1(childNode.x, childNode.y, endNode.x, endNode.y),
                       rs.get_optimal_path_length((childNode.x, childNode.y, childNode.theta),
                                                   (endNode.x, endNode.y, endNode.theta), self.minR))
        if self.heuristic == 'hybriddiag':
            return min(utils.diag_dist(childNode.x, childNode.y, endNode.x, endNode.y),
                       rs.get_optimal_path_length((childNode.x, childNode.y, childNode.theta),
                                                   (endNode.x, endNode.y, endNode.theta), self.minR))
        return 0.0  # 'greedy'

    def find_path(self) -> Tuple[Optional[List[Node]], Optional[List[Node]]]:
        pathHistory = []
        gearChoices = [Gear.FORWARD, Gear.REVERSE]
        steeringChoices = [Steering.LEFT, Steering.STRAIGHT, Steering.RIGHT]
        choices = [(gear, steering) for gear in gearChoices for steering in steeringChoices]

        startNode = Node(self.x, self.y, self.theta, (Gear.FORWARD, Steering.STRAIGHT))
        endNode = Node(self.x_f, self.y_f, self.theta_f, (Gear.FORWARD, Steering.STRAIGHT))

        # heapq, not queue.PriorityQueue - PriorityQueue's lock-per-push/pop
        # is pure overhead here (single-threaded search), and this heap gets
        # pushed to on every one of the up to 6 children of every expanded
        # node, so that overhead was compounding across the whole search.
        open_heap: List[Tuple[float, "Node"]] = []
        openList = 999999 * np.ones((GRID_SIZE, GRID_SIZE, self.thetaBins + 1))
        closedList = 999999 * np.ones((GRID_SIZE, GRID_SIZE, self.thetaBins + 1))

        heapq.heappush(open_heap, (startNode.f, startNode))
        pathFound = False
        nodesExpanded = 0
        currentNode = startNode
        explored_points: List[Tuple[float, float]] = []   # (x, y) cm, one per expansion - for progress_callback

        while open_heap and not pathFound:
            currentNode = heapq.heappop(open_heap)[1]
            openList[currentNode.x_g, currentNode.y_g, currentNode.theta_g] = 999999
            nodesExpanded += 1

            if endNode == currentNode:
                pathFound = True
                break

            if self.simulate:
                pathHistory.append(currentNode)

            if self.progress_callback is not None:
                explored_points.append((currentNode.x, currentNode.y))
                if nodesExpanded % self.progress_interval == 0:
                    self.progress_callback(explored_points)

            for choice in choices:
                if choice[0] == -currentNode.prevAction[0] and choice[1] == -currentNode.prevAction[1]:
                    continue  # no immediate direction reversal on the same primitive

                x_child, y_child, theta_child = self.calculate_next_node(currentNode, choice)

                front_x = x_child + REAR_AXLE_TO_CENTER_CM * np.cos(theta_child)
                front_y = y_child + REAR_AXLE_TO_CENTER_CM * np.sin(theta_child)
                if self.map.collide_with_point(front_x, front_y):
                    continue

                childNode = Node(x_child, y_child, theta_child, prevAction=choice, parent=currentNode)
                childNode.g = currentNode.g + self.transition_cost(currentNode.prevAction, choice)
                childNode.h = self._heuristic(childNode, endNode)
                childNode.f = childNode.g + childNode.h

                out_of_bounds = (childNode.x_g < 0 or childNode.x_g >= GRID_SIZE
                                  or childNode.y_g < 0 or childNode.y_g >= GRID_SIZE)
                if out_of_bounds or openList[childNode.x_g, childNode.y_g, childNode.theta_g] <= childNode.g:
                    continue
                if out_of_bounds or closedList[childNode.x_g, childNode.y_g, childNode.theta_g] <= childNode.g:
                    continue

                heapq.heappush(open_heap, (childNode.f, childNode))
                openList[childNode.x_g, childNode.y_g, childNode.theta_g] = childNode.g

            closedList[currentNode.x_g, currentNode.y_g, currentNode.theta_g] = currentNode.g

        if self.progress_callback is not None:
            self.progress_callback(explored_points)   # final call - last partial batch + lets caller know it's done

        path = None
        if pathFound:
            path = []
            node = currentNode
            while node != startNode:
                path.append(node)
                node = node.parent
            path.reverse()

        if self.simulate:
            return path, pathHistory
        return path, None

    def calculate_next_node(self, currentNode: Node, choice) -> Tuple[float, float, float]:
        gear, steering = choice

        if steering == Steering.STRAIGHT:
            x_b = currentNode.x + gear * self.L * np.cos(currentNode.theta)
            y_b = currentNode.y + gear * self.L * np.sin(currentNode.theta)
            theta_b = currentNode.theta
            return x_b, y_b, theta_b

        x_c = currentNode.x + steering * self.minR * np.sin(currentNode.theta)
        y_c = currentNode.y - steering * self.minR * np.cos(currentNode.theta)

        theta_t = -steering * self.L / self.minR
        theta_b = utils.M(currentNode.theta + gear * theta_t)

        x_ca = currentNode.x - x_c
        y_ca = currentNode.y - y_c

        x_b = x_c + (x_ca * np.cos(gear * theta_t) - y_ca * np.sin(gear * theta_t))
        y_b = y_c + (x_ca * np.sin(gear * theta_t) + y_ca * np.cos(gear * theta_t))

        return x_b, y_b, theta_b
