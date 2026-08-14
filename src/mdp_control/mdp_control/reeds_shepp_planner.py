#!/usr/bin/env python3
"""
Dubins path planner for a forward-driving Ackermann vehicle.
Produces (x, y, yaw) paths between poses that never require a curvature
tighter than the vehicle's minimum turning radius (min_turn_radius =
wheelbase / tan(max_steering_angle)), unlike naive point-to-point
interpolation.
"""

import itertools
import math
from typing import List, Optional, Tuple

Pose = Tuple[float, float, float]


def _pi_2_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _mod2pi(theta: float) -> float:
    return theta - 2.0 * math.pi * math.floor(theta / (2.0 * math.pi))


class DubinsPathPlanner:
    def __init__(self, min_turn_radius: float = 0.35):
        self.r_min = min_turn_radius

    # -- standoff / TSP helpers (used for task-level obstacle-face planning) --
    def calc_standoff_pose(self, x: float, y: float, face: str, distance: float = 0.30) -> Pose:
        face = face.upper()
        if face in ['N', 'UP', 'NORTH']:
            return (x, y + distance, -math.pi / 2)
        elif face in ['S', 'DOWN', 'SOUTH']:
            return (x, y - distance, math.pi / 2)
        elif face in ['E', 'RIGHT', 'EAST']:
            return (x + distance, y, math.pi)
        elif face in ['W', 'LEFT', 'WEST']:
            return (x - distance, y, 0.0)
        return (x, y + distance, -math.pi / 2)

    def path_length(self, p1: Pose, p2: Pose) -> float:
        segs = self._shortest_dubins(p1, p2)
        if segs is None:
            return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        d1, d2, d3, _ = segs
        return (d1 + d2 + d3) * self.r_min

    def reeds_shepp_distance(self, p1: Pose, p2: Pose) -> float:
        return self.path_length(p1, p2)

    def solve_tsp(self, start_pose: Pose, target_poses: List[Pose]) -> List[int]:
        best_order = list(range(len(target_poses)))
        best_cost = float('inf')
        for perm in itertools.permutations(range(len(target_poses))):
            cost = 0.0
            cur = start_pose
            for idx in perm:
                cost += self.path_length(cur, target_poses[idx])
                cur = target_poses[idx]
            if cost < best_cost:
                best_cost, best_order = cost, list(perm)
        return best_order

    # -- Dubins primitives (Shkel & Lumelsky / LaValle formulation) -----------
    @staticmethod
    def _LSL(alpha, beta, d):
        sa, sb, ca, cb = math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta)
        c_ab = math.cos(alpha - beta)
        p_sq = 2 + d ** 2 - 2 * c_ab + 2 * d * (sa - sb)
        if p_sq < 0:
            return None
        tmp = math.atan2(cb - ca, d + sa - sb)
        return _mod2pi(-alpha + tmp), math.sqrt(p_sq), _mod2pi(beta - tmp), ('L', 'S', 'L')

    @staticmethod
    def _RSR(alpha, beta, d):
        sa, sb, ca, cb = math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta)
        c_ab = math.cos(alpha - beta)
        p_sq = 2 + d ** 2 - 2 * c_ab + 2 * d * (sb - sa)
        if p_sq < 0:
            return None
        tmp = math.atan2(ca - cb, d - sa + sb)
        return _mod2pi(alpha - tmp), math.sqrt(p_sq), _mod2pi(-beta + tmp), ('R', 'S', 'R')

    @staticmethod
    def _LSR(alpha, beta, d):
        sa, sb, ca, cb = math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta)
        c_ab = math.cos(alpha - beta)
        p_sq = -2 + d ** 2 + 2 * c_ab + 2 * d * (sa + sb)
        if p_sq < 0:
            return None
        p = math.sqrt(p_sq)
        tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        return _mod2pi(-alpha + tmp), p, _mod2pi(-_mod2pi(beta) + tmp), ('L', 'S', 'R')

    @staticmethod
    def _RSL(alpha, beta, d):
        sa, sb, ca, cb = math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta)
        c_ab = math.cos(alpha - beta)
        p_sq = d ** 2 - 2 + 2 * c_ab - 2 * d * (sa + sb)
        if p_sq < 0:
            return None
        p = math.sqrt(p_sq)
        tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        return _mod2pi(alpha - tmp), p, _mod2pi(beta - tmp), ('R', 'S', 'L')

    @staticmethod
    def _RLR(alpha, beta, d):
        sa, sb, ca, cb = math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta)
        c_ab = math.cos(alpha - beta)
        tmp = (6.0 - d ** 2 + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
        if abs(tmp) > 1.0:
            return None
        d2 = _mod2pi(2 * math.pi - math.acos(tmp))
        d1 = _mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + d2 / 2.0)
        d3 = _mod2pi(alpha - beta - d1 + d2)
        return d1, d2, d3, ('R', 'L', 'R')

    @staticmethod
    def _LRL(alpha, beta, d):
        sa, sb, ca, cb = math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta)
        c_ab = math.cos(alpha - beta)
        tmp = (6.0 - d ** 2 + 2.0 * c_ab + 2.0 * d * (-sa + sb)) / 8.0
        if abs(tmp) > 1.0:
            return None
        d2 = _mod2pi(2 * math.pi - math.acos(tmp))
        d1 = _mod2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + d2 / 2.0)
        d3 = _mod2pi(_mod2pi(beta) - alpha - d1 + d2)
        return d1, d2, d3, ('L', 'R', 'L')

    def _shortest_dubins(self, p1: Pose, p2: Pose) -> Optional[Tuple[float, float, float, Tuple[str, str, str]]]:
        x1, y1, yaw1 = p1
        x2, y2, yaw2 = p2
        dx, dy = x2 - x1, y2 - y1
        d = math.hypot(dx, dy) / self.r_min
        theta = _mod2pi(math.atan2(dy, dx))
        alpha = _mod2pi(yaw1 - theta)
        beta = _mod2pi(yaw2 - theta)

        best = None
        for planner in (self._LSL, self._RSR, self._LSR, self._RSL, self._RLR, self._LRL):
            res = planner(alpha, beta, d)
            if res is None:
                continue
            d1, d2, d3, mode = res
            cost = abs(d1) + abs(d2) + abs(d3)
            if best is None or cost < best[0]:
                best = (cost, d1, d2, d3, mode)

        if best is None:
            return None
        _, d1, d2, d3, mode = best
        return d1, d2, d3, mode

    def generate_dubins_path(self, p1: Pose, p2: Pose, step_size: float = 0.05) -> List[Pose]:
        """Samples the shortest curvature-feasible Dubins path from pose p1 to p2."""
        segs = self._shortest_dubins(p1, p2)
        if segs is None:
            return [p1]
        d1, d2, d3, mode = segs
        lengths = (d1 * self.r_min, d2 * self.r_min, d3 * self.r_min)

        x, y, yaw = p1
        path = [(x, y, yaw)]
        for length, turn in zip(lengths, mode):
            n_steps = max(1, int(round(length / step_size)))
            step = length / n_steps
            for _ in range(n_steps):
                if turn == 'S':
                    x += step * math.cos(yaw)
                    y += step * math.sin(yaw)
                else:
                    dyaw = step / self.r_min
                    if turn == 'L':
                        yaw_mid = yaw + dyaw / 2.0
                        yaw += dyaw
                    else:  # 'R'
                        yaw_mid = yaw - dyaw / 2.0
                        yaw -= dyaw
                    x += step * math.cos(yaw_mid)
                    y += step * math.sin(yaw_mid)
                    yaw = _pi_2_pi(yaw)
                path.append((x, y, yaw))
        return path

    def generate_path_through_waypoints(self, waypoints_xy: List[Tuple[float, float]],
                                         start_yaw: float, step_size: float = 0.05) -> List[Pose]:
        """Chains Dubins segments through a list of (x, y) waypoints. Heading at each
        interior waypoint is estimated from the direction between its neighbours, so
        the dense path passes through every waypoint exactly while never demanding
        more curvature than min_turn_radius allows at the junctions."""
        n = len(waypoints_xy)
        if n == 0:
            return []
        if n == 1:
            return [(waypoints_xy[0][0], waypoints_xy[0][1], start_yaw)]

        # Heading at each interior waypoint = circular mean of the incoming and
        # outgoing segment directions. Using the direct prev->next vector instead
        # (skipping the point itself) badly misestimates the heading at "peak"
        # waypoints - e.g. a lateral detour to clear an obstacle - which forces
        # Dubins into huge unnecessary loops to reconcile the wrong heading.
        yaws = [start_yaw]
        for i in range(1, n - 1):
            ix, iy = waypoints_xy[i]
            px, py = waypoints_xy[i - 1]
            nx, ny = waypoints_xy[i + 1]
            in_dir = math.atan2(iy - py, ix - px)
            out_dir = math.atan2(ny - iy, nx - ix)
            yaws.append(math.atan2(math.sin(in_dir) + math.sin(out_dir),
                                    math.cos(in_dir) + math.cos(out_dir)))
        last_x, last_y = waypoints_xy[-1]
        prev_x, prev_y = waypoints_xy[-2]
        yaws.append(math.atan2(last_y - prev_y, last_x - prev_x))

        dense_path: List[Pose] = []
        for i in range(n - 1):
            p1 = (waypoints_xy[i][0], waypoints_xy[i][1], yaws[i])
            p2 = (waypoints_xy[i + 1][0], waypoints_xy[i + 1][1], yaws[i + 1])
            seg = self.generate_dubins_path(p1, p2, step_size=step_size)
            if dense_path and seg:
                seg = seg[1:]  # drop duplicate junction point
            dense_path.extend(seg)
        return dense_path


# Kept for backwards compatibility with the previous (fake) planner's class name.
ReedsSheppPlanner = DubinsPathPlanner


def main():
    planner = DubinsPathPlanner(min_turn_radius=0.35)
    start = (0.0, 0.0, math.pi / 2)
    obstacles = [
        (0.5, 1.3, 'LEFT'),
        (0.5, 0.7, 'DOWN'),
        (1.2, 0.9, 'RIGHT'),
        (1.5, 1.5, 'DOWN'),
        (1.5, 0.4, 'UP')
    ]
    standoffs = [planner.calc_standoff_pose(x, y, f) for x, y, f in obstacles]
    order = planner.solve_tsp(start, standoffs)
    print(f"Optimal TSP Visiting Sequence: {order}")


if __name__ == '__main__':
    main()

