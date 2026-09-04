import random
import itertools
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__ + '\..')))

from algo import utils
from algo.objects.Obstacle import Obstacle
from algo.enumerations import Gear, Steering
import algo.pathfinding.reeds_shepp as rs
from algo import constants as c
import copy

class Hamiltonian():
    def __init__(self, map, obstacles, x_start, y_start, theta_start, theta_offset=0, 
                          metric='euclidean', minR=25) -> None:
        assert -np.pi < theta_start <= np.pi
        assert -np.pi < theta_offset <= np.pi
        self.map = map
        self.obstacles = obstacles
        self.checkpoints = []
        self.start = (x_start, y_start, theta_start)
        self.theta_offset = theta_offset
        self.metric = metric
        self.minR = minR
        # ADDED: Planners can report image positions that cannot be safely scanned.
        self.unreachable_obstacles = []

    def _reachable_obstacles(self):
        """Return obstacles with at least one valid image-recognition checkpoint."""
        reachable = []
        self.unreachable_obstacles = []
        for obstacle in self.obstacles:
            if obstacle_to_checkpoint(self.map, obstacle, self.theta_offset) is None:
                self.unreachable_obstacles.append(obstacle)
            else:
                reachable.append(obstacle)
        return reachable

    def find_brute_force_path(self):
        # CHANGED: Compute an exact Hamiltonian/TSP ordering only across image
        # positions that have a valid, collision-free recognition checkpoint.
        reachable = self._reachable_obstacles()
        if not reachable:
            return []

        obstacle_permutations = itertools.permutations(reachable)
        shortest_distance = float('inf')
        shortest_path = []
        for obstacle_path in obstacle_permutations:
            current_pos = self.start
            total_distance = 0
            for obstacle in obstacle_path:
                checkpoint = obstacle_to_checkpoint(self.map, obstacle, self.theta_offset)
                if self.metric == 'euclidean':
                    distance = utils.l2(current_pos[0], current_pos[1], checkpoint[0], checkpoint[1])
                elif self.metric == 'reeds-shepp':
                    distance = rs.get_optimal_path_length(current_pos, checkpoint, self.minR)

                total_distance += distance
                current_pos = checkpoint
            if total_distance < shortest_distance:
                shortest_distance = total_distance
                shortest_path = obstacle_path[:]
        return list(shortest_path)

    def find_nearest_neighbor_path(self):
        current_pos = self.start
        path = []

        # CHANGED: Do not mutate a collection while iterating over it. This
        # keeps the greedy fallback deterministic and reports unreachable images.
        obstacles = self._reachable_obstacles()
        while obstacles:
            nearest_neighbor = None
            minDist = float('inf')
            for obstacle in obstacles:
                checkpoint = obstacle_to_checkpoint(self.map, obstacle, self.theta_offset)
                if self.metric == 'euclidean':
                    dist = utils.l2(current_pos[0], current_pos[1], checkpoint[0], checkpoint[1])
                elif self.metric == 'reeds-shepp':
                    dist = rs.get_optimal_path_length(current_pos, checkpoint, self.minR)
                
                if dist < minDist:
                    minDist = dist
                    nearest_neighbor = obstacle
                
            
            if nearest_neighbor is None:
                break
            path.append(nearest_neighbor)
            obstacles.remove(nearest_neighbor)
            current_pos = obstacle_to_checkpoint(self.map, nearest_neighbor, self.theta_offset)
        
        return path

def obstacle_to_checkpoint(map, obstacle: Obstacle, theta_offset):
    starting_x, starting_y = utils.grid_to_coords(obstacle.x_g, obstacle.y_g)
    starting_x += offset_x(obstacle.facing)
    starting_y += offset_y(obstacle.facing)
    # The checkpoint lies outward from the side carrying the image. The robot's
    # front-mounted camera points back toward the obstacle at the checkpoint.
    starting_image_to_pos_theta = offset_theta(obstacle.facing, 0)

    theta_scan_list = [0]
    r_scan_list = [c.IMAGE_DISTANCE_CM]

    
    for r_scan in r_scan_list:
        for theta_scan in theta_scan_list:
            cur_image_to_pos_theta = utils.M(starting_image_to_pos_theta + theta_scan)
            cur_x = starting_x + r_scan*np.cos(cur_image_to_pos_theta)
            cur_y = starting_y + r_scan*np.sin(cur_image_to_pos_theta)
            theta = utils.M(cur_image_to_pos_theta + np.pi - theta_offset)

            if not map.collide_with_point(cur_x, cur_y) and not \
                map.collide_with_point(cur_x + 0.5*c.REAR_AXLE_TO_CENTER*np.cos(theta), cur_y + 0.5*c.REAR_AXLE_TO_CENTER*np.sin(theta)) and not \
                map.collide_with_point(cur_x - 0.5*c.REAR_AXLE_TO_CENTER*np.cos(theta), cur_y - 0.5*c.REAR_AXLE_TO_CENTER*np.sin(theta)):
                
                cur_x -= c.REAR_AXLE_TO_CENTER*np.cos(theta)
                cur_y -= c.REAR_AXLE_TO_CENTER*np.sin(theta)
                return (cur_x, cur_y, theta, obstacle.id)

    return None

def obstacle_to_checkpoint_all(map, obstacle: Obstacle, theta_offset):
    starting_x, starting_y = utils.grid_to_coords(obstacle.x_g, obstacle.y_g)
    starting_x += offset_x(obstacle.facing)
    starting_y += offset_y(obstacle.facing)
    starting_image_to_pos_theta = offset_theta(obstacle.facing, 0)

    valid_checkpoints = []

    theta_scan_list = [0]
    r_scan_list = [c.IMAGE_DISTANCE_CM]
    
    
    for r_scan in r_scan_list:
        for theta_scan in theta_scan_list:
            cur_image_to_pos_theta = utils.M(starting_image_to_pos_theta + theta_scan)
            cur_x = starting_x + r_scan*np.cos(cur_image_to_pos_theta)
            cur_y = starting_y + r_scan*np.sin(cur_image_to_pos_theta)
            theta = utils.M(cur_image_to_pos_theta + np.pi - theta_offset)

            if not map.collide_with_point(cur_x, cur_y) and not \
                map.collide_with_point(cur_x + 0.5*c.REAR_AXLE_TO_CENTER*np.cos(theta), cur_y + 0.5*c.REAR_AXLE_TO_CENTER*np.sin(theta)) and not \
                map.collide_with_point(cur_x - 0.5*c.REAR_AXLE_TO_CENTER*np.cos(theta), cur_y - 0.5*c.REAR_AXLE_TO_CENTER*np.sin(theta)):
                
                cur_x -= c.REAR_AXLE_TO_CENTER*np.cos(theta)
                cur_y -= c.REAR_AXLE_TO_CENTER*np.sin(theta)
                valid_checkpoints.append((cur_x, cur_y, theta, obstacle.id))

    return valid_checkpoints


def path_travel_time(path, L, forward_speed, reverse_speed, gear_change_time, steering_change_time):
    """Calculate calibrated traversal time for a Hybrid A* path in seconds."""
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


def find_shortest_time_hamiltonian(map, obstacles, start, theta_offset, astar_args,
                                   recognition_seconds=1.0, forward_speed=20.0,
                                   reverse_speed=15.0, gear_change_time=0.5,
                                   steering_change_time=0.15):
    """Find the exact minimum-time order and collision-aware route for each image.

    Each image uses its first valid scanning pose. Directed edge costs are obtained
    from Hybrid A* routes, then an exact Held-Karp dynamic program evaluates the
    physical-time edge costs without factorial permutation growth.
    """
    # ADDED: Route order is based on actual car motions and calibrated seconds,
    # not straight-line distance.
    from algo.pathfinding.hybrid_astar import HybridAStar

    checkpoints = {}
    unreachable = []
    for obstacle in obstacles:
        candidates = obstacle_to_checkpoint_all(map, obstacle, theta_offset)
        if candidates:
            checkpoints[obstacle] = candidates[0]
        else:
            unreachable.append(obstacle)

    reachable = list(checkpoints)
    if not reachable:
        return [], [], 0.0, unreachable

    def plan_leg(source, destination):
        planner = HybridAStar(
            map=map, x_0=source[0], y_0=source[1], theta_0=source[2],
            x_f=destination[0], y_f=destination[1], theta_f=destination[2],
            theta_offset=astar_args['theta_offset'],
            steeringChangeCost=astar_args['steeringChangeCost'],
            gearChangeCost=astar_args['gearChangeCost'], L=astar_args['L'],
            minR=astar_args['minR'], heuristic='euclidean', simulate=False,
            thetaBins=astar_args['thetaBins'], cost_mode='time',
            forward_speed=forward_speed, reverse_speed=reverse_speed,
            gear_change_time=gear_change_time, steering_change_time=steering_change_time)
        path, _ = planner.find_path()
        if path is None:
            return None, float('inf')
        return path, path_travel_time(path, astar_args['L'], forward_speed, reverse_speed,
                                      gear_change_time, steering_change_time)

    edges = {}
    start_key = 'start'
    for destination in reachable:
        edges[(start_key, destination)] = plan_leg(start, checkpoints[destination])
    for source in reachable:
        for destination in reachable:
            if source is not destination:
                edges[(source, destination)] = plan_leg(checkpoints[source], checkpoints[destination])

    # Exact Held-Karp Hamiltonian search scales as O(n^2 * 2^n), rather than n!.
    count = len(reachable)
    best = {}
    for index, destination in enumerate(reachable):
        path, travel_time = edges[(start_key, destination)]
        if path is not None:
            best[(1 << index, index)] = (travel_time, None)

    for mask in range(1, 1 << count):
        for last in range(count):
            state = best.get((mask, last))
            if state is None:
                continue
            for nxt in range(count):
                if mask & (1 << nxt):
                    continue
                _, travel_time = edges[(reachable[last], reachable[nxt])]
                if not np.isfinite(travel_time):
                    continue
                next_key = (mask | (1 << nxt), nxt)
                candidate = state[0] + travel_time
                if next_key not in best or candidate < best[next_key][0]:
                    best[next_key] = (candidate, last)

    full_mask = (1 << count) - 1
    finishes = [(best[(full_mask, last)][0], last) for last in range(count)
                if (full_mask, last) in best]
    best_order, best_paths, best_time = [], [], float('inf')
    if finishes:
        travel_time, last = min(finishes)
        indexes, mask = [], full_mask
        while last is not None:
            indexes.append(last)
            previous = best[(mask, last)][1]
            mask ^= 1 << last
            last = previous
        indexes.reverse()
        best_order = [reachable[index] for index in indexes]
        source = start_key
        for destination in best_order:
            best_paths.append(edges[(source, destination)][0])
            source = destination
        best_time = travel_time + recognition_seconds * count

    if not best_order:
        unreachable.extend(reachable)
    return best_order, best_paths, best_time, unreachable


def offset_x(facing: str):
    if facing == 'N':
        return 5.
    elif facing == 'S':
        return 5.
    elif facing == 'E':
        return 10.
    elif facing == 'W':
        return 0.


def offset_y(facing: str):
    if facing == 'N':
        return 10.
    elif facing == 'S':
        return 0.
    elif facing == 'E':
        return 5.
    elif facing == 'W':
        return 5.
    
def offset_theta(facing: str, theta_offset: float):
    return utils.M(utils.facing_to_rad(facing) + theta_offset)

def generate_random_obstacles(grid_size, obstacle_count):
    if grid_size < 100:
        offset = 5
    else:
        offset = 50
    obstacles = []
    directions = ['N', 'S', 'E', 'W']

    while len(obstacles) < obstacle_count:
        x = random.randint(offset, grid_size - offset)
        y = random.randint(offset, grid_size - offset)
        direction = random.choice(directions)
        obstacles.append(Obstacle(x, y, direction))

    return obstacles


def print_grid(grid_size, obstacles):
    path = []
    for y in range(grid_size - 1, -1, -1):
        for x in range(grid_size):
            position = (x, y)
            if (0 <= x <= 2) and (0 <= y <= 2):
                print("C", end=" ")  # Starting point
            elif any(obstacle.x_g == x and obstacle.y_g == y for obstacle in obstacles):
                direction = next(
                    (obstacle.facing for obstacle in obstacles if (obstacle.x_g, obstacle.y_g) == position), None)
                print(direction, end=" " if direction else ".")  # Obstacle facing direction
            elif (x, y) in path:
                print("*", end=" ")  # Mark the path with "*"
            else:
                print(".", end=" ")  # Empty space
        print()


if __name__ == "__main__":
    obstacles = [Obstacle(10, 10, 'N'), Obstacle(20, 10, 'S'), Obstacle(10, 20, 'E'), Obstacle(20, 20, 'W'), 
                 Obstacle(38, 38, 'N')]
    from objects.OccupancyMap import OccupancyMap
    map = OccupancyMap(obstacles) 
    tsp = Hamiltonian(obstacles, 5, 15, 0, -np.pi/2, 'euclidean')
    path = tsp.find_nearest_neighbor_path()
    print("\nShortest Path:")
    print(path)
