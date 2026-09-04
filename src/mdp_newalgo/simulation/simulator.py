import pygame
import numpy as np
from sys import exit
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__ + '\..')))

from algo.enumerations import Gear, Steering
from algo.objects.Border import Border, VirtualBorderWall
from algo.objects.Obstacle import Obstacle, VirtualWall
from algo.objects.OccupancyMap import OccupancyMap
from algo.pathfinding.hybrid_astar import HybridAStar
from algo.pathfinding.hamiltonian import Hamiltonian, obstacle_to_checkpoint, obstacle_to_checkpoint_all, \
    find_shortest_time_hamiltonian
from algo.simulation.testing import get_maps
from algo import utils
from typing import List
from algo import constants as c

# CHANGED: Imports are package-qualified so `python -m algo.simulation.simulator`
# works reliably when run from the directory containing the `algo` package.

class Simulator:
    def __init__(self, obstacles:List[Obstacle], hamiltonian_args, astar_args,
                 time_limit_seconds: float=120.0, recognition_seconds: float=1.0,
                 target_images: int=None, ordering_method: str='shortest_time',
                 forward_speed: float=20.0, reverse_speed: float=15.0,
                 gear_change_time: float=.5, steering_change_time: float=.15,
                 playback_speed: float=1.5):
        self.hamiltonian_args = hamiltonian_args
        self.astar_args = astar_args

        # CHANGED: Initialise Pygame before creating display/font resources.
        pygame.init()
        self.screen = pygame.display.set_mode((c.WIDTH, c.HEIGHT))
        self.screen.fill('white')
        pygame.display.set_caption('Algorithm Simulator')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.small_font = pygame.font.Font(None, 16)

        self.map_surface = pygame.Surface((c.MAP_WIDTH, c.MAP_HEIGHT))
        self.map_surface.fill('azure')
        # ADDED: Render a 40 x 40 grid so each arena cell is visible.
        self.draw_grid()
        self.start_surface = pygame.Surface((c.START_ZONE_SIZE_CM*c.MAP_WIDTH/c.ARENA_SIZE_CM,
                                             c.START_ZONE_SIZE_CM*c.MAP_HEIGHT/c.ARENA_SIZE_CM))
        self.start_surface.fill('aquamarine')

        self.path_surface = pygame.Surface((c.MAP_WIDTH, c.MAP_HEIGHT))
        self.path_surface = self.path_surface.convert_alpha()
        self.path_surface.fill((0,0,0,0))

        left_border = Border(c.BORDER_THICKNESS, c.MAP_HEIGHT + 2*c.BORDER_THICKNESS, 
                            c.MAP_X0 - c.BORDER_THICKNESS, c.MAP_Y0 - c.BORDER_THICKNESS)
        right_border = Border(c.BORDER_THICKNESS, c.MAP_HEIGHT + 2*c.BORDER_THICKNESS, 
                            c.MAP_X0 + c.MAP_WIDTH, c.MAP_Y0 - c.BORDER_THICKNESS)
        top_border = Border(c.MAP_WIDTH + 2*c.BORDER_THICKNESS, c.BORDER_THICKNESS, 
                            c.MAP_X0 - c.BORDER_THICKNESS, c.MAP_Y0 - c.BORDER_THICKNESS)
        bottom_border = Border(c.MAP_WIDTH + 2*c.BORDER_THICKNESS, c.BORDER_THICKNESS, 
                            c.MAP_X0 - c.BORDER_THICKNESS, c.MAP_Y0 + c.MAP_HEIGHT)


        self.borders = pygame.sprite.Group()
        self.borders.add(left_border)
        self.borders.add(right_border)
        self.borders.add(top_border)
        self.borders.add(bottom_border)

        self.obstacles = pygame.sprite.Group()
        self.virtual_walls = pygame.sprite.Group()
        self.virtual_wall_surface = pygame.Surface((c.MAP_WIDTH, c.MAP_HEIGHT), pygame.SRCALPHA)
        self.virtual_wall_surface.fill((0, 0, 0, 0))

        for obstacle in obstacles:
            self.obstacles.add(obstacle)
            vw = VirtualWall(obstacle.x_g, obstacle.y_g)
            self.virtual_walls.add(vw)
            self.virtual_wall_surface.blit(vw.image, (vw.rect.x - c.MAP_X0, vw.rect.y - c.MAP_Y0))
        
        left_border_wall = VirtualBorderWall(10*c.MAP_WIDTH/200, c.MAP_HEIGHT, 0, 0)
        right_border_wall = VirtualBorderWall(10*c.MAP_WIDTH/200, c.MAP_HEIGHT, 
                                               c.MAP_WIDTH - 10*c.MAP_WIDTH/200, 0)
        top_border_wall = VirtualBorderWall(c.MAP_WIDTH, 10*c.MAP_HEIGHT/200, 0, 0)
        bottom_border_wall = VirtualBorderWall(c.MAP_WIDTH, 10*c.MAP_HEIGHT/200, 
                                               0, c.MAP_HEIGHT - 10*c.MAP_HEIGHT/200)
        

        self.virtual_walls.add(left_border_wall)
        self.virtual_walls.add(right_border_wall)
        self.virtual_walls.add(top_border_wall)
        self.virtual_walls.add(bottom_border_wall)

        self.virtual_wall_surface.blit(left_border_wall.image, left_border_wall.rect.topleft)
        self.virtual_wall_surface.blit(right_border_wall.image, right_border_wall.rect.topleft)
        self.virtual_wall_surface.blit(top_border_wall.image, top_border_wall.rect.topleft)
        self.virtual_wall_surface.blit(bottom_border_wall.image, bottom_border_wall.rect.topleft)

        self.virtual_wall_surface.fill((255, 0, 0, 24),special_flags=pygame.BLEND_RGBA_MIN)

        # ADDED: State for frame-by-frame robot animation.
        self.animation_nodes = []
        self.animation_index = 0
        self.animation_elapsed = 0.0
        self.animation_step_seconds = 0.12
        self.robot_pose = None

        # ADDED: Explicit task state turns the visualizer into a timed image-
        # recognition simulator instead of treating route planning as completion.
        self.time_limit_seconds = time_limit_seconds
        self.recognition_seconds = recognition_seconds
        self.target_images = target_images if target_images is not None else len(obstacles)
        self.ordering_method = ordering_method
        self.elapsed_seconds = 0.0
        self.recognition_elapsed = 0.0
        self.recognition_obstacle = None
        self.recognised_obstacles = set()
        self.unreachable_obstacles = set()
        self.leg_end_indexes = {}
        self.node_durations = []
        self.planned_shortest_time = None
        self.task_state = 'PLANNING'
        self.task_finished = False
        self.obstacle_names = {id(obstacle): f'Image {index + 1}'
                               for index, obstacle in enumerate(obstacles)}
        # ADDED: Calibrated movement model used by both optimisation and animation.
        self.forward_speed = forward_speed
        self.reverse_speed = reverse_speed
        self.gear_change_time = gear_change_time
        self.steering_change_time = steering_change_time
        self.playback_speed = playback_speed

    def draw_grid(self):
        """Draw the 40 x 40 arena grid on the map background."""
        # CHANGED: Use subtle grid lines so obstacles, paths, and the robot remain readable.
        cell_width = c.MAP_WIDTH / c.GRID_SIZE
        cell_height = c.MAP_HEIGHT / c.GRID_SIZE
        grid_colour = (198, 215, 224)

        for i in range(c.GRID_SIZE + 1):
            pygame.draw.line(self.map_surface, grid_colour,
                             (round(i * cell_width), 0),
                             (round(i * cell_width), c.MAP_HEIGHT), 1)
            pygame.draw.line(self.map_surface, grid_colour,
                             (0, round(i * cell_height)),
                             (c.MAP_WIDTH, round(i * cell_height)), 1)

    def draw_axes(self):
        """Draw x/y grid-coordinate labels around the movement area."""
        # ADDED: Coordinates confirm the visible grid corresponds to the 40 x 40 map.
        cell_width = c.MAP_WIDTH / c.GRID_SIZE
        cell_height = c.MAP_HEIGHT / c.GRID_SIZE
        axis_colour = (35, 55, 65)

        for coordinate in range(0, c.GRID_SIZE + 1, 5):
            x = c.MAP_X0 + round(coordinate * cell_width)
            y = c.MAP_Y0 + c.MAP_HEIGHT - round(coordinate * cell_height)
            x_label = self.small_font.render(str(coordinate), True, axis_colour)
            y_label = self.small_font.render(str(coordinate), True, axis_colour)
            self.screen.blit(x_label, (x - x_label.get_width() // 2, c.MAP_Y0 + c.MAP_HEIGHT + 3))
            self.screen.blit(y_label, (c.MAP_X0 - y_label.get_width() - 6, y - y_label.get_height() // 2))

        x_axis = self.small_font.render('X coordinate (grid cells)', True, axis_colour)
        y_axis = self.small_font.render('Y coordinate (grid cells)', True, axis_colour)
        self.screen.blit(x_axis, (c.MAP_X0 + c.MAP_WIDTH // 2 - x_axis.get_width() // 2, 877))
        self.screen.blit(y_axis, (8, c.MAP_Y0 - 20))

    def draw_obstacle_state(self, obstacle):
        """Show whether an image is pending, being recognised, completed, or unreachable."""
        # ADDED: Recognition is a simulator event, not merely a planned endpoint.
        obstacle_x, obstacle_y = utils.grid_to_coords(obstacle.x_g, obstacle.y_g)
        centre_x, centre_y = utils.coords_to_pixelcoords(x=obstacle_x + 2.5,
                                                          y=obstacle_y + 2.5)
        obstacle_key = id(obstacle)

        if obstacle_key in self.recognised_obstacles:
            pygame.draw.circle(self.screen, (28, 150, 80), (centre_x, centre_y), 13)
            pygame.draw.line(self.screen, 'white', (centre_x - 6, centre_y),
                             (centre_x - 1, centre_y + 5), 3)
            pygame.draw.line(self.screen, 'white', (centre_x - 1, centre_y + 5),
                             (centre_x + 7, centre_y - 6), 3)
        elif obstacle is self.recognition_obstacle:
            pygame.draw.circle(self.screen, (245, 145, 25), (centre_x, centre_y), 13)
            pygame.draw.circle(self.screen, 'white', (centre_x, centre_y), 7, 2)
        elif obstacle_key in self.unreachable_obstacles:
            pygame.draw.circle(self.screen, (195, 45, 45), (centre_x, centre_y), 13)
            pygame.draw.line(self.screen, 'white', (centre_x - 5, centre_y - 5),
                             (centre_x + 5, centre_y + 5), 3)
            pygame.draw.line(self.screen, 'white', (centre_x - 5, centre_y + 5),
                             (centre_x + 5, centre_y - 5), 3)

    def draw_robot(self, x, y, theta, action=None):
        """Draw the robot at its rear-axle pose, rotated to its current heading."""
        # CHANGED: A pointed nose and front label make the robot heading explicit.
        centre_x = x + c.REAR_AXLE_TO_CENTER * np.cos(theta)
        centre_y = y + c.REAR_AXLE_TO_CENTER * np.sin(theta)
        pixel_x, pixel_y = utils.coords_to_pixelcoords(x=centre_x, y=centre_y)

        heading = np.array([np.cos(theta), -np.sin(theta)])
        sideways = np.array([-heading[1], heading[0]])
        centre = np.array([pixel_x, pixel_y])
        pixels_per_cm = c.MAP_WIDTH / c.ARENA_SIZE_CM
        half_length = c.ROBOT_LENGTH_CM * pixels_per_cm / 2
        half_width = c.ROBOT_WIDTH_CM * pixels_per_cm / 2
        body = [centre + heading * half_length,
                centre + heading * (half_length * .35) + sideways * half_width,
                centre - heading * half_length + sideways * half_width,
                centre - heading * half_length - sideways * half_width,
                centre + heading * (half_length * .35) - sideways * half_width]
        pygame.draw.polygon(self.screen, (35, 105, 180), body)
        pygame.draw.polygon(self.screen, 'black', body, 2)

        # Match the front camera and obstacle image face in the same cyan colour.
        front_centre = centre + heading * half_length
        bumper_a = front_centre - sideways * (half_width * .72)
        bumper_b = front_centre + sideways * (half_width * .72)
        pygame.draw.line(self.screen, 'black', bumper_a, bumper_b, 12)
        pygame.draw.line(self.screen, (0, 225, 255), bumper_a, bumper_b, 7)
        pygame.draw.circle(self.screen, 'black', front_centre, 8)
        pygame.draw.circle(self.screen, (245, 250, 255), front_centre, 4)

        # Long direction arrow remains readable while the robot is moving.
        arrow_end = centre + heading * (half_length + 24)
        pygame.draw.line(self.screen, (0, 45, 55), centre, arrow_end, 5)
        arrowhead = [arrow_end,
                     arrow_end - heading * 13 + sideways * 8,
                     arrow_end - heading * 13 - sideways * 8]
        pygame.draw.polygon(self.screen, (0, 225, 255), arrowhead)
        pygame.draw.polygon(self.screen, (0, 45, 55), arrowhead, 2)

    def draw_task_status(self):
        """Draw the timed-recognition result and current robot state."""
        # ADDED: Required assessment information: timer, recognised count, and outcome.
        remaining = max(0.0, self.time_limit_seconds - self.elapsed_seconds)
        timer_colour = (190, 45, 45) if remaining <= 15 else (25, 45, 55)
        recognised = len(self.recognised_obstacles)

        self.screen.blit(self.font.render('TASK STATUS', True, (25, 45, 55)), (895, 280))
        self.screen.blit(self.small_font.render(
            f'Time remaining: {remaining:05.1f} s', True, timer_colour), (895, 310))
        self.screen.blit(self.small_font.render(
            f'Recognised: {recognised} / {self.target_images}', True, (25, 45, 55)), (895, 335))
        self.screen.blit(self.small_font.render(
            f'Order: {"Shortest-time Hamiltonian" if self.ordering_method == "shortest_time" else ("Exact Hamiltonian" if self.ordering_method == "brute_force" else "Nearest neighbour")}',
            True, (25, 45, 55)), (895, 360))

        if self.planned_shortest_time is not None:
            self.screen.blit(self.small_font.render(
                f'Predicted route time: {self.planned_shortest_time:.1f} s', True, (25, 45, 55)),
                (895, 382))

        state_colours = {'PLANNING': (55, 85, 100), 'MOVING': (35, 95, 175),
                         'RECOGNISING': (205, 115, 10), 'COMPLETE': (25, 135, 70),
                         'TIME EXPIRED': (190, 45, 45), 'INCOMPLETE': (190, 45, 45)}
        self.screen.blit(self.font.render(self.task_state, True,
                         state_colours.get(self.task_state, (25, 45, 55))), (895, 410))

        if self.robot_pose is not None:
            x, y, theta, action = self.robot_pose
            if action is not None:
                gear = 'FORWARD' if action[0] == Gear.FORWARD else 'REVERSE'
                robot_label = self.small_font.render(
                    f'Robot: ({x / 5:.1f}, {y / 5:.1f}), {np.degrees(utils.M(theta)):.0f}°',
                    True, (25, 45, 55))
                action_label = self.small_font.render(f'Motion: {gear} / {action[1].name}', True, (25, 45, 55))
                self.screen.blit(robot_label, (895, 440))
                self.screen.blit(action_label, (895, 463))

        if self.unreachable_obstacles:
            self.screen.blit(self.small_font.render(
                f'Unreachable images: {len(self.unreachable_obstacles)}', True, (190, 45, 45)),
                (895, 490))

    def complete_recognition(self, obstacle):
        """Record recognition only after the robot finishes its dwell at an image."""
        # ADDED: Each obstacle can be counted once because object identity is stored in a set.
        self.recognised_obstacles.add(id(obstacle))
        self.recognition_obstacle = None
        self.recognition_elapsed = 0.0
        if len(self.recognised_obstacles) >= self.target_images:
            self.task_state = 'COMPLETE'
            self.task_finished = True
        else:
            self.task_state = 'MOVING'

    def update_task(self, delta_seconds):
        """Advance movement, recognition dwell, and the task countdown by one frame."""
        # ADDED: Enforce the time limit against simulated robot movement time.
        if self.task_finished or not self.animation_nodes:
            return

        # CHANGED: The visual playback can be accelerated while the countdown
        # still advances in calibrated simulated seconds.
        delta_seconds *= self.playback_speed
        self.elapsed_seconds += delta_seconds
        if self.elapsed_seconds >= self.time_limit_seconds:
            self.elapsed_seconds = self.time_limit_seconds
            self.task_state = 'TIME EXPIRED'
            self.task_finished = True
            return

        if self.recognition_obstacle is not None:
            self.task_state = 'RECOGNISING'
            self.recognition_elapsed += delta_seconds
            if self.recognition_elapsed >= self.recognition_seconds:
                self.complete_recognition(self.recognition_obstacle)
            return

        self.animation_elapsed += delta_seconds
        next_index = self.animation_index + 1
        next_duration = self.node_durations[next_index] if next_index < len(self.node_durations) else 0
        if self.animation_elapsed < next_duration:
            return

        self.animation_elapsed -= next_duration
        self.animation_index += 1
        if self.animation_index >= len(self.animation_nodes):
            self.task_state = 'INCOMPLETE'
            self.task_finished = True
            return

        node = self.animation_nodes[self.animation_index]
        self.robot_pose = (node.x, node.y, node.theta, node.prevAction)
        self.task_state = 'MOVING'
        reached_obstacle = self.leg_end_indexes.get(self.animation_index)
        if reached_obstacle is not None:
            self.recognition_obstacle = reached_obstacle
            self.recognition_elapsed = 0.0

    def draw_labels(self):
        """Draw a high-contrast legend and the start-zone label."""
        # CHANGED: Replace scattered coloured text with a single readable side panel.
        panel = pygame.Rect(875, 50, 290, 470)
        pygame.draw.rect(self.screen, (255, 255, 255), panel, border_radius=10)
        pygame.draw.rect(self.screen, (80, 105, 115), panel, 2, border_radius=10)

        title = self.font.render('ROBOT MOVEMENT AREA', True, (20, 45, 55))
        subtitle = self.small_font.render('2.0 m × 2.0 m  |  40 × 40 grid', True, (45, 75, 85))
        start_label = self.small_font.render('START ZONE', True, (10, 60, 55))
        image_label = self.small_font.render('Red obstacle edge  IMAGE FRONT', True, (25, 45, 55))
        safe_label = self.small_font.render('Red overlay  Safety / no-go area', True, (25, 45, 55))
        robot_label = self.small_font.render('Cyan camera + arrow  ROBOT FRONT', True, (25, 45, 55))

        self.screen.blit(title, (895, 72))
        self.screen.blit(subtitle, (895, 98))
        pygame.draw.line(self.screen, (205, 215, 220), (895, 126), (1145, 126), 1)
        pygame.draw.rect(self.screen, 'black', (898, 143, 36, 22))
        pygame.draw.rect(self.screen, (235, 35, 45), (898, 143, 36, 6))
        self.screen.blit(image_label, (945, 143))
        pygame.draw.rect(self.screen, (235, 105, 105), (897, 179, 34, 15))
        self.screen.blit(safe_label, (945, 174))
        pygame.draw.polygon(self.screen, (35, 105, 180), [(897, 217), (926, 207), (926, 227)])
        pygame.draw.line(self.screen, (0, 225, 255), (925, 208), (925, 226), 6)
        pygame.draw.polygon(self.screen, (0, 225, 255), [(939, 217), (926, 210), (926, 224)])
        self.screen.blit(robot_label, (945, 207))
        pygame.draw.line(self.screen, (205, 215, 220), (895, 255), (1145, 255), 1)

        self.screen.blit(start_label, (c.MAP_X0 + 8, c.MAP_Y0 + c.MAP_HEIGHT - 26))

    def start_simulation(self):
        map = OccupancyMap(self.obstacles)
        done = False

        tsp = Hamiltonian(obstacles=self.obstacles, map=map, x_start=self.hamiltonian_args['x_start'],
                          y_start=self.hamiltonian_args['y_start'], 
                          theta_start=self.hamiltonian_args['theta_start'],
                          theta_offset=self.hamiltonian_args['theta_offset'], 
                          metric=self.hamiltonian_args['metric'],
                          minR=self.hamiltonian_args['minR'])
        print("Starting simulator...")
        
        current_pos = tsp.start
        allPaths = []
        numNodes = 0
        
        while True:
            delta_seconds = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            
            # CHANGED: Clear every frame to prevent old robot text/shapes from lingering.
            self.screen.fill((242, 246, 247))
            self.screen.blit(self.map_surface, (c.MAP_X0, c.MAP_Y0))
            self.screen.blit(self.virtual_wall_surface, (c.MAP_X0, c.MAP_Y0))
            self.screen.blit(self.start_surface, (c.MAP_X0, c.MAP_Y0 + c.MAP_HEIGHT
                                                   - c.START_ZONE_SIZE_CM*c.MAP_HEIGHT/c.ARENA_SIZE_CM))
            self.borders.draw(self.screen)
            self.obstacles.draw(self.screen)
            self.draw_axes()
            for obstacle in self.obstacles:
                self.draw_obstacle_state(obstacle)
            self.draw_labels()

            colors = [(0, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (0, 255, 255, 255)]

            if not done:
                if self.ordering_method == 'shortest_time':
                    # ADDED: Exact permutation search using Hybrid A* edge times.
                    obstacle_path, allPaths, self.planned_shortest_time, unreachable = \
                        find_shortest_time_hamiltonian(
                            map, list(self.obstacles), tsp.start, self.hamiltonian_args['theta_offset'],
                            self.astar_args, recognition_seconds=self.recognition_seconds,
                            forward_speed=self.forward_speed, reverse_speed=self.reverse_speed,
                            gear_change_time=self.gear_change_time,
                            steering_change_time=self.steering_change_time)
                    self.unreachable_obstacles = {id(obstacle) for obstacle in unreachable}
                else:
                    # Exact-distance and greedy modes are retained for comparison.
                    obstacle_path = (tsp.find_brute_force_path() if self.ordering_method == 'brute_force'
                                     else tsp.find_nearest_neighbor_path())
                    self.unreachable_obstacles = {id(obstacle) for obstacle in tsp.unreachable_obstacles}
                    for obstacle in obstacle_path:
                        path = None
                        valid_checkpoints = obstacle_to_checkpoint_all(map, obstacle,
                                                                       theta_offset=self.hamiltonian_args['theta_offset'])
                        while path is None and valid_checkpoints:
                            checkpoint = valid_checkpoints.pop(0)
                            algo = HybridAStar(map, x_0=current_pos[0], y_0=current_pos[1], theta_0=current_pos[2],
                                               x_f=checkpoint[0], y_f=checkpoint[1], theta_f=checkpoint[2],
                                               theta_offset=self.astar_args['theta_offset'],
                                               steeringChangeCost=self.astar_args['steeringChangeCost'],
                                               gearChangeCost=self.astar_args['gearChangeCost'], L=self.astar_args['L'],
                                               minR=self.astar_args['minR'], heuristic=self.astar_args['heuristic'],
                                               simulate=False, thetaBins=self.astar_args['thetaBins'])
                            path, _ = algo.find_path()
                        if path is not None:
                            allPaths.append(path)
                            current_pos = (path[-1].x, path[-1].y, path[-1].theta)
                        else:
                            self.unreachable_obstacles.add(id(obstacle))
                
                idx = 0
                for path in allPaths:    
                    print("Drawing path on simulator...")
                    numNodes += len(path)
                    current_pos = (path[-1].x, path[-1].y, path[-1].theta)
                    x, y = utils.coords_to_pixelcoords(x=path[-1].x + c.REAR_AXLE_TO_CENTER*np.cos(path[-1].theta), 
                                                        y=path[-1].y + c.REAR_AXLE_TO_CENTER*np.sin(path[-1].theta))
                    pygame.draw.lines(self.path_surface, 'black', True, [(x-10,y-10),(x+10,y+10)], 3)
                    pygame.draw.lines(self.path_surface, 'black', True, [(x-10,y+10),(x+10,y-10)], 3)
            
                    color = colors[idx % len(colors)]
                    self.draw_final_path(path, color)
                    idx += 1
                    
                # ADDED: Flatten all planned legs into animation frames. Each
                # Node preserves both the robot pose and its forward/reverse/turn action.
                self.animation_nodes = []
                self.leg_end_indexes = {}
                self.node_durations = []
                for path, obstacle in zip(allPaths, [obstacle for obstacle in obstacle_path
                                                      if id(obstacle) not in self.unreachable_obstacles]):
                    previous_action = (Gear.FORWARD, Steering.STRAIGHT)
                    self.animation_nodes.extend(path)
                    for node in path:
                        action = node.prevAction
                        speed = self.forward_speed if action[0] == Gear.FORWARD else self.reverse_speed
                        duration = self.astar_args['L'] / speed
                        if action[0] != previous_action[0]:
                            duration += self.gear_change_time
                        if action[1] != previous_action[1]:
                            duration += self.steering_change_time
                        self.node_durations.append(duration)
                        previous_action = action
                    self.leg_end_indexes[len(self.animation_nodes) - 1] = obstacle
                if self.animation_nodes:
                    self.robot_pose = (tsp.start[0], tsp.start[1], tsp.start[2], None)
                    self.animation_index = -1
                    self.task_state = 'MOVING'
                else:
                    self.task_state = 'INCOMPLETE'
                    self.task_finished = True
                done = True
                print(f"Total path length: {numNodes*self.astar_args['L']}cm")

            self.screen.blit(self.path_surface, (0, 0))

            self.update_task(delta_seconds)

            if self.robot_pose is not None:
                self.draw_robot(*self.robot_pose)
            self.draw_task_status()

            pygame.display.update()
        
    def draw_final_path(self, path, color):
        points = []
        x, y = utils.coords_to_pixelcoords(x=path[0].parent.x, y=path[0].parent.y)
        points.append((x, y))
        for node in path:
            x, y = utils.coords_to_pixelcoords(x=node.x, y=node.y)
            points.append((x, y))
            if node.prevAction[0] == Gear.FORWARD:
                pygame.draw.circle(self.path_surface, (0, 255, 0, 255), (x, y), 4)
            else:
                pygame.draw.circle(self.path_surface, (255, 0, 0, 255), (x, y), 4)

        pygame.draw.lines(self.path_surface, color, False, points, width=3)
    
    def draw_path_history(self, pathHistory):

        for node in pathHistory[1:]:
            x0, y0 = utils.coords_to_pixelcoords(x=node.parent.x, y=node.parent.y)
            x1, y1 = utils.coords_to_pixelcoords(x=node.x, y=node.y)
            pygame.draw.line(self.path_surface, (255, 0, 255, 64), (x0, y0), (x1, y1))

if __name__ == "__main__":
    # Five reachable, axis-aligned 10 cm obstacles for the assessment demo.
    map = [Obstacle(8, 12, 'E', 1), Obstacle(20, 8, 'N', 2),
           Obstacle(30, 12, 'W', 3), Obstacle(10, 28, 'S', 4),
           Obstacle(26, 30, 'W', 5)]
    
    hamiltonian_args = {'obstacles': map, 'x_start': 15, 'y_start': 10, 'theta_start': np.pi/2, 
                        'theta_offset': 0, 'metric': 'euclidean', 'minR': 26.5}
    astar_args = {'steeringChangeCost': 10, 'gearChangeCost': 10, 'L': 26.5*np.pi/4/5, 'theta_offset': 0,
                    'minR': 25, 'heuristic': 'euclidean', 'simulate': False, 'thetaBins': 24}

    sim = Simulator(map, hamiltonian_args, astar_args, time_limit_seconds=120,
                    recognition_seconds=1, target_images=5, ordering_method='shortest_time')
    sim.start_simulation()
