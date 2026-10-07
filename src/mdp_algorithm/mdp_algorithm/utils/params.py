#!/usr/bin/env python3
"""
Task 1 car, costmap, planner and path-follower settings - one object, SI
units (metres, radians, seconds), Nav2 parameter names where Nav2 has one.

Each number lives in ONE file: the car's wheelbase and steering limits in the
URDF (mdp_description/urdf/mdp_robot.urdf.xacro, see car_from_urdf), all the
rest in mdp_bringup/config/navigation.yaml. task1_runner gets both as ROS
parameters (mdp.launch.py) and calls configure(); without ROS (tests, offline
tools) load() reads the same two files. The planner and follower read ACTIVE when they build
a costmap, start a search or are constructed - never at import time. Until
configure() is called, ACTIVE is load().

The planner computes in centimetres internally (planning/costmap.py); the
*_cm helpers here are the only place that converts.
"""

import os
from dataclasses import dataclass, fields

import xml.etree.ElementTree as ET

import yaml

# YAML section -> the fields in it. The ROS parameter name is "<section>.<field>".
SECTIONS = {
    'robot': ('wheelbase', 'footprint_rear', 'footprint_front', 'footprint_half_width',
              'steering_limit_left', 'steering_limit_right',
              'minimum_turning_radius_left', 'minimum_turning_radius_right'),
    'costmap': ('resolution', 'footprint_padding', 'inflation_radius', 'cost_scaling_factor'),
    'planner': ('turning_radius_margin', 'step_size', 'xy_search_resolution',
                'angle_quantization_bins', 'cost_penalty', 'reverse_penalty', 'change_penalty',
                'steering_change_penalty', 'analytic_expansion_max_cost',
                'analytic_expansion_max_length', 'analytic_expansion_interval',
                'max_iterations', 'max_planning_time', 'goal_xy_tolerance', 'goal_yaw_tolerance',
                'checkpoint_standoff'),
    'follower': ('lookahead_dist', 'use_velocity_scaled_lookahead_dist', 'desired_linear_vel',
                 'xy_goal_tolerance', 'cusp_tolerance', 'max_path_error',
                 'use_regulated_linear_velocity_scaling', 'max_lateral_accel',
                 'regulated_linear_scaling_min_speed', 'max_reverse_linear_vel',
                 'approach_velocity_scaling_dist', 'min_approach_linear_velocity', 'creep_linear_vel',
                 'path_tracking', 'feedback_k_e', 'feedback_k_theta', 'feedforward_preview_time',
                 'lqr_q_lateral', 'lqr_q_heading', 'lqr_q_steer', 'lqr_r', 'steering_time_constant',
                 'pose_latency'),
}


@dataclass(frozen=True)
class PlannerParams:
    # --- robot -------------------------------------------------------------
    wheelbase: float
    footprint_rear: float
    footprint_front: float
    footprint_half_width: float
    steering_limit_left: float
    steering_limit_right: float
    minimum_turning_radius_left: float
    minimum_turning_radius_right: float
    # --- costmap -----------------------------------------------------------
    resolution: float
    footprint_padding: float
    inflation_radius: float
    cost_scaling_factor: float
    # --- planner (Hybrid A*) ----------------------------------------------
    turning_radius_margin: float
    step_size: float
    xy_search_resolution: float
    angle_quantization_bins: int
    cost_penalty: float
    reverse_penalty: float
    change_penalty: float
    steering_change_penalty: float
    analytic_expansion_max_cost: int
    analytic_expansion_max_length: float
    analytic_expansion_interval: int
    max_iterations: int
    max_planning_time: float
    goal_xy_tolerance: float
    goal_yaw_tolerance: float
    checkpoint_standoff: float
    # --- follower (pure pursuit) -------------------------------------------
    lookahead_dist: float
    use_velocity_scaled_lookahead_dist: bool
    desired_linear_vel: float
    xy_goal_tolerance: float
    cusp_tolerance: float
    max_path_error: float
    use_regulated_linear_velocity_scaling: bool
    max_lateral_accel: float
    regulated_linear_scaling_min_speed: float
    max_reverse_linear_vel: float
    approach_velocity_scaling_dist: float
    min_approach_linear_velocity: float
    creep_linear_vel: float
    path_tracking: str
    feedback_k_e: float
    feedback_k_theta: float
    feedforward_preview_time: float
    lqr_q_lateral: float
    lqr_q_heading: float
    lqr_q_steer: float
    lqr_r: float
    steering_time_constant: float
    pose_latency: float

    # --- derived, centimetres (the planner's internal unit) -----------------
    @property
    def plan_turn_radius_left_cm(self) -> float:
        return self.minimum_turning_radius_left * self.turning_radius_margin * 100.0

    @property
    def plan_turn_radius_right_cm(self) -> float:
        return self.minimum_turning_radius_right * self.turning_radius_margin * 100.0

    @property
    def symmetric_turn_radius_cm(self) -> float:
        """One radius drivable both ways (the wider side), for code that has
        only one: the visiting-order distances (visiting_order.py)."""
        return max(self.minimum_turning_radius_left, self.minimum_turning_radius_right) * 100.0

    @property
    def checkpoint_standoff_cm(self) -> float:
        return self.checkpoint_standoff * 100.0


def _share(package: str, *path) -> str:
    """A file of `package`: installed (a sourced workspace), else the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory(package), *path)
    except Exception:   # not sourced / ament_index missing: running from the source tree
        src = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..')
        return os.path.normpath(os.path.join(src, package, *path))


# URDF xacro property -> PlannerParams / ROS parameter name (robot.<name>).
URDF_CAR = {'wheelbase': 'wheelbase', 'rear_track': 'rear_track', 'front_track': 'front_track',
            'wheel_radius': 'wheel_radius', 'steer_left': 'steering_limit_left',
            'steer_right': 'steering_limit_right'}


def car_from_urdf(xacro_file: str = None) -> dict:
    """The car's measured numbers from the URDF's xacro properties (plain
    numbers there): {"wheelbase": 0.143, "steering_limit_left": ..., ...}."""
    root = ET.parse(xacro_file or _share('mdp_description', 'urdf', 'mdp_robot.urdf.xacro')).getroot()
    props = {e.get('name'): e.get('value') for e in root.iter('{http://www.ros.org/wiki/xacro}property')}
    return {name: float(props[prop]) for prop, name in URDF_CAR.items()}


def planner_flat(navigation_yaml: str = None) -> dict:
    """navigation.yaml (task1_runner's ROS parameter file) as {"<section>.<field>": value}."""
    with open(navigation_yaml or _share('mdp_bringup', 'config', 'navigation.yaml')) as f:
        sections = yaml.safe_load(f)['task1_runner']['ros__parameters']
    return {f'{s}.{k}': v for s, values in sections.items() for k, v in values.items()}


def from_flat(values: dict) -> PlannerParams:
    """PlannerParams from {"<section>.<field>": value} (ROS parameter names).
    Every setting must be there; names that are not settings are ignored."""
    types = {f.name: f.type for f in fields(PlannerParams)}
    kw, missing = {}, []
    for section, names in SECTIONS.items():
        for name in names:
            key = f'{section}.{name}'
            if key in values:
                kw[name] = types[name](values[key])
            else:
                missing.append(key)
    if missing:
        raise KeyError(f'missing settings: {", ".join(missing)}')
    return PlannerParams(**kw)


def load(xacro_file: str = None, navigation_yaml: str = None) -> PlannerParams:
    """The settings straight from the URDF + navigation.yaml."""
    car = {f'robot.{k}': v for k, v in car_from_urdf(xacro_file).items()}
    return from_flat({**car, **planner_flat(navigation_yaml)})


def to_flat(params: PlannerParams) -> dict:
    """{"<section>.<field>": value} for every setting - what a node declares."""
    return {f'{s}.{n}': getattr(params, n) for s, names in SECTIONS.items() for n in names}


def configure(params: PlannerParams) -> None:
    """Make `params` the settings every later plan / costmap / follower uses."""
    global ACTIVE
    ACTIVE = params


def __getattr__(name):
    # ACTIVE before any configure(): the YAML files (loaded on first use).
    if name == 'ACTIVE':
        configure(load())
        return ACTIVE
    raise AttributeError(name)
