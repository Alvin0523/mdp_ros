# mdp_ros

ROS2 (Jazzy) workspace for the MDP Ackermann-steered robot — simulation (Gazebo), real
hardware (Raspberry Pi 4B + STM32), autonomy, path planning, and vision.

Managed with [pixi](https://pixi.sh) — `pixi install` sets up the full ROS2 + build
environment (no system-wide ROS install needed). See `pixi.toml` for tasks (`pixi run sim`,
`pixi run real`, `pixi run build`, etc.) and dependencies.

## Packages (`src/`)

| Package | Description |
|---|---|
| `mdp_description` | URDF and meshes for the robot |
| `mdp_bringup` | Top-level launch files (sim, real hardware) + Task 1/2 runner scripts |
| `mdp_algorithm/mdp_planning` | Path planning building blocks (Reeds-Shepp/Dubins + TSP, pure-pursuit, spline) |
| `mdp_algorithm/wayp_plan_tools` | Waypoint loading/saving and pursuit control tools |
| `mdp_bridge` | STM32 serial bridge (`serial_bridge_node`) + Android Bluetooth bridge (`bluetooth_bridge_node`, placeholder) |
| `mdp_vision/mdp_yolo` | Camera capture (sim fallback) + YOLO detection node |

## External dependencies (not submodules)

`src/mdp_vision/` also holds two upstream projects, cloned directly rather than vendored as git
submodules — they're built from source on the Raspberry Pi only (see
[`docs/pi-camera-vision.md`](docs/pi-camera-vision.md) for the full why/how, including
Pi-specific build patches):

```bash
cd src/mdp_vision
git clone https://git.libcamera.org/libcamera/libcamera.git
git clone https://github.com/christianrauch/camera_ros.git
```

- [`libcamera`](https://libcamera.org/) — Linux camera stack, talks to the Pi's CSI camera sensor.
- [`camera_ros`](https://github.com/christianrauch/camera_ros) — ROS2 node wrapping libcamera,
  publishes `/camera/image_raw`, `/camera/image_raw/compressed`, `/camera/camera_info`.

Neither is needed on a non-Pi machine (sim uses Gazebo's own camera sensor instead) — the build
deps for them are scoped to `linux-aarch64` only in `pixi.toml`.

## Docs

- [`docs/pi-camera-vision.md`](docs/pi-camera-vision.md) — vision pipeline architecture (sim vs
  real), the Pi camera build, Foxglove streaming notes, and known open issues.
