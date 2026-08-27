# mdp_ros

ROS2 (Jazzy) workspace for the MDP Ackermann-steered robot — simulation (Gazebo), real
hardware (Raspberry Pi 4B + STM32), autonomy, path planning, and vision.

Managed with [pixi](https://pixi.sh) — `pixi install` sets up the full ROS2 + build
environment (no system-wide ROS install needed). See `pixi.toml` for tasks (`pixi run sim`,
`pixi run hardware`, `pixi run build`, etc.) and dependencies.

## Packages (`src/`)

| Package | Description |
|---|---|
| `mdp_description` | URDF and meshes for the robot |
| `mdp_bringup` | Top-level launch files (sim, real hardware) |
| `mdp_control` | Autonomy, path planning, and task-runner nodes |
| `mdp_hardware_bridge` | Serial bridge between the STM32 firmware's custom protocol and ROS2 topics |
| `wayp_plan_tools` | Waypoint loading/saving and pursuit control tools |
| `src/vision/mdp_vision` | Camera capture (sim fallback) + YOLO detection node |

## External dependencies (not submodules)

`src/vision/` also holds two upstream projects, cloned directly rather than vendored as git
submodules — they're built from source on the Raspberry Pi only (see
[`docs/pi-camera-vision.md`](docs/pi-camera-vision.md) for the full why/how, including
Pi-specific build patches):

```bash
cd src/vision
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
