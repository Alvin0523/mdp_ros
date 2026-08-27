# Vision pipeline: sim + real Pi camera

## Pipeline overview

```
Sim (Gazebo):
  camera_link sensor plugin --(ros_gz_bridge)--> /camera/image_raw, /camera/camera_info
                                                        |
Real (Pi 4B + Camera Module v2 / imx219):              v
  camera_ros node --------------------------------> yolo_detector.py (mdp_vision)
    /camera/image_raw (raw)                              | cv_bridge -> ultralytics YOLO
    /camera/image_raw/compressed (JPEG)                  v
    /camera/camera_info                              /yolo_result (std_msgs/String)
```

- Sim: `mini_akm_robot.urdf` has a `<sensor type="camera">` on `camera_link`; `task2_sim.launch.py`
  bridges `/camera/image_raw` + `/camera/camera_info` from Gazebo.
- Real: `real.launch.py` launches `camera_ros`'s `camera_node` (package `camera_ros`, built from
  source, see below) and `yolo_detector.py`, pointed at `/camera/image_raw`.
- `yolo_detector.py`'s `model_path` param defaults to
  `get_package_share_directory('mdp_vision')/models/yolo26n_ncnn_model` (see "Known open issues"
  #1 below for why NCNN rather than a raw `.pt` checkpoint). Both `*.pt` files and
  `models/*_ncnn_model/` directories are gitignored, so the model must be exported/copied to any
  new machine manually — `pixi run colcon build --packages-select mdp_vision` then installs
  whatever's in `models/` into the share dir.

## Why camera_ros is built from source, not `pixi install`ed

`ros-jazzy-camera-ros` / `ros-jazzy-libcamera` exist as robostack-jazzy conda packages, but
installing them fails to *solve* on `linux-aarch64` (unrelated to the Pi itself) — the
`camera_ros` metapackage's dependency chain (`image_view` → `camera_calibration_parsers` →
`python_abi`/`ros2-distro-mutex`) doesn't cohere on that platform in the channel as of this
writing. So instead: clone `libcamera` + `camera_ros` source into this workspace's `src/` and
build them with `pixi run colcon build` like any other package. Build-tool deps for this
(`meson`, `pkg-config`, `colcon-meson`, `yaml`, `pyyaml`, `ply`, `jinja2`, `gnutls`, `openssl`,
`libtiff`, `libudev`, plus camera_ros's own `ros-jazzy-camera-info-manager` +
`ros-jazzy-image-transport`) are in `pixi.toml` under `[target.linux-aarch64.dependencies]` —
Linux-aarch64-only since this is Pi-specific; the laptop/sim side never needs it.

**Setup on a new Pi:**
```bash
cd ~/mdp/mdp_ros/src
git clone https://git.libcamera.org/libcamera/libcamera.git   # upstream, NOT the raspberrypi fork
git clone https://github.com/christianrauch/camera_ros.git
cd ~/mdp/mdp_ros
pixi install
pixi run colcon build --packages-select libcamera camera_ros --event-handlers console_direct+ \
  --meson-args -Dpipelines=rpi/vc4 -Dtracing=disabled -Dqcam=disabled -Dpycamera=disabled
```
(`src/libcamera` and `src/camera_ros` are currently untracked/uncommitted in this repo — they're
external upstream source, not this project's code. Consider adding them to `.gitignore`.)

Camera Module v2 = Sony IMX219, which **upstream** libcamera supports directly — no need for the
`raspberrypi/libcamera` fork (that's only required for Camera Module 3 / IMX708). Confirmed via
`cam -l`:
```
1: External camera 'imx219' (/base/soc/i2c0mux/i2c@1/imx219@10)
```

### Meson flags, and why

- `-Dpipelines=rpi/vc4`: excludes the Pi-5-only `rpi/pisp` pipeline handler, which needs
  `libpisp`, which needs `linux/dma-heap.h` — a kernel UAPI header conda's bundled sysroot doesn't
  have. Pi 4B only needs `rpi/vc4` anyway.
- `-Dtracing=disabled -Dqcam=disabled -Dpycamera=disabled`: optional features unrelated to camera
  capture that each hit their own unrelated build breaks (LTTng header triggering
  `-Werror=missing-field-initializers` under GCC 14, mainly) — disabled rather than chased.

### Local (untracked) sysroot patches — **will not survive a `.pixi` env rebuild**

Two structs/macros are missing from conda's bundled kernel UAPI headers (older than what the
Pi's actual running kernel and libcamera's code expect) and had to be patched directly into the
pixi env on the Pi:

1. `~/mdp/mdp_ros/.pixi/envs/default/aarch64-conda-linux-gnu/sysroot/usr/include/linux/sched.h`
   — appended `struct clone_args` + `CLONE_ARGS_SIZE_VER0/1/2` (added in Linux 5.3 for `clone3()`,
   needed by `libcamera`'s `process.cpp`).
2. `~/mdp/mdp_ros/.pixi/envs/default/aarch64-conda-linux-gnu/sysroot/usr/include/asm-generic/unistd.h`
   — appended `#define __NR_clone3 435`.

Both patches are guarded (`#ifndef ...`) so they're idempotent and harmless if a future
conda-forge release fixes this upstream. If the Pi's `.pixi/envs/default` is ever wiped and
reinstalled from scratch, these need reapplying before `libcamera` will build — the exact `cat >>`
commands are in this session's history; ask to have them turned into a proper setup script if this
becomes a recurring pain.

## Streaming to Foxglove

- `pixi run foxglove` runs `foxglove_bridge`; connect Foxglove Studio to
  `ws://<pi-tailscale-ip>:8765`.
- `camera_ros` **hardcodes** exactly two image publishers in `CameraNode.cpp` — `image_raw`
  (raw) and `image_raw/compressed` (JPEG). It does **not** use `image_transport`'s automatic
  "advertise every discovered plugin" behavior, so installing more `image_transport` plugins
  (e.g. `ros-jazzy-foxglove-compressed-video-transport` for H.264) does **nothing** on its own —
  confirmed by reading its source. Getting H.264 out of it would require a separate
  `image_transport republish raw foxglove` bridge node subscribed to `/camera/image_raw`, which is
  more moving parts (extra CPU for encode/decode, extra latency) and was not integrated into any
  launch file. Not pursued past a manual one-off test.
- For actually-fast viewing: use `/camera/image_raw/compressed` (JPEG) in the Foxglove Image
  panel, not raw. `camera_node` is pinned to `640x480` in `real.launch.py` (was auto-selecting
  `800x600`) to cut data volume further.
- If it's still slow: check whether Tailscale is going direct or relaying — `tailscale ping
  <pi>` on the client. A relayed (DERP) connection adds real latency that no amount of
  compression fixes. Same-LAN / wired Ethernet between the Pi and the viewing machine is the
  single biggest lever if available.

## Known open issues

1. **YOLO inference crashes (SIGILL) on the Pi — fixed via two independent changes, needs retest.**

   `yolo_detector.py` loaded the model fine, then died with exit code `-4` on the first actual
   inference. Two separate wrong-architecture problems were stacked here, found and fixed in
   this order:

   **a) Wrong BLAS variant.** `numpy`/`torch` resolved conda-forge's `nvpl` (NVIDIA Performance
   Libraries) BLAS backend — `libcblas.so.3: undefined symbol: nvpl_blas_core_scabs1`. NVPL
   targets NVIDIA Grace/Graviton (SVE-capable) *server* ARM chips, not a Pi 4B's Cortex-A72.

   *Why conda picked the wrong one, and why this needed a manual fix*: this workspace uses
   `pixi`/conda-forge for the whole ROS2 stack (not plain pip), and conda-forge packages BLAS as
   a separate, swappable component — `numpy`, `torch`, `opencv` all dynamically link against
   whichever BLAS variant is present (`openblas`, `mkl`, `nvpl`, ...). The solver just picks one
   to satisfy the whole environment from its own internal ranking; it doesn't inspect the actual
   CPU the environment will run on, and `linux-aarch64` as a platform tag covers everything from
   a Pi 4B to an NVIDIA Grace server. There's also no runtime fallback once the wrong one is
   linked — a compiled library either has the CPU instruction or it hard-crashes hitting it,
   there's no "try it, catch, fall back" happening automatically. (Plain `pip install numpy`
   would likely have avoided this entirely — PyPI wheels bundle a widely-compatible OpenBLAS by
   default — but numpy comes from conda here since the rest of the ROS2 stack needs it from
   there too.) Fixed by pinning the conda BLAS build variant explicitly in `pixi.toml`:
   ```toml
   [dependencies]
   libblas = { version = "*", build = "*openblas" }
   liblapack = { version = "*", build = "*openblas" }
   ```
   Confirmed `nvpl` no longer appears anywhere in `pixi.lock` for either platform.

   **b) Still crashed after (a) — switched to NCNN.** Even with the BLAS fix applied and
   confirmed on the Pi (`pixi list | grep blas` showed `openblas` correctly), YOLO still crashed
   identically. `pixi list` also showed `nvidia_cublas` (~660MB, a CUDA runtime library) present
   despite there being no NVIDIA GPU anywhere in this system — `torch`, pulled in via `pip`
   through `ultralytics`, had resolved PyPI's default CUDA-enabled build rather than a CPU-only
   one. Rather than keep chasing exact torch/CUDA wheel selection, switched to running the model
   via **NCNN** instead of raw `.pt`/torch inference, per Ultralytics' own
   [Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/) (NCNN is their
   recommended format for ARM, and benchmarks ~4x faster than raw PyTorch on a Pi 5 in their own
   numbers). `yolo26n.pt` exported via `model.export(format='ncnn')`, model now lives at
   `models/yolo26n_ncnn_model/` (the `_ncnn_model` directory suffix is required by ultralytics'
   own format-detection, not a naming choice). `ncnn`/`pnnx` added to `pixi.toml`'s
   `[pypi-dependencies]` (resolved proper per-platform wheels, confirmed for `linux-aarch64`).

   NCNN's own inference engine is self-contained and doesn't use system BLAS at all - but the
   BLAS pin from (a) is still needed regardless, since `ultralytics` still imports `torch` and
   uses `numpy` for everything *around* the model (image preprocessing, NMS, result
   postprocessing) no matter which format runs the model itself.

   *Open question, not yet decided*: `pnnx` is only used at export time
   (`model.export(format='ncnn')`, already run once on a dev machine), not by `yolo_detector.py`
   at runtime - only `ncnn` is needed to load and run the already-exported model. Whether `pnnx`
   is needed on the Pi depends on whether the Pi is expected to run its own export locally, or
   just receive the already-exported `models/yolo26n_ncnn_model/` directory (gitignored, so it
   isn't pulled in by `git pull` either way - needs an explicit copy, e.g. `scp`, same as the
   old `.pt` file did per the note above).

   **Verified so far**: end-to-end on the dev machine only (node loads the NCNN model and runs
   the exact `image_callback()` code path without crashing). **Not yet re-tested on the physical
   Pi** — pull, `pixi install`, `pixi run real` or `pixi run vision`, confirm `yolo_detector.py`
   survives its first inference instead of dying with exit code `-4`.
2. **`mini_akm_real_robot.urdf` has no `camera_link` / TF frame for the camera.** It's a
   control-only URDF (5 links: `base_link` + 4 actuated wheel/steering joints) — no visuals, no
   `camera_link`, no `laser_link`, no `base_footprint`. YOLO detection itself doesn't care (no TF
   lookups), but anything spatial later (projecting a detection into the robot frame) will need a
   static transform for `camera_link` added back, either in the URDF or via
   `static_transform_publisher` in `real.launch.py`.
3. **`pixi.toml`'s `ros-jazzy-foxglove-compressed-video-transport` line was added then dropped**
   (committed, not just locally reverted) since H.264 wasn't pursued and `camera_ros` doesn't use
   `image_transport`'s plugin auto-discovery anyway (see above) — confirmed not present in the
   current `pixi.toml`.
4. Cosmetic-only, not a bug: `Could not enable FIFO RT scheduling policy: ... Operation not
   permitted` on every `ros2_control_node` startup — needs `CAP_SYS_NICE` (or root) on the Pi to
   go away; controllers work fine without it, just without RT scheduling guarantees.

## Camera parameters pinned in `real.launch.py`

`camera_node`'s `Node(...)` params were previously bare (`width`/`height` only), which left two
warnings on every launch - `camera_node` logs the exact param name each time, which is how these
were found:

- `camera: 0` and `format: 'RGB888'` - silences "no camera selected"/"no pixel format selected,
  auto-selecting XRGB8888". `RGB888` (3-channel, no alpha) was chosen over the auto-picked
  `XRGB8888` since it matches `yolo_detector.py`'s `cv_bridge.imgmsg_to_cv2(...,
  desired_encoding='bgr8')` conversion more directly.
- `camera_info_url: 'package://mdp_vision/config/imx219_640x480.yaml'` - a generic IMX219
  calibration (community-published, `UbiquityRobotics/raspicam_node`'s `camerav2_1280x960.yaml`
  scaled 0.5x for this resolution), **not measured on this robot's specific camera unit**. Silences
  "calibration file not found" and gives `CameraInfo` real (not identity) values; recalibrate on
  this exact unit with `ros2 run camera_calibration cameracalibrator` if a task ever needs precise
  metric distance/offset estimation from the image - YOLO classification doesn't.

## Real-hardware quirks (non-camera, found along the way)

- STM32 bridge enumerates as `/dev/ttyACM0` on this Pi, not the `/dev/ttyUSB0` launch-file
  default — `pixi run real` hardcodes the override.
- If `camera_node` fails with `failed to acquire camera` / `Pipeline handler in use by another
  process`, something else still has `/dev/media0` open — `ps aux | grep camera_node` and kill
  the stale one before retrying.
