# Sourced by pixi on activation (linux-64 only, see pixi.toml).
# On a laptop with an NVIDIA GPU next to an integrated one (PRIME), Gazebo's
# rendering (GUI + camera sensor) otherwise lands on the integrated GPU - and
# Mesa prints "libEGL warning: ... failed to create dri2 screen" while probing
# the NVIDIA card it has no driver for. These route OpenGL/EGL to the NVIDIA
# driver instead. Only set when the NVIDIA driver is actually loaded, so
# Intel/AMD-only machines are untouched. Opt out: MDP_GPU=off pixi run ...
if [ "${MDP_GPU:-auto}" != "off" ] \
   && [ -z "${__NV_PRIME_RENDER_OFFLOAD:-}" ] \
   && [ -r /proc/driver/nvidia/version ] \
   && [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
    export __NV_PRIME_RENDER_OFFLOAD=1
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi
