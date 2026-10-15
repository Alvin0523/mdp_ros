#!/usr/bin/env bash
# This machine's Zenoh router (rmw_zenohd) - every ROS task depends on it
# (pixi.toml `router`). Started once, in the background, and left running like
# the ros2 daemon, so the next `pixi run` finds it at once. rmw_zenoh nodes only
# find each other through a router, and a node that comes up before it retries
# its connection 1-4 s later - hence waiting here until it listens.
# The laptop's router also connects to the Pi's ($MDP_ZENOH_PI, over Tailscale):
# one TCP link carries everything between the machines.
# Stop it: pkill -x rmw_zenohd      Log: ~/.ros/log/zenohd.log
set -u
[ "${RMW_IMPLEMENTATION:-}" = "rmw_zenoh_cpp" ] || exit 0   # MDP_RMW=fastdds: no router

listening() { (exec 3<>/dev/tcp/127.0.0.1/7447) 2>/dev/null; }
listening && exit 0

if [ "$(uname -m)" != "aarch64" ]; then   # not the Pi (the only linux-aarch64 machine)
    export ZENOH_CONFIG_OVERRIDE="connect/endpoints=[\"$MDP_ZENOH_PI\"]"
fi
mkdir -p "$HOME/.ros/log"
# Own session, so Ctrl+C on the launch does not stop it (macOS has no setsid).
detach=$(command -v setsid || true)
nohup $detach "$CONDA_PREFIX/lib/rmw_zenoh_cpp/rmw_zenohd" > "$HOME/.ros/log/zenohd.log" 2>&1 < /dev/null &
for _ in $(seq 200); do
    listening && exit 0
    sleep 0.05
done
echo "zenoh router did not start - see ~/.ros/log/zenohd.log" >&2
exit 1
