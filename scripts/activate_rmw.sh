# Sourced by pixi on activation (every platform, see pixi.toml): which ROS
# middleware this shell uses. Both machines must use the same one.
#   zenoh (default) - rmw_zenoh: each machine runs a router (zenohd, started by
#                     mdp.launch.py), the laptop's connects to the Pi's over
#                     Tailscale. One TCP link instead of Fast DDS's discovery
#                     chatter, which took seconds over the relayed NTUSECURE
#                     link (2026-10-15).
#   fastdds         - Fast DDS with ROS_STATIC_PEERS (pixi.toml), as before.
# Pick one: MDP_RMW=fastdds pixi run ...
if [ "${MDP_RMW:-zenoh}" = "fastdds" ]; then
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
else
    export RMW_IMPLEMENTATION=rmw_zenoh_cpp
    # rmw_zenoh's own default config still sets a key Zenoh deprecated: without
    # this every node prints a two-line zenoh_config WARN at start.
    export RUST_LOG="${RUST_LOG:-warn,zenoh_config=error}"
fi
