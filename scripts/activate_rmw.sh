# Sourced by pixi on activation (every platform, see pixi.toml): the ROS
# middleware. Both machines must use the same one.
#   cyclone (default) - Cyclone DDS. Under load on the relayed NTUSECURE link it
#                       drops camera frames like Fast DDS (Zenoh's one TCP link
#                       queued them: 1.2 s ping, YOLO round trip up to 1.7 s,
#                       2026-10-15), and it discovers faster than Fast DDS.
#   fastdds           - Fast DDS, as before: MDP_RMW=fastdds pixi run ...
# Peers: ROS_STATIC_PEERS in pixi.toml (the Tailscale addresses), for either.
if [ "${MDP_RMW:-cyclone}" = "fastdds" ]; then
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
else
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    # Left to itself Cyclone uses ONE interface, "selected arbitrarily": wlan0
    # on the Pi, an address the laptop cannot reach over NTUSECURE. So: loopback
    # (this machine's nodes - still up if Tailscale drops) + this machine's
    # Tailscale address (100.64.0.0/10). No Tailscale: Cyclone's own pick, the
    # robot still works on its own.
    if [ -z "${CYCLONEDDS_URI:-}" ]; then
        _ts_ip=$( { ip -4 -o addr 2>/dev/null | awk '{print $4}'; ifconfig 2>/dev/null | awk '/inet /{print $2}'; } \
            | cut -d/ -f1 | grep -E '^100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.' | head -n1)
        if [ -n "$_ts_ip" ]; then
            export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface address=\"127.0.0.1\"/><NetworkInterface address=\"$_ts_ip\"/></Interfaces></General></Domain></CycloneDDS>"
        else
            echo "activate_rmw.sh: no Tailscale address - Cyclone DDS on its default interface (no laptop link)" >&2
        fi
        unset _ts_ip
    fi
fi
