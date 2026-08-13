#!/usr/bin/env bash
# Builds the standalone eProsima Micro-XRCE-DDS-Agent (host-side micro-ROS Agent),
# pinned to v3.0.1. NOT built via the ROS2 `micro_ros_agent` colcon package —
# that package's CMake SuperBuild pins an old (v2.4.3) Micro-XRCE-DDS-Agent that
# fails to compile against this workspace's fmt 12.x (spdlog/fmt API changes
# since 2022). The standalone v3.0.1 build has no such conflict and bridges into
# the same ROS2 DDS graph directly (default middleware is Fast DDS).
set -euo pipefail

AGENT_TAG="v3.0.1"
AGENT_DIR="../Micro-XRCE-DDS-Agent"

if [ ! -d "$AGENT_DIR" ]; then
    git clone --branch "$AGENT_TAG" --depth 1 \
        https://github.com/eProsima/Micro-XRCE-DDS-Agent.git "$AGENT_DIR"
fi

if [ -x "$AGENT_DIR/build/MicroXRCEAgent" ]; then
    echo "MicroXRCEAgent already built at $AGENT_DIR/build/MicroXRCEAgent"
    exit 0
fi

mkdir -p "$AGENT_DIR/build"
cd "$AGENT_DIR/build"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

echo "Built $AGENT_DIR/build/MicroXRCEAgent"
