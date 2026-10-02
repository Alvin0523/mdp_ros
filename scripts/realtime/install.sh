#!/usr/bin/env bash
# Allow real-time priority for ros2_control's controller loop (99-mdp-realtime.conf).
# Run once per machine on the Pi: pixi run realtime (asks for the sudo password),
# then log out and in again (new SSH session) - limits apply from the next login.
set -euo pipefail
CONF=99-mdp-realtime.conf
USER_NAME=$(id -un)
cd "$(dirname "$0")"

sudo groupadd -f realtime
sudo usermod -aG realtime "$USER_NAME"
sudo install -m 0644 "$CONF" "/etc/security/limits.d/$CONF"

echo "Installed. Log out and in again (new SSH session), then check:"
echo "  ulimit -r          -> 99"
echo "  pixi run pi        -> no 'Could not enable FIFO RT scheduling' warning"
