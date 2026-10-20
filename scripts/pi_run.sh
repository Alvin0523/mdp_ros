#!/usr/bin/env bash
# The car's pane in `pixi run car` (scripts/car.sh): wait until the Pi answers on SSH,
# then log in and run `pixi run pi <launch args>` there. Ctrl+C stops it; the pane stays
# (zellij: ENTER runs this again - waits, logs in, runs).
#   scripts/pi_run.sh [launch args...]   e.g. task:=1 step:=true (a bare 0/1/2 = task:=N)
#   MDP_PI=user@host scripts/pi_run.sh   another Pi (default grp14@100.102.151.31, Tailscale)
args=()
for a in "$@"; do
    case "$a" in [0-2]) args+=("task:=$a") ;; *) args+=("$a") ;; esac
done
pi="${MDP_PI:-grp14@100.102.151.31}"
host="${pi#*@}"
printf 'waiting for the car (%s) ' "$pi"
# SSH's port, not ping: over Tailscale's relay ICMP got no answer while SSH worked (2026-10-15).
until nc -z -G 2 "$host" 22 >/dev/null 2>&1; do printf '.'; sleep 1; done
echo " up"
# pixi lives in ~/.pixi/bin, which only .bashrc puts on the PATH - not an SSH command's shell.
exec ssh -t -o ConnectTimeout=5 "$pi" "export PATH=\$HOME/.pixi/bin:\$PATH; cd ~/dev/mdp_ros && pixi run pi ${args[*]-}"
