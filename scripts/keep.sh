#!/usr/bin/env bash
# Run a command in a pane that never closes (zellij/car.kdl): Ctrl+C stops the command,
# then "ENTER to run again" - more Ctrl+C there does nothing. (A zellij command pane
# closes on a second Ctrl+C once its command has stopped.) Quit the window: Ctrl+Q.
#   scripts/keep.sh <command> [args...]
# A handler, not an ignore: the command still gets Ctrl+C (exec resets handlers), this
# loop just carries on.
trap ':' INT
while true; do
    "$@"
    code=$?
    printf '\n\033[1;33m── stopped (exit %s) - ENTER to run again ──\033[0m\n' "$code"
    until read -r; do sleep 0.2; done        # Ctrl+C or Ctrl+D here: keep waiting
done
