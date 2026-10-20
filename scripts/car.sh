#!/usr/bin/env bash
# `pixi run car` / `pixi run sim-car`: both halves of a run + Foxglove's bridge in one zellij
# window, for any task. Launch arguments go to the car's half; the laptop's half gets the ones it
# takes (task:=, model:=) and sim:=true in sim.
#   scripts/car.sh real|sim [launch args...]     no task:= given -> task:=2
#   pixi run car task:=1 step:=true              pixi run sim-car task:=1 obstacles:=tablet
# Panes: left the car (real: waits for the Pi, SSHes in - scripts/pi_run.sh), right top the
# laptop, right bottom Foxglove. Ctrl+C stops a pane's command; the pane stays (scripts/keep.sh):
# ENTER runs it again. Quit everything: Ctrl+Q. Without zellij: prints the commands to run.
set -u
where="${1:?real or sim}"
shift
args=("$@")
case " ${args[*]-} " in *" task:="*) ;; *) args+=("task:=2") ;; esac

laptop=()
for a in "${args[@]}"; do
    case "$a" in task:=*|model:=*) laptop+=("$a") ;; esac
done
if [ "$where" = sim ]; then
    car=(pixi run sim "${args[@]}")
    laptop=(pixi run laptop sim:=true "${laptop[@]}")
    car_name="car (sim)"
else
    car=(bash scripts/pi_run.sh "${args[@]}")
    laptop=(pixi run laptop "${laptop[@]}")
    car_name="car (Pi)"
fi

if ! command -v zellij >/dev/null 2>&1; then
    echo "zellij is not installed - run these in three terminals (from mdp_ros/):"
    [ "$where" = sim ] && echo "  ${car[*]}" || echo "  on the Pi: pixi run pi ${args[*]}"
    echo "  ${laptop[*]}"
    echo "  pixi run foxglove"
    exit 1
fi

kdl_args() { local out="" a; for a in "$@"; do out+=" \"${a//\"/\\\"}\""; done; printf '%s' "$out"; }
layout="${TMPDIR:-/tmp}/mdp_car_$$.kdl"
cat > "$layout" <<EOF
layout {
    pane size=1 borderless=true {
        plugin location="zellij:tab-bar"
    }
    pane split_direction="vertical" {
        pane name="$car_name" command="bash" size="55%" {
            args "scripts/keep.sh"$(kdl_args "${car[@]}")
        }
        pane split_direction="horizontal" {
            pane name="laptop" command="bash" {
                args "scripts/keep.sh"$(kdl_args "${laptop[@]}")
            }
            pane name="foxglove" command="bash" size="30%" {
                args "scripts/keep.sh" "pixi" "run" "foxglove"
            }
        }
    }
    pane size=2 borderless=true {
        plugin location="zellij:status-bar"
    }
}
EOF
exec zellij --layout "$layout"
