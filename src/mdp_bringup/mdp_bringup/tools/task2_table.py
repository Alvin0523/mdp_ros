"""Build task 2's checkpoint 1 -> 2 route table (tasks/task2_routes.py) - on the
laptop, once, and again whenever a setting it depends on changes (the runner
then says so and plans that leg live until it is rebuilt).

    pixi run task2-table                 every route; Hybrid A* where no clean shape fits
    pixi run task2-table --no-hybrid     clean shapes only (quick)
    pixi run task2-table --out FILE

Writes config/task2_routes.json.gz in the source tree (rebuild the workspace or use
--symlink-install, as `pixi run build` does, for the runner to see it), and
reports every layout it could not do.
"""

import argparse
import dataclasses
import gzip
import json
import os
import sys
import time
from collections import Counter

from mdp_algorithm.planning.planner import plan_leg
from mdp_algorithm.utils import params as planner_params
from mdp_bringup.tasks import task2_routes
from mdp_bringup.utils import obstacle_layout

CONFIG = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'config'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(CONFIG, task2_routes.TABLE_FILE))
    ap.add_argument('--layout', default=os.path.join(CONFIG, 'tasks.yaml'))
    ap.add_argument('--navigation', default=os.path.join(CONFIG, 'navigation.yaml'))
    ap.add_argument('--no-hybrid', action='store_true', help='clean shapes only, no Hybrid A*')
    args = ap.parse_args(argv)

    settings = task2_routes.task2_settings(args.navigation)
    car = planner_params.load(navigation_yaml=args.navigation)
    # Task 2's margin round the obstacles, as task2_runner plans with.
    car = dataclasses.replace(car, footprint_padding=float(settings['obstacle_padding']))
    planner_params.configure(car)
    arena = obstacle_layout.load_task2(args.layout)

    t0, last = time.monotonic(), [0.0]

    def progress(done, total, key, d2, width, how):
        if time.monotonic() - last[0] > 2.0 or done == total:
            last[0] = time.monotonic()
            print(f"  {done}/{total}  {key}  d2 {d2:.2f}  bar {width:.2f}  {how or 'NONE'}"
                  f"  ({time.monotonic() - t0:.0f} s)", flush=True)

    table = task2_routes.build_table(arena, car, settings, None if args.no_hybrid else plan_leg, progress)
    with (gzip.open(args.out, 'wt') if args.out.endswith('.gz') else open(args.out, 'w')) as f:
        json.dump(table, f, separators=(',', ':'))

    hows = Counter(h for rows in table['how'].values() for row in rows for h in row)
    print(f"\n{args.out}: {sum(hows.values())} layouts in {time.monotonic() - t0:.0f} s - "
          + ', '.join(f"{n} {h or 'NO ROUTE'}" for h, n in hows.most_common()))
    missing = [(key, d2, w) for key, rows in table['routes'].items() for d2, row in zip(table['d2'], rows)
               for w, r in zip(table['bar_width'], row) if r is None]
    for key, d2, w in missing:
        print(f"  NO ROUTE  {key}  d2 {d2:.2f} m  bar {w:.2f} m - planned live in a run")
    return 0


if __name__ == '__main__':
    sys.exit(main())
