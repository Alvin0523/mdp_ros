#!/usr/bin/env python3
"""Export a YOLO model to CoreML (Mac): src/mdp_vision/models/<name>.pt ->
<name>.mlpackage next to it. The detector then uses it on a Mac by itself
(yolo_detector.coreml_twin), on the Neural Engine. Run once per model, on the Mac:

    pixi run export-coreml                 # best_v5.pt
    pixi run export-coreml best_v4.pt
    pixi run export-coreml best_v5.pt --int8   # smaller/faster, check accuracy first

then rebuild mdp_vision (the task does) so the export is installed.
"""
import argparse
import os
import sys

MODELS = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'mdp_vision', 'models')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', nargs='?', default='best_v5.pt', help='file under src/mdp_vision/models/ or a path')
    ap.add_argument('--imgsz', type=int, default=640, help='input size (best_v5 was trained at 640)')
    ap.add_argument('--int8', action='store_true', help='8-bit weights instead of 16-bit (check detections after)')
    args = ap.parse_args()
    if sys.platform != 'darwin':
        print('CoreML export needs macOS (it only runs there anyway) - nothing to do here.')
        return 1
    path = args.model if os.sep in args.model else os.path.join(MODELS, args.model)
    from ultralytics import YOLO
    out = YOLO(path).export(format='coreml', imgsz=args.imgsz, half=not args.int8, int8=args.int8)
    print(f'CoreML model: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
