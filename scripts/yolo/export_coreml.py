#!/usr/bin/env python3
"""Export a YOLO model to CoreML (Mac): src/mdp_vision/models/<name>.pt ->
<name>.mlpackage next to it. The detector then uses it on a Mac by itself
(yolo_detector.coreml_twin), on the Neural Engine. Run once per model, on the Mac:

    pixi run export-coreml                 # best_v5.pt
    pixi run export-coreml best_v4.pt
    pixi run export-coreml task2_yolo26s.pt    # task 2: arrows + bullseye
    pixi run export-coreml best_v5.pt --int8   # smaller/faster, check accuracy first

then rebuild mdp_vision (the task does) so the export is installed.
"""
import argparse
import os
import sys

MODELS = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'mdp_vision', 'models')


def patch_int_cast():
    """YOLO26's attention block casts a 1-element shape tensor to int, and numpy >= 2.4
    (2.5 here) no longer turns a 1-element array into a scalar, so coremltools 9 fails
    ("only 0-dimensional arrays can be converted to Python scalars", task2_yolo26s.pt
    2026-10-15; Ultralytics wants numpy <= 2.3.5 for this, coremltools issue 2633). Same
    value via .item(); detections matched the .pt on all 27 test images (<= 1 px)."""
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch import ops
    cast = ops._cast

    def _cast(context, node, dtype, dtype_name):
        x = ops._get_inputs(context, node, expected=1)[0]
        if isinstance(x.val, np.ndarray) and x.val.ndim > 0 and x.val.size == 1:
            context.add(mb.const(val=dtype(x.val.item()), name=node.name))
            return
        cast(context, node, dtype, dtype_name)
    ops._cast = _cast


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
    patch_int_cast()
    from ultralytics import YOLO
    out = YOLO(path).export(format='coreml', imgsz=args.imgsz, half=not args.int8, int8=args.int8)
    print(f'CoreML model: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
