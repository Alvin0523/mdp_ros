#!/usr/bin/env python3
"""Time YOLO on this machine with every backend it has: CUDA (NVIDIA), CoreML
(.mlpackage, Mac), MPS (Apple GPU), CPU - same model, same frames.

    pixi run bench-yolo                       # best_v5, 100 frames of 640x480
    pixi run bench-yolo best_v5.pt --frames 200 --images path/to/jpgs

Frames: --images (a folder of .jpg/.png, e.g. a dataset's test set) or grey
640x480 frames (the camera's size). Prints ms per frame and FPS per backend,
and with --images how many boxes each finds (they should agree).
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

MODELS = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'mdp_vision', 'models')


def frames(images, n):
    if images:
        import cv2
        files = sorted(glob.glob(os.path.join(images, '*.jpg')) + glob.glob(os.path.join(images, '*.png')))
        imgs = [cv2.imread(f) for f in files[:n]]
        return [i for i in imgs if i is not None] or None
    return [np.full((480, 640, 3), 114, np.uint8)] * n


def bench(model, imgs, device, conf):
    kw = {} if device is None else {'device': device}
    for img in imgs[:5]:                       # warm up (first calls compile / load)
        model(img, verbose=False, conf=conf, **kw)
    t0, boxes = time.perf_counter(), 0
    for img in imgs:
        boxes += sum(len(r.boxes) for r in model(img, verbose=False, conf=conf, **kw))
    ms = (time.perf_counter() - t0) / len(imgs) * 1000.0
    return ms, boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', nargs='?', default='best_v5.pt')
    ap.add_argument('--frames', type=int, default=100)
    ap.add_argument('--images', help='folder of test images (default: grey 640x480 frames)')
    ap.add_argument('--conf', type=float, default=0.5, help='as the detector (min_confidence)')
    args = ap.parse_args()
    from ultralytics import YOLO
    import torch
    pt = args.model if os.sep in args.model else os.path.join(MODELS, args.model)
    imgs = frames(args.images, args.frames)
    if not imgs:
        print(f'No images in {args.images}')
        return 1
    runs = []
    if torch.cuda.is_available():
        runs.append((f'CUDA ({torch.cuda.get_device_name(0)})', pt, 0))
    ml = pt[:-3] + '.mlpackage'
    if sys.platform == 'darwin':
        if os.path.isdir(ml):
            runs.append(('CoreML (Neural Engine)', ml, None))
        else:
            print(f'(no {os.path.basename(ml)} - `pixi run export-coreml` first to time CoreML)')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        runs.append(('MPS (Apple GPU)', pt, 'mps'))
    runs.append(('CPU', pt, 'cpu'))
    print(f'{os.path.basename(pt)}, {len(imgs)} frames' + (f' from {args.images}' if args.images else ' (grey 640x480)'))
    for name, path, device in runs:
        try:
            ms, boxes = bench(YOLO(path, task='detect'), imgs, device, args.conf)
            extra = f'   {boxes} boxes' if args.images else ''
            print(f'  {name:28s} {ms:7.1f} ms/frame  {1000.0 / ms:6.1f} FPS{extra}')
        except Exception as e:                 # one backend failing shouldn't stop the rest
            print(f'  {name:28s} failed: {e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
