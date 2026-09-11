#!/usr/bin/env python3
"""
Import MDP symbol source images into models/symbols/ as GPU-ready 512x512 RGB
PNGs, alongside the .obj/.mtl that reference them.

    pixi run import-symbols /path/to/source/images

Then run `pixi run panels` to regenerate the matching panel meshes.

WHY THIS IS NOT JUST A COPY:

  * Downscale to a uniform 512x512. Source crops are ~2000-2500px and the size
    varies per file; a single power-of-two square is what a GPU texture wants.
    LANCZOS resampling specifically - these are hard-edged symbols on flat
    backgrounds, where cheaper filters alias the edges visibly.

  * RGBA -> RGB. The source crops carry an alpha channel that is fully opaque,
    so it is a wasted fourth channel. Flattening it drops ~25% of the data for
    no visual change. If a source image ever DOES have real transparency this
    script warns, because silently flattening it onto black would be wrong.

The repo deliberately keeps only the 512x512 result, with no higher-resolution
master - so keep the original source crops somewhere safe outside the tree if
you might need to regenerate at a different size. See
docs/rpi/sim_assets.md.
"""
import argparse
import os
import sys

try:
    from PIL import Image
except ImportError:
    raise SystemExit(
        "Pillow is required: pixi add pillow  (or run inside the pixi env)")

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)  # mdp_description/
TEX_DIR = os.path.join(PKG, 'models', 'symbols')

TARGET = 512


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Import symbol PNGs into models/symbols/ '
                    f'as {TARGET}x{TARGET} RGB.')
    ap.add_argument('source', help='directory of source *.png images')
    ap.add_argument('--size', type=int, default=TARGET,
                    help=f'output square size, default {TARGET}')
    ap.add_argument('--dry-run', action='store_true',
                    help='report what would change, write nothing')
    ns = ap.parse_args()

    if not os.path.isdir(ns.source):
        raise SystemExit(f'source dir not found: {ns.source}')

    names = sorted(f for f in os.listdir(ns.source) if f.lower().endswith('.png'))
    if not names:
        raise SystemExit(f'no *.png found in {ns.source}')

    os.makedirs(TEX_DIR, exist_ok=True)
    src_total = dst_total = 0
    warned = []

    for name in names:
        src = os.path.join(ns.source, name)
        dst = os.path.join(TEX_DIR, name)

        im = Image.open(src)
        src_size, src_mode = im.size, im.mode
        src_kb = os.path.getsize(src) / 1024
        src_total += src_kb

        if im.mode == 'RGBA':
            lo, _ = im.getchannel('A').getextrema()
            if lo < 255:
                # Real transparency - flattening would composite onto black.
                warned.append(name)

        if im.mode != 'RGB':
            im = im.convert('RGB')
        if im.size != (ns.size, ns.size):
            im = im.resize((ns.size, ns.size), Image.LANCZOS)

        if not ns.dry_run:
            im.save(dst, 'PNG', optimize=True)
            dst_total += os.path.getsize(dst) / 1024

        print(f'  {name:<20} {src_size[0]}\u00b2 {src_mode} '
              f'-> {ns.size}\u00b2 RGB   ({src_kb:.0f}K'
              + (f' -> {os.path.getsize(dst)/1024:.0f}K)'
                 if not ns.dry_run else ' , dry-run)'))

    verb = 'would import' if ns.dry_run else 'imported'
    print(f'\n{verb} {len(names)} textures into {TEX_DIR}')
    if not ns.dry_run:
        print(f'{src_total/1024:.1f} MB -> {dst_total/1024:.1f} MB')

    if warned:
        print(f'\nWARNING: {len(warned)} image(s) have real transparency, which '
              f'has been flattened onto black:', ', '.join(warned),
              '\n  Check these look right - the usual MDP crops are fully '
              'opaque, so this may mean the source is not what you expect.',
              file=sys.stderr)

    if not ns.dry_run:
        print('\nNext: pixi run panels   (regenerate the panel meshes)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
