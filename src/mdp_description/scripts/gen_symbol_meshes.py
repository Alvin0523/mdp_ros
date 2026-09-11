#!/usr/bin/env python3
"""
Generate UV-mapped quad meshes (.obj + .mtl) for each MDP symbol texture.

Gazebo Harmonic (ogre2) does NOT reliably render an <albedo_map> applied to a
primitive <box>/<plane> on a dynamically spawned entity - it shows up as a
blank/black panel. The supported way to put an image on a surface is a mesh
with UV coordinates + a material that names the texture (Wavefront .obj/.mtl or
Collada .dae). This script emits, for every PNG in
models/symbols/textures/, a tiny 2-triangle quad .obj that UV-maps the full
image, plus a matching .mtl. spawn_obstacles.py then references
<mesh><uri>...symbol.obj</uri> for each obstacle's decal face.

Quad geometry: a PANEL_SIDE x PANEL_SIDE square centred at the origin, lying in
the X-Z plane with its outward normal along +Y - a panel that faces "North" / +Y
by default, already at real size and already upright with image-up on world +Z.
So in the world SDF it needs <scale>1 1 1</scale> (NOT 0.061 - it is not a unit
quad) and yaw only, no roll or pitch:

    N -> yaw 0    E -> yaw -pi/2    S -> yaw pi    W -> yaw +pi/2

    pixi run panels

RUN THIS BY HAND after adding or removing a symbol PNG. Nothing invokes it
automatically - not the build, not the launches. (An earlier version of this
docstring claimed CMakeLists ran it at build time; it never did.) Output is
written into the source tree and installed with the rest of models/.

Consumers: the obstacle decals in mdp_description/worlds/task1_arena.sdf, which
is hand-maintained and is the single source of truth for the sim. (Previously
spawn_obstacles.py, then generate_arena_world.py - both deleted.) See
docs/rpi/sim_assets.md for the whole pipeline.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)  # mdp_description/
# ONE flat directory holding all three files per symbol: .png, .obj, .mtl.
# This is deliberate, not laziness - see the map_Kd note in write_obj().
SYM_DIR = os.path.join(PKG, 'models', 'symbols')
TEX_DIR = SYM_DIR
OUT_DIR = SYM_DIR

PANEL_SIDE = 0.061  # 6.1cm, matches the real scannable image size
H = PANEL_SIDE / 2.0


def write_obj(stem: str):
    """Write panels/<stem>.obj + <stem>.mtl for the given texture stem."""
    obj_path = os.path.join(OUT_DIR, f'{stem}.obj')
    mtl_path = os.path.join(OUT_DIR, f'{stem}.mtl')
    mtl_name = f'mat_{stem}'
    # map_Kd MUST be a bare filename, with the PNG sitting in this same
    # directory. A split layout (meshes in panels/, images in textures/, with
    # map_Kd ../textures/<stem>.png) was tried and FAILED IN GAZEBO: the mesh
    # loads but the texture never binds, so the decal renders as a plain grey
    # panel. Gazebo's mesh loader does not reliably follow parent-directory
    # traversal when resolving map_Kd. Keep all three files together.
    tex_rel = f'{stem}.png'

    # Quad in the X-Z plane, normal +Y. Vertices CCW seen from +Y.
    #   v1 (-H, 0, -H)  uv (0,0)
    #   v2 ( H, 0, -H)  uv (1,0)
    #   v3 ( H, 0,  H)  uv (1,1)
    #   v4 (-H, 0,  H)  uv (0,1)
    obj = f"""# MDP symbol panel quad for {stem} (auto-generated, do not edit)
mtllib {stem}.mtl
o panel_{stem}
v {-H:.5f} 0.0 {-H:.5f}
v {H:.5f} 0.0 {-H:.5f}
v {H:.5f} 0.0 {H:.5f}
v {-H:.5f} 0.0 {H:.5f}
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
vn 0.0 1.0 0.0
usemtl {mtl_name}
f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1
"""
    mtl = f"""# auto-generated, do not edit
newmtl {mtl_name}
Ka 1.000 1.000 1.000
Kd 1.000 1.000 1.000
Ks 0.000 0.000 0.000
d 1.0
illum 1
map_Kd {tex_rel}
"""
    with open(obj_path, 'w') as f:
        f.write(obj)
    with open(mtl_path, 'w') as f:
        f.write(mtl)


def main():
    if not os.path.isdir(TEX_DIR):
        raise SystemExit(f"texture dir not found: {TEX_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)
    stems = sorted(
        os.path.splitext(f)[0] for f in os.listdir(TEX_DIR)
        if f.lower().endswith('.png')
    )
    for stem in stems:
        write_obj(stem)
    print(f"Generated {len(stems)} symbol panel meshes into {OUT_DIR}")


if __name__ == '__main__':
    main()
