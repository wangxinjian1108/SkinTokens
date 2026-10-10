"""Render a rigged GLB in X-ray style: textured surface replaced by a
semi-transparent shell with a faint wireframe, so the generated skeleton reads
clearly.

Usage:
    python scripts/render_xray.py <glb> <out.png> [bone] [axis] [degrees]
"""

import importlib.util
import math
import sys
from pathlib import Path

import bpy  # must come before bmesh/mathutils when using the bpy pip wheel
from mathutils import Vector

REPO = Path(__file__).resolve().parent.parent


def _load_render_pose():
    spec = importlib.util.spec_from_file_location("render_pose", REPO / "scripts" / "render_pose.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_xray_material(alpha_max: float, alpha_min: float = 0.03):
    """Transparent shell whose silhouette stays readable.

    A wireframe overlay is deliberately not used: on dense scanned meshes
    (hundreds of thousands of triangles) a screen-space wireframe covers
    essentially every pixel and turns the shell opaque.
    """
    mat = bpy.data.materials.new("xray")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    layer = nt.nodes.new("ShaderNodeLayerWeight")
    rng = nt.nodes.new("ShaderNodeMapRange")

    layer.inputs["Blend"].default_value = 0.5
    rng.inputs["From Min"].default_value = 0.0
    rng.inputs["From Max"].default_value = 1.0
    rng.inputs["To Min"].default_value = alpha_min
    rng.inputs["To Max"].default_value = alpha_max

    bsdf.inputs["Base Color"].default_value = (0.62, 0.70, 0.86, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.45

    nt.links.new(layer.outputs["Facing"], rng.inputs["Value"])
    nt.links.new(rng.outputs["Result"], bsdf.inputs["Alpha"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def main(src, out, bone="none", axis="X", degrees=0.0, mode="joints"):
    rp = _load_render_pose()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)

    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.name != "Icosphere"]
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        raise SystemExit("no armature in file")

    lo = Vector((math.inf,) * 3)
    hi = Vector((-math.inf,) * 3)
    for ob in meshes:
        for corner in ob.bound_box:
            w = ob.matrix_world @ Vector(corner)
            lo = Vector((min(lo[i], w[i]) for i in range(3)))
            hi = Vector((max(hi[i], w[i]) for i in range(3)))
    radius = max((hi - lo).length / 2, 1e-3)

    if bone != "none":
        pb = arms[0].pose.bones[bone]
        pb.rotation_mode = "XYZ"
        euler = [0.0, 0.0, 0.0]
        euler["XYZ".index(axis)] = math.radians(float(degrees))
        pb.rotation_euler = euler

    bpy.context.view_layer.update()

    mat = make_xray_material(alpha_max=0.09, alpha_min=0.02)
    for ob in meshes:
        ob.data.materials.clear()
        ob.data.materials.append(mat)

    ico = bpy.data.objects.get("Icosphere")
    if ico:
        ico.hide_render = True

    rp.add_skeleton_overlay(
        arms,
        joint_radius=radius * 0.010,
        bone_radius=radius * 0.0042,
        bone_color=(1.0, 0.45, 0.10, 1.0),
        emission=2.2,
        mode=mode,
    )
    for arm in arms:
        arm.hide_render = True

    scene = bpy.context.scene
    rp.setup_camera(scene, meshes)
    rp.setup_light(scene)

    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.02, 0.02, 0.03, 1)
    scene.world = world

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 48
    scene.cycles.transparent_max_bounces = 16
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 1000
    scene.render.filepath = out
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    print(f"[render] wrote {out} (bone={bone} axis={axis} deg={degrees} mode={mode})")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    main(*argv)
