"""Render a quick EEVEE preview of a GLB (mesh + armature) for visual inspection."""

import math
import sys

import bpy
from mathutils import Vector


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)


def bounds(objects):
    lo = Vector((math.inf, math.inf, math.inf))
    hi = Vector((-math.inf, -math.inf, -math.inf))
    for ob in objects:
        for corner in ob.bound_box:
            world = ob.matrix_world @ Vector(corner)
            lo = Vector((min(lo[i], world[i]) for i in range(3)))
            hi = Vector((max(hi[i], world[i]) for i in range(3)))
    return lo, hi


def main(src, out):
    clear()
    import_glb(src)

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    targets = meshes + arms
    if not targets:
        raise SystemExit("nothing to render")

    lo, hi = bounds(meshes or targets)
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 1e-3)

    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    direction = Vector((1.0, -1.4, 0.7)).normalized()
    cam.location = center + direction * radius * 2.6
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = (center - cam.location).to_track_quat("-Z", "Y")
    cam_data.lens = 50

    for name, vec in (("key", Vector((1.6, -1.2, 1.4))), ("fill", Vector((-1.4, -1.0, 0.4)))):
        light_data = bpy.data.lights.new(name, type="SUN")
        light_data.energy = 4.0 if name == "key" else 1.8
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light.rotation_mode = "QUATERNION"
        light.rotation_quaternion = (-vec).to_track_quat("-Z", "Y")

    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.06, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    scene.world = world

    # Headless boxes usually lack EGL, so EEVEE cannot start; Cycles on CPU always works.
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 900
    scene.render.resolution_y = 900
    scene.render.filepath = out
    scene.render.image_settings.file_format = "PNG"

    # Show the armature as overlays only; keep bones out of the shaded render.
    for arm in arms:
        arm.hide_render = True

    bpy.ops.render.render(write_still=True)
    print(f"[render] wrote {out}")


if __name__ == "__main__":
    main(sys.argv[-2], sys.argv[-1])
