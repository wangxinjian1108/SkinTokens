"""Render the generated skeleton as real geometry (joint spheres + bone rods).

Cycles does not draw armature bones, so the skeleton is rebuilt as mesh
primitives placed at each bone's head/tail in world space.
"""

import math
import sys

import bpy  # must come before bmesh/mathutils when using the bpy pip wheel
import bmesh
from mathutils import Matrix, Vector


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def bounds(objects):
    lo = Vector((math.inf,) * 3)
    hi = Vector((-math.inf,) * 3)
    for ob in objects:
        for corner in ob.bound_box:
            world = ob.matrix_world @ Vector(corner)
            lo = Vector((min(lo[i], world[i]) for i in range(3)))
            hi = Vector((max(hi[i], world[i]) for i in range(3)))
    return lo, hi


def add_sphere(bm, center, radius):
    # Only transform the verts created by this op: tagging "new" verts and
    # selecting them later also re-transforms previously created geometry.
    ret = bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=radius)
    bmesh.ops.transform(bm, matrix=Matrix.Translation(center), verts=ret["verts"])


def add_rod(bm, a, b, radius):
    direction = b - a
    length = direction.length
    if length < 1e-6:
        return
    ret = bmesh.ops.create_cone(
        bm, cap_ends=True, segments=10, radius1=radius, radius2=radius, depth=length
    )
    verts = ret["verts"]
    rot = Vector((0, 0, 1)).rotation_difference(direction.normalized()).to_matrix().to_4x4()
    matrix = Matrix.Translation((a + b) / 2) @ rot
    bmesh.ops.transform(bm, matrix=matrix, verts=verts)


def build_skeleton_mesh(arms):
    bm = bmesh.new()
    joints = []
    for arm in arms:
        for bone in arm.data.bones:
            head = arm.matrix_world @ bone.head_local
            tail = arm.matrix_world @ bone.tail_local
            joints.append((head, tail))
    if not joints:
        raise SystemExit("no bones found")

    pts = Vector((0, 0, 0))
    for head, tail in joints:
        pts += head + tail
    pts /= 2 * len(joints)
    span = max((Vector(p) - pts).length for pair in joints for p in pair)
    radius = max(span * 0.012, 1e-4)

    for head, tail in joints:
        add_sphere(bm, head, radius * 1.6)
        add_rod(bm, head, tail, radius * 0.55)

    mesh = bpy.data.meshes.new("skeleton")
    bm.to_mesh(mesh)
    bm.free()
    ob = bpy.data.objects.new("skeleton", mesh)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def main(src, out):
    clear()
    bpy.ops.import_scene.gltf(filepath=src)
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.name != "Icosphere"]
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        raise SystemExit("no armature in file")

    skel = build_skeleton_mesh(arms)

    mat = bpy.data.materials.new("bone")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (1.0, 0.45, 0.1, 1)
    bsdf.inputs["Emission Color"].default_value = (1.0, 0.45, 0.1, 1)
    bsdf.inputs["Emission Strength"].default_value = 0.6
    skel.data.materials.append(mat)

    for ob in meshes:
        ob.hide_render = True

    lo, hi = bounds([skel])
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 1e-3)

    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    direction = Vector((0.9, -1.5, 0.35)).normalized()
    cam.location = center + direction * radius * 2.4
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = (center - cam.location).to_track_quat("-Z", "Y")
    cam_data.lens = 50

    light_data = bpy.data.lights.new("key", type="SUN")
    light_data.energy = 3.0
    light = bpy.data.objects.new("key", light_data)
    scene.collection.objects.link(light)
    light.rotation_mode = "QUATERNION"
    light.rotation_quaternion = Vector((-1.0, 1.0, -1.0)).to_track_quat("-Z", "Y")

    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.04, 0.04, 0.05, 1)
    scene.world = world

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 32
    scene.render.resolution_x = 900
    scene.render.resolution_y = 900
    scene.render.filepath = out
    scene.render.image_settings.file_format = "PNG"

    bpy.ops.render.render(write_still=True)
    print(f"[render] wrote {out} with {sum(len(a.data.bones) for a in arms)} bones")


if __name__ == "__main__":
    main(sys.argv[-2], sys.argv[-1])
