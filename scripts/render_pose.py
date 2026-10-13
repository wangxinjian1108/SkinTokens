"""Render a GLB with an optional bone pose, using a fixed camera.

Used to demonstrate the difference between the raw mesh and the rigged result:
the geometry is identical, so the only visible change appears once a bone moves.

Usage:
    python scripts/render_pose.py <glb> <out.png> <bone> <axis> <degrees>
    python scripts/render_pose.py <glb> <out.png> none
"""

import math
import sys

import bpy  # must come before bmesh/mathutils when using the bpy pip wheel
import bmesh
from mathutils import Matrix, Vector

# Camera direction is fixed so rest/posed renders of the same asset line up;
# the framing distance is derived from the mesh bbox before posing.
CAM_DIR = Vector((0.35, -1.0, 0.16))


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_camera(scene, meshes):
    lo = Vector((math.inf,) * 3)
    hi = Vector((-math.inf,) * 3)
    for ob in meshes:
        for corner in ob.bound_box:
            w = ob.matrix_world @ Vector(corner)
            lo = Vector((min(lo[i], w[i]) for i in range(3)))
            hi = Vector((max(hi[i], w[i]) for i in range(3)))
    if not math.isfinite(lo.x):
        lo, hi = Vector((-1, -1, -1)), Vector((1, 1, 1))
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 1e-3)

    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    direction = CAM_DIR.normalized()
    cam.location = center + direction * radius * 2.9
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = (center - cam.location).to_track_quat("-Z", "Y")
    cam_data.lens = 50
    # blender's default clip_end of 100 hides models whose own scale is large
    cam_data.clip_end = max(100.0, radius * 20)
    return cam


def setup_light(scene):
    for name, vec, energy in (
        ("key", Vector((1.1, -1.0, 0.9)), 3.5),
        ("fill", Vector((-1.2, -0.8, 0.3)), 1.5),
    ):
        data = bpy.data.lights.new(name, type="SUN")
        data.energy = energy
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
        light.rotation_mode = "QUATERNION"
        light.rotation_quaternion = (-vec).to_track_quat("-Z", "Y")


def add_skeleton_overlay(arms, joint_radius=0.006, bone_radius=0.003,
                         bone_color=(1.0, 0.35, 0.05, 1.0), emission=1.2,
                         mode="joints"):
    """Draw the rig as geometry.

    mode="joints": draw the kinematic chain, i.e. an edge from every joint to
        each of its children. This is how a skeleton is normally visualised and
        it is always connected.

    mode="bones": draw each bone from its head to its tail. The tails come from
        the model's predicted per-joint length/orientation, so on long limb
        segments the tail stops short of the child joint and the chain looks
        broken even though the hierarchy is intact.
    """
    bm = bmesh.new()
    if mode == "joints":
        points = []
        segments = []
        for arm in arms:
            for bone in arm.pose.bones:
                head = arm.matrix_world @ bone.head
                points.append(head)
                for child in bone.children:
                    segments.append((head, arm.matrix_world @ child.head))
    else:
        points = []
        segments = []
        for arm in arms:
            for bone in arm.pose.bones:
                head = arm.matrix_world @ bone.head
                points.append(head)
                segments.append((head, arm.matrix_world @ bone.tail))

    for p in points:
        sphere = bmesh.ops.create_uvsphere(
            bm, u_segments=12, v_segments=8, radius=joint_radius
        )
        bmesh.ops.transform(bm, matrix=Matrix.Translation(p), verts=sphere["verts"])

    for head, tail in segments:
        # NOTE: only ever transform the verts returned by the op that created
        # them. Tracking "new" verts with a tag flag silently also re-transforms
        # earlier geometry, which pushes the skeleton far outside the mesh.
        d = tail - head
        if d.length < 1e-6:
            continue
        ret = bmesh.ops.create_cone(
            bm, cap_ends=True, segments=8,
            radius1=bone_radius, radius2=bone_radius, depth=d.length
        )
        rot = Vector((0, 0, 1)).rotation_difference(d.normalized()).to_matrix().to_4x4()
        bmesh.ops.transform(
            bm, matrix=Matrix.Translation((head + tail) / 2) @ rot, verts=ret["verts"]
        )
    mesh = bpy.data.meshes.new("skeleton")
    bm.to_mesh(mesh)
    bm.free()
    ob = bpy.data.objects.new("skeleton", mesh)
    bpy.context.scene.collection.objects.link(ob)
    mat = bpy.data.materials.new("bone")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = bone_color
    bsdf.inputs["Emission Color"].default_value = bone_color
    bsdf.inputs["Emission Strength"].default_value = emission
    ob.data.materials.append(mat)
    return ob


def main(src, out, bone="none", axis="X", degrees=0.0):
    clear()
    bpy.ops.import_scene.gltf(filepath=src)
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]

    if bone != "none" and arms:
        arm = arms[0]
        pb = arm.pose.bones[bone]
        pb.rotation_mode = "XYZ"
        idx = "XYZ".index(axis)
        euler = [0.0, 0.0, 0.0]
        euler[idx] = math.radians(float(degrees))
        pb.rotation_euler = euler

    scene = bpy.context.scene
    bpy.context.view_layer.update()

    if arms:
        add_skeleton_overlay(arms)
        for arm in arms:
            arm.hide_render = True

    for ob in bpy.data.objects:
        if ob.name == "Icosphere":
            ob.hide_render = True

    meshes_for_cam = [
        o for o in bpy.data.objects if o.type == "MESH" and o.name != "Icosphere"
    ]
    setup_camera(scene, meshes_for_cam)
    setup_light(scene)
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.06, 1)
    scene.world = world

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 24
    scene.render.resolution_x = 900
    scene.render.resolution_y = 900
    scene.render.filepath = out
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    print(f"[render] wrote {out} (bone={bone} axis={axis} deg={degrees})")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    main(*argv)
