"""Move each bone's tail onto its unique child's head, then re-export the GLB.

This is the same thing `connect_tail_to_unique_child=True` does inside
BpyParser.export, applied as a post-process so an already generated file can be
fixed without re-running the model.

Only the bone tail (a display/length property) is touched. Bone heads, the
parent hierarchy and every skin weight stay exactly as they were, so the
deformation is unchanged; the skeleton simply stops looking disconnected.

Usage: python scripts/connect_bone_tails.py <in.glb> <out.glb>
"""

import sys

import bpy  # must come before bmesh/mathutils when using the bpy pip wheel


def main(src, dst):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)

    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not armatures:
        raise SystemExit("no armature in input")

    # The glTF importer adds a bone display shape we do not want to re-export.
    for ob in list(bpy.data.objects):
        if ob.name == "Icosphere":
            bpy.data.objects.remove(ob, do_unlink=True)

    moved = 0
    for arm in armatures:
        bpy.context.view_layer.objects.active = arm
        arm.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        for bone in arm.data.edit_bones:
            kids = list(bone.children)
            if len(kids) == 1:
                bone.tail = kids[0].head.copy()
                moved += 1
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.export_scene.gltf(filepath=dst)
    print(f"[connect] moved {moved} bone tails -> {dst}")


if __name__ == "__main__":
    main(sys.argv[-2], sys.argv[-1])
