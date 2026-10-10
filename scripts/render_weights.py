"""Colour a rigged mesh by its dominant bone, to inspect skin weight quality.

Each vertex gets the colour of the bone with the largest weight, drawn as a
point-domain colour attribute and rendered flat. A correct rig looks like clean
body-part patches; scrambled weights look like confetti.

Usage: python scripts/render_weights.py <in.glb> <out.png> [bone] [axis] [deg]
"""

import colorsys
import importlib.util
import math
import sys
from pathlib import Path

import bpy  # must come before bmesh/mathutils when using the bpy pip wheel

REPO = Path(__file__).resolve().parent.parent


def load_render_pose():
    spec = importlib.util.spec_from_file_location("render_pose", REPO / "scripts" / "render_pose.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def palette(n):
    # golden-angle hues, bright and well separated
    return [colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.75, 1.0) for i in range(n)]


def main(src, out, bone="none", axis="X", degrees=0.0):
    rp = load_render_pose()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)

    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.name != "Icosphere"]
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        raise SystemExit("no armature")

    if bone != "none":
        pb = arms[0].pose.bones[bone]
        pb.rotation_mode = "XYZ"
        euler = [0.0, 0.0, 0.0]
        euler["XYZ".index(axis)] = math.radians(float(degrees))
        pb.rotation_euler = euler
    bpy.context.view_layer.update()

    bone_names = [b.name for b in arms[0].data.bones]
    cols = palette(len(bone_names))
    bone_color = {n: cols[i] for i, n in enumerate(bone_names)}

    for ob in meshes:
        me = ob.data
        name_of = {g.index: g.name for g in ob.vertex_groups}
        attr = me.color_attributes.new(
            name="dominant", type="FLOAT_COLOR", domain="POINT"
        )
        for v in me.vertices:
            best, bw = None, -1.0
            for g in v.groups:
                if g.weight > bw:
                    bw, best = g.weight, name_of.get(g.group)
            r, gg, b = bone_color.get(best, (0.2, 0.2, 0.2))
            attr.data[v.index].color = (r, gg, b, 1.0)

        mat = bpy.data.materials.new("wcol")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out_node = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        ca = nt.nodes.new("ShaderNodeVertexColor")
        ca.layer_name = "dominant"
        nt.links.new(ca.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.9
        nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
        me.materials.clear()
        me.materials.append(mat)

    ico = bpy.data.objects.get("Icosphere")
    if ico:
        ico.hide_render = True

    scene = bpy.context.scene
    rp.setup_camera(scene, meshes)
    rp.setup_light(scene)
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.03, 0.03, 0.04, 1)
    scene.world = world

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 32
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 1000
    scene.render.filepath = out
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    print(f"[weights] wrote {out}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    main(*argv)
