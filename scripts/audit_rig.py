"""Audit a rigged GLB: how well each bone's influence matches its location, plus
a bone -> owned-region table that is useful when animating.

Usage: python scripts/audit_rig.py <in.glb>
"""

import sys

import bpy  # must come before bmesh/mathutils when using the bpy pip wheel
import numpy as np


def main(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.name != "Icosphere"]
    ob = max(meshes, key=lambda o: len(o.data.vertices))
    arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]

    names = [g.name for g in ob.vertex_groups]
    V = np.array([v.co for v in ob.data.vertices])
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))

    dom = np.full(len(V), -1, dtype=np.int32)
    for v in ob.data.vertices:
        best, bw = -1, -1.0
        for g in v.groups:
            if g.weight > bw:
                bw, best = g.weight, g.group
        dom[v.index] = best

    bones = arm.data.bones
    heads = {n: np.array(bones[n].head_local[:]) for n in names}
    segs = {}
    for n in names:
        kids = [c.name for c in bones[n].children]
        end = np.mean([heads[k] for k in kids], axis=0) if kids else np.array(bones[n].tail_local[:])
        segs[n] = (heads[n], end)

    def pt_seg(p, a, b):
        ab = b - a
        denom = float(ab @ ab)
        if denom < 1e-12:
            return float(np.linalg.norm(p - a))
        t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
        return float(np.linalg.norm(p - (a + t * ab)))

    rng = np.random.default_rng(0)
    sample = rng.choice(len(V), size=min(6000, len(V)), replace=False)
    hit = 0
    dist_to_dom = []
    for i in sample:
        p = V[i]
        d = np.array([pt_seg(p, segs[n][0], segs[n][1]) for n in names])
        nearest = int(np.argmin(d))
        if nearest == dom[i]:
            hit += 1
        dist_to_dom.append(d[dom[i]] / diag)
    dist_to_dom = np.array(dist_to_dom)
    print(f"file: {path}")
    print(f"  verts={len(V)}  bones={len(names)}  diag={diag:.3f}")
    print(f"  dominant bone == nearest bone: {100.0*hit/len(sample):.1f}%  (n={len(sample)})")
    print("  dist(vertex, its dominant bone)/diag: median=%.3f  p90=%.3f" % (
        np.median(dist_to_dom), np.percentile(dist_to_dom, 90)))
    print()
    print("  %-10s %-24s %11s  %-24s %s" % ("bone", "head(x,y,z)", "owned verts", "centroid(x,y,z)", "offset/diag"))
    for j, n in enumerate(names):
        own = np.where(dom == j)[0]
        if len(own) == 0:
            print("  %-10s %-24s %11d  %-24s %s" % (n, str(np.round(heads[n], 3)), 0, "-", "-"))
            continue
        c = V[own].mean(0)
        off = float(np.linalg.norm(c - heads[n]) / diag)
        flag = "  <-- far" if off > 0.20 else ""
        print("  %-10s %-24s %11d  %-24s %.3f%s" % (
            n, str(np.round(heads[n], 3)), len(own), str(np.round(c, 3)), off, flag))


if __name__ == "__main__":
    main(sys.argv[-1])
