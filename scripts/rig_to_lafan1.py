#!/usr/bin/env python3
"""Collapse a TokenRig / SkinTokens rig onto the 22-joint LaFAN1 skeleton.

TokenRig picks the joint count per asset (34 for a typical human, but 47/64/73
happen too - even for the same mesh, since sampling is stochastic).  This tool
turns any of those into the fixed 22-joint skeleton used by LaFAN1 BVH files,
so motion retargeting has a stable, name-based mapping.

Joints that are not part of the target skeleton are removed, but **their
skinning weights are first added to the nearest surviving ancestor** (fingers ->
hand, toe tips -> toe, ...).  Weight rows therefore keep summing to 1.0 and the
mesh is not distorted; the affected vertices simply become rigid with respect
to that ancestor.  Positions / normals / uvs / indices / materials / textures
are copied through untouched.

Examples
--------
    python scripts/rig_to_lafan1.py --input rig.glb --output rig_22.glb --verify
    python scripts/rig_to_lafan1.py --input runs/raw --output runs/rig22
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rig_postprocess import collapse as collapse_mod          # noqa: E402
from rig_postprocess import gltf_io                          # noqa: E402
from rig_postprocess.lafan1 import LAFAN1_JOINTS             # noqa: E402
from rig_postprocess.skeleton import label_humanoid, load_skeleton  # noqa: E402


def discover_inputs(path: str) -> list[str]:
    if os.path.isdir(path):
        return sorted(glob(os.path.join(path, "**", "*.glb"), recursive=True))
    return [path]


def plan_outputs(input_root: str, output_root: str, sources: list[str]) -> list[str]:
    """One GLB file input -> exactly ``output_root``; a directory -> mirror it."""
    if os.path.isfile(input_root):
        return [output_root]
    return [
        os.path.splitext(os.path.join(output_root, os.path.relpath(source, input_root)))[0] + ".glb"
        for source in sources
    ]


def is_inside(path: str, root: str) -> bool:
    path, root = os.path.abspath(path), os.path.abspath(root)
    return path == root or path.startswith(root + os.sep)


def verify(source: str, result: str) -> dict:
    """Independent checks on the produced file."""
    src_gltf, _ = gltf_io.read_glb(source)
    dst_gltf, dst_bin = gltf_io.read_glb(result)

    src_attrs = src_gltf["meshes"][0]["primitives"][0]["attributes"]
    dst_attrs = dst_gltf["meshes"][0]["primitives"][0]["attributes"]

    joint_nodes = dst_gltf["skins"][0]["joints"]
    names = [dst_gltf["nodes"][j].get("name") for j in joint_nodes]
    weights = gltf_io.read_accessor(dst_gltf, dst_bin, dst_attrs["WEIGHTS_0"]).astype(np.float64)

    src_skel = load_skeleton(src_gltf)
    src_labels = label_humanoid(src_skel)
    src_by_label = {name: node for node, name in src_labels.items()}
    dst_skel = load_skeleton(dst_gltf)
    dst_by_name = {dst_skel.name[j]: j for j in joint_nodes}

    position_error = max(
        (float(np.linalg.norm(src_skel.head(src_by_label[name]) - dst_skel.head(dst_by_name[name])))
         for name in LAFAN1_JOINTS if name in src_by_label and name in dst_by_name),
        default=float("nan"),
    )
    return {
        "joint_count": len(joint_nodes),
        "names_match_lafan1_order": names == LAFAN1_JOINTS,
        "geometry_accessors_untouched": all(
            src_attrs[k] == dst_attrs[k] for k in ("POSITION", "NORMAL", "TEXCOORD_0") if k in src_attrs
        ),
        "max_weight_row_sum_error": float(np.abs(weights.sum(axis=1) - 1.0).max()),
        "max_joint_position_error": position_error,
    }


def process_one(source: str, destination: str, args: argparse.Namespace) -> dict:
    gltf, bin_parts, report = collapse_mod.collapse_to_target(
        source,
        renormalize=not args.no_renormalize,
        strict=not args.non_strict,
    )
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    gltf_io.write_glb(destination, gltf, bin_parts)

    result = report.as_dict()
    if args.verify:
        result["verification"] = verify(source, destination)
    if args.map_out:
        map_path = os.path.splitext(destination)[0] + ".bone_map.json"
        with open(map_path, "w") as fh:
            json.dump(result["mapping"], fh, indent=2, sort_keys=True)
        result["mapping_file"] = map_path
    return result


def print_report(result: dict) -> None:
    dropped = result["dropped_by_reason"]
    dropped_counts = ", ".join(f"{k}={len(v)}" for k, v in sorted(dropped.items())) or "none"
    print(f"  {result['source_joints']} -> {result['kept_joints']} joints "
          f"({result['vertices']} verts) | dropped: {dropped_counts}")
    print(f"  weight mass re-routed: {result['weight_mass_moved'] * 100:.2f}% | "
          f"row-sum error {result['max_row_sum_error_before']:.2e} -> {result['max_row_sum_error_after']:.2e}")
    if result["coincident_pairs"]:
        pairs = ", ".join(f"{a}~{b}({d:.5f})" for a, b, d in result["coincident_pairs"][:4])
        print(f"  note: coincident parent/child joints in the source rig: {pairs}")
    if "verification" in result:
        for key, value in result["verification"].items():
            print(f"  verify.{key} = {value}")
    if "mapping_file" in result:
        print(f"  mapping -> {result['mapping_file']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="GLB file or directory")
    parser.add_argument("--output", required=True, help="GLB file or directory")
    parser.add_argument("--map-out", action="store_true", help="also write <output>.bone_map.json")
    parser.add_argument("--verify", action="store_true", help="run consistency checks on the result")
    parser.add_argument("--no-renormalize", action="store_true", help="keep raw merged weights")
    parser.add_argument("--non-strict", action="store_true",
                        help="do not fail when a target joint is missing from the source rig")
    parser.add_argument("--report", help="write the full report of every file as JSON here")
    args = parser.parse_args()

    # never walk into the output directory when it lives inside the input tree
    sources = [s for s in discover_inputs(args.input) if not is_inside(s, args.output)]
    if not sources:
        print(f"no .glb files found under {args.input}", file=sys.stderr)
        return 1
    destinations = plan_outputs(args.input, args.output, sources)

    reports = []
    failures = 0
    for source, destination in zip(sources, destinations):
        print(f"[{os.path.basename(source)}] -> {destination}")
        try:
            result = process_one(source, destination, args)
        except Exception as exc:  # noqa: BLE001 - report and continue with the batch
            failures += 1
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        print_report(result)
        reports.append(result)

    if args.report:
        with open(args.report, "w") as fh:
            json.dump(reports, fh, indent=2, sort_keys=True)
        print(f"report -> {args.report}")

    print("\ndone: %d ok, %d failed" % (len(reports), failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
