"""Collapse an arbitrary TokenRig skeleton onto a fixed target skeleton.

Every joint of the source rig that is not part of the target skeleton is
removed, and its skinning weights are **added to the nearest surviving
ancestor** first (so vertex weight rows keep summing to 1.0 and the mesh does
not collapse).  The mesh itself is copied through byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import gltf_io
from .lafan1 import LAFAN1_JOINTS, is_finger
from .skeleton import index_by_label, label_humanoid, load_skeleton

COMPONENT_TYPE_UINT8 = 5121
COMPONENT_TYPE_UINT16 = 5123


@dataclass
class CollapseReport:
    source: str = ""
    vertices: int = 0
    source_joints: int = 0
    kept_joints: int = 0
    missing_targets: list[str] = field(default_factory=list)
    dropped_by_reason: dict[str, list[str]] = field(default_factory=dict)
    weight_mass_moved: float = 0.0            # fraction of all weight mass
    max_row_sum_error_before: float = 0.0
    max_row_sum_error_after: float = 0.0
    coincident_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)   # source node name -> target name

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "vertices": self.vertices,
            "source_joints": self.source_joints,
            "kept_joints": self.kept_joints,
            "missing_targets": self.missing_targets,
            "dropped_by_reason": self.dropped_by_reason,
            "weight_mass_moved": self.weight_mass_moved,
            "max_row_sum_error_before": self.max_row_sum_error_before,
            "max_row_sum_error_after": self.max_row_sum_error_after,
            "coincident_pairs": self.coincident_pairs,
            "mapping": self.mapping,
        }


def _coalesce(js: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum weights that map onto the same joint and pack them into 4 slots."""
    n_vertices = js.shape[0]
    out_j = np.full((n_vertices, 4), -1, dtype=np.int16)
    out_w = np.zeros((n_vertices, 4), dtype=np.float32)
    for slot in range(js.shape[1]):
        joint = js[:, slot].astype(np.int16)
        weight = weights[:, slot]
        active = weight > 0
        if not active.any():
            continue
        match = (out_j == joint[:, None]) & active[:, None]
        has_match = match.any(axis=1)
        free_slot = (out_j >= 0).sum(axis=1)
        dest = np.where(has_match, match.argmax(axis=1), free_slot)
        rows = np.nonzero(active)[0]
        out_j[rows, dest[rows]] = joint[rows]
        out_w[rows, dest[rows]] += weight[rows]
    out_j[out_j < 0] = 0
    return out_j.astype(np.uint8), out_w


def _node_matrix_column_major(matrix: np.ndarray) -> list[float]:
    return [float(v) for v in matrix.T.reshape(-1)]


def collapse_to_target(
    path: str,
    target_joints: list[str] | None = None,
    *,
    renormalize: bool = True,
    strict: bool = True,
    coincident_eps: float = 1e-3,
) -> tuple[dict, list, CollapseReport]:
    """Build the collapsed GLB description for ``path``.

    Returns ``(gltf_json, bin_parts, report)`` ready for :func:`gltf_io.write_glb`.
    """
    target_joints = target_joints or LAFAN1_JOINTS
    gltf, bin_chunk = gltf_io.read_glb(path)
    if not gltf.get("skins"):
        raise ValueError(f"{path}: no skin found")
    report = CollapseReport(source=path)

    skel = load_skeleton(gltf)
    labels = label_humanoid(skel)
    by_label = index_by_label(labels)

    report.source_joints = len(skel.joint_nodes)
    report.missing_targets = [name for name in target_joints if name not in by_label]
    if report.missing_targets and strict:
        raise ValueError(
            f"{path}: skeleton is missing target joints {report.missing_targets}; "
            f"pass strict=False to keep going"
        )

    kept_nodes = [by_label[name] for name in target_joints if name in by_label]
    kept_set = set(kept_nodes)
    report.kept_joints = len(kept_nodes)

    # --- where does every dropped joint send its weight? -------------------
    new_index_of_node = {node: k for k, node in enumerate(kept_nodes)}
    new_slot_of_old: dict[int, int] = {}
    for old_index, node in enumerate(skel.joint_nodes):
        if node in kept_set:
            new_slot_of_old[old_index] = new_index_of_node[node]
            continue
        reason = "finger" if is_finger(labels.get(node, "")) else (
            "unlabelled" if node not in labels else "extra")
        report.dropped_by_reason.setdefault(reason, []).append(skel.name[node])

        ancestor = skel.parent[node]
        while ancestor != -1 and ancestor not in kept_set:
            ancestor = skel.parent[ancestor]
        if ancestor == -1:
            raise ValueError(f"{path}: dropped joint {skel.name[node]} has no surviving ancestor")
        new_slot_of_old[old_index] = new_index_of_node[ancestor]

    # --- skin weights ------------------------------------------------------
    primitive = gltf["meshes"][0]["primitives"][0]
    joints_accessor = primitive["attributes"]["JOINTS_0"]
    weights_accessor = primitive["attributes"]["WEIGHTS_0"]
    joint_component_type = gltf["accessors"][joints_accessor]["componentType"]
    if joint_component_type not in (COMPONENT_TYPE_UINT8, COMPONENT_TYPE_UINT16):
        raise ValueError(f"{path}: unsupported JOINTS_0 componentType {joint_component_type}")
    joint_dtype = np.uint8 if joint_component_type == COMPONENT_TYPE_UINT8 else np.uint16

    old_joints = gltf_io.read_accessor(gltf, bin_chunk, joints_accessor).astype(np.int64)
    old_weights = gltf_io.read_accessor(gltf, bin_chunk, weights_accessor).astype(np.float64)
    report.vertices = old_joints.shape[0]

    row_sums_before = old_weights.sum(axis=1)
    report.max_row_sum_error_before = float(np.abs(row_sums_before - 1.0).max())

    remap = np.zeros(len(skel.joint_nodes), dtype=np.uint8)
    for old_index, new_slot in new_slot_of_old.items():
        remap[old_index] = new_slot
    new_joints = remap[np.clip(old_joints, 0, len(skel.joint_nodes) - 1)]

    # how much of the total weight mass is being re-routed
    dropped_old = [i for i, node in enumerate(skel.joint_nodes) if node not in kept_set]
    if dropped_old:
        moved = np.isin(old_joints, dropped_old)
        report.weight_mass_moved = float((old_weights * moved).sum() / old_weights.sum())

    packed_joints, packed_weights = _coalesce(new_joints, old_weights.astype(np.float32))
    row_sums_after = packed_weights.sum(axis=1)
    if renormalize:
        bad = np.abs(row_sums_after - 1.0) > 1e-5
        if bad.any():
            packed_weights[bad] /= row_sums_after[bad][:, None]
        row_sums_after = packed_weights.sum(axis=1)
    report.max_row_sum_error_after = float(np.abs(row_sums_after - 1.0).max())

    order = np.argsort(-packed_weights, axis=1, kind="stable")
    packed_joints = np.take_along_axis(packed_joints, order, axis=1)
    packed_weights = np.take_along_axis(packed_weights, order, axis=1)

    # --- inverse bind matrices --------------------------------------------
    old_ibm = gltf_io.read_accessor(
        gltf, bin_chunk, gltf["skins"][0]["inverseBindMatrices"]).astype(np.float32)
    new_ibm = np.stack([old_ibm[skel.joint_nodes.index(node)] for node in kept_nodes])

    # --- rewrite nodes / scene / skin -------------------------------------
    node_parent: dict[int, int] = {}
    for i, node in enumerate(gltf["nodes"]):
        for child in node.get("children", []):
            node_parent[child] = i

    joint_set = set(skel.joint_nodes)
    other_nodes = [i for i in range(len(gltf["nodes"])) if i not in joint_set]
    new_nodes: list[dict] = []
    node_index_new = {node: k for k, node in enumerate(other_nodes)}
    for node in kept_nodes:
        node_index_new[node] = len(other_nodes) + kept_nodes.index(node)

    def new_children(old_node: int) -> list[int]:
        out = []
        for child in gltf["nodes"][old_node].get("children", []):
            if child in joint_set:
                if child in kept_set:
                    out.append(node_index_new[child])
            else:
                out.append(node_index_new[child])
        return out

    for old_index in other_nodes:
        node = dict(gltf["nodes"][old_index])
        children = new_children(old_index)
        if children:
            node["children"] = children
        else:
            node.pop("children", None)
        new_nodes.append(node)

    for step, node in enumerate(kept_nodes):
        parent = skel.parent[node]
        while parent != -1 and parent not in kept_set:
            parent = skel.parent[parent]
        if parent == -1:                       # attach to the original non-joint parent
            parent = node_parent[node]
            while parent in joint_set:
                parent = node_parent[parent]
        local = np.linalg.inv(skel.matrix_world[parent]) @ skel.matrix_world[node]
        entry = {
            "name": target_joints[step],
            "matrix": _node_matrix_column_major(local),
            "children": [node_index_new[c] for c in skel.kids[node] if c in kept_set],
        }
        if not entry["children"]:
            del entry["children"]
        new_nodes.append(entry)
        report.mapping[skel.name[node]] = target_joints[step]

    # make sure the armature/root node actually lists the retained root joint
    root_joint = kept_nodes[next(k for k, n in enumerate(kept_nodes) if skel.parent[n] == -1)]
    root_old_parent = node_parent[root_joint]
    while root_old_parent in joint_set:
        root_old_parent = node_parent[root_old_parent]
    top = new_nodes[node_index_new[root_old_parent]]
    top.setdefault("children", [])
    if node_index_new[root_joint] not in top["children"]:
        top["children"].append(node_index_new[root_joint])
        top["children"].sort()

    gltf["nodes"] = new_nodes
    gltf["scenes"] = [{"name": gltf["scenes"][0].get("name", "Scene"),
                       "nodes": [node_index_new[n] for n in gltf["scenes"][0]["nodes"]]}]
    gltf["scene"] = 0

    # --- patch the BIN in place --------------------------------------------
    # JOINTS_0 / WEIGHTS_0 keep their accessors (same count and type); only the
    # bytes change.  The inverse-bind-matrix accessor just gets shorter.  Mesh
    # data is never touched and the file does not grow.
    ibm_accessor = gltf["skins"][0]["inverseBindMatrices"]
    patches: list[tuple[int, bytes]] = []
    for accessor_index, array in (
        (joints_accessor, packed_joints.astype(joint_dtype)),
        (weights_accessor, packed_weights.astype(np.float32)),
        (ibm_accessor, new_ibm.astype(np.float32)),
    ):
        offset, _, _, _ = gltf_io.accessor_layout(gltf, accessor_index)
        view = gltf["bufferViews"][gltf["accessors"][accessor_index]["bufferView"]]
        payload = np.ascontiguousarray(array).tobytes()
        if len(payload) > view["byteLength"]:
            raise ValueError(f"accessor {accessor_index}: patched data does not fit its buffer view")
        patches.append((offset, payload))

    gltf["accessors"][ibm_accessor]["count"] = len(kept_nodes)
    gltf["skins"][0]["joints"] = [node_index_new[n] for n in kept_nodes]
    if "skeleton" in gltf["skins"][0]:
        gltf["skins"][0]["skeleton"] = node_index_new[kept_nodes[0]]

    # --- diagnostics -------------------------------------------------------
    height = float(np.ptp([skel.head(j)[1] for j in skel.joint_nodes])) or 1.0
    for node in skel.joint_nodes:
        parent = skel.parent[node]
        if parent == -1:
            continue
        distance = float(np.linalg.norm(skel.head(node) - skel.head(parent)))
        if distance < coincident_eps * height:
            report.coincident_pairs.append((skel.name[parent], skel.name[node], distance))

    if bin_chunk is None:
        raise ValueError(f"{path}: expected a BIN chunk")
    bin_parts = gltf_io.patch_bin(bin_chunk, patches)
    return gltf, bin_parts, report
