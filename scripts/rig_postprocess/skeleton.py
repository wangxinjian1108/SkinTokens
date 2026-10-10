"""Skeleton extraction and topology-based semantic labelling.

TokenRig names its joints ``bone_0 ... bone_{N-1}``; there are no semantic
names in the file.  The hierarchy is a standard humanoid though, so the body
parts can be recovered from topology plus rest-pose geometry alone:

* the root is ``Hips``;
* the child whose subtree reaches highest continues the spine, until the first
  branch point - that branch point is the chest (``Spine1/Spine2``);
* at the chest, the child that stays closest to the spine axis is the ``Neck``
  (its child is ``Head``) and the two children reaching furthest sideways are
  the arms;
* the remaining children of the root are the legs;
* everything hanging off a ``Hand`` is a finger chain.

Convention (standard glTF, and the one Make-It-Animatable uses): Y up, the
character faces +Z, so in a right-handed Y-up frame **+X is the character's
left**.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    if "scale" in node:
        m = np.diag(list(node["scale"]) + [1.0]) @ m
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        m = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
            [0, 0, 0, 1],
        ]) @ m
    if "translation" in node:
        t = np.eye(4)
        t[:3, 3] = node["translation"]
        m = t @ m
    return m


@dataclass
class Skeleton:
    """Joints of a skinned glTF: node indices, hierarchy and rest transforms."""

    joint_nodes: list[int]           # joint order as referenced by the skin
    name: dict[int, str]             # node index -> node name
    parent: dict[int, int]           # node index -> parent joint node index (-1 for root)
    kids: dict[int, list[int]]
    matrix_world: dict[int, np.ndarray]
    matrix_local: dict[int, np.ndarray]
    skin_index: int

    @property
    def root(self) -> int:
        return next(j for j in self.joint_nodes if self.parent[j] == -1)

    def head(self, node: int) -> np.ndarray:
        return self.matrix_world[node][:3, 3]

    def subtree(self, node: int) -> list[int]:
        out, stack = [], [node]
        while stack:
            cur = stack.pop()
            out.append(cur)
            stack.extend(self.kids[cur])
        return out


def load_skeleton(gltf: dict) -> Skeleton:
    """Build the joint hierarchy from a glTF JSON dict (no BIN access needed)."""
    skin_index = 0
    joint_nodes = list(gltf["skins"][skin_index]["joints"])
    joint_set = set(joint_nodes)

    matrix_world: dict[int, np.ndarray] = {}

    def walk(node_index: int, parent_matrix: np.ndarray) -> None:
        m = parent_matrix @ _local_matrix(gltf["nodes"][node_index])
        matrix_world[node_index] = m
        for child in gltf["nodes"][node_index].get("children", []):
            walk(child, m)

    for scene in gltf["scenes"]:
        for root_node in scene["nodes"]:
            walk(root_node, np.eye(4))

    parent = {j: -1 for j in joint_nodes}
    kids = {j: [] for j in joint_nodes}
    for j in joint_nodes:
        for child in gltf["nodes"][j].get("children", []):
            if child in joint_set:
                parent[child] = j
                kids[j].append(child)

    return Skeleton(
        joint_nodes=joint_nodes,
        name={i: gltf["nodes"][i].get("name", f"node{i}") for i in range(len(gltf["nodes"]))},
        parent=parent,
        kids=kids,
        matrix_world=matrix_world,
        matrix_local={i: _local_matrix(gltf["nodes"][i]) for i in joint_nodes},
        skin_index=skin_index,
    )


_SPINE_NAMES = ["Spine", "Spine1", "Spine2", "Spine3"]


def label_humanoid(skel: Skeleton) -> dict[int, str]:
    """Return ``{joint_node: semantic_label}``; unknown joints get ``None``."""
    label: dict[int, str | None] = {j: None for j in skel.joint_nodes}
    root = skel.root
    label[root] = "Hips"

    def max_y(node: int) -> float:
        return max(skel.head(u)[1] for u in skel.subtree(node))

    def lateral_reach(node: int, axis_x: float) -> float:
        return max(abs(skel.head(u)[0] - axis_x) for u in skel.subtree(node))

    # spine: keep following the single highest-reaching child until a branch
    spine_child = max(skel.kids[root], key=max_y)
    spine_chain, cur = [spine_child], spine_child
    while len(skel.kids[cur]) == 1:
        cur = skel.kids[cur][0]
        spine_chain.append(cur)
    for step, node in enumerate(spine_chain):
        label[node] = _SPINE_NAMES[min(step, len(_SPINE_NAMES) - 1)]
    chest = spine_chain[-1]

    def follow(start: int, names: list[str]) -> list[int]:
        chain, cur = [], start
        for name in names:
            if cur is None:
                break
            label[cur] = name
            chain.append(cur)
            nxt = skel.kids[cur]
            cur = nxt[0] if len(nxt) == 1 else None
        return chain

    # neck / head: the chest child closest to the spine axis
    chest_kids = skel.kids[chest]
    if chest_kids:
        axis_x = skel.head(chest)[0]
        neck = min(chest_kids, key=lambda c: lateral_reach(c, axis_x))
        label[neck] = "Neck"
        neck_kids = skel.kids[neck]
        if len(neck_kids) == 1:
            follow(neck_kids[0], ["Head"])

        # arms: the chest children reaching furthest sideways (+X = left)
        arms = sorted((c for c in chest_kids if c != neck),
                      key=lambda c: -lateral_reach(c, axis_x))[:2]
        for node, side in zip(sorted(arms, key=lambda c: -skel.head(c)[0]), ("Left", "Right")):
            chain = follow(node, [f"{side}Shoulder", f"{side}Arm", f"{side}ForeArm", f"{side}Hand"])
            if not chain or label.get(chain[-1]) != f"{side}Hand":
                continue
            for finger_id, finger_root in enumerate(sorted(skel.kids[chain[-1]], key=lambda c: -skel.head(c)[0]), 1):
                stack = [(finger_root, 1)]
                while stack:
                    node_, depth = stack.pop()
                    label[node_] = f"{side}HandFinger{finger_id}_{depth}"
                    stack.extend((c, depth + 1) for c in skel.kids[node_])

    # legs: the remaining root children (+X = left)
    legs = [c for c in skel.kids[root] if c != spine_child]
    for node, side in zip(sorted(legs, key=lambda c: -skel.head(c)[0]), ("Left", "Right")):
        follow(node, [f"{side}UpLeg", f"{side}Leg", f"{side}Foot", f"{side}Toe"])

    return {node: name for node, name in label.items() if name is not None}


def index_by_label(labels: dict[int, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for node, name in labels.items():
        out.setdefault(name, node)
    return out
