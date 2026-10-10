"""Post-process TokenRig / SkinTokens output rigs.

TokenRig emits a rig whose joint count is decided by the autoregressive model
(34 for a typical human, but it can be 47, 64, 73, ... depending on the asset
and on sampling).  Downstream use (e.g. BVH motion retargeting) needs a fixed,
well-defined skeleton instead.

This package collapses an arbitrary TokenRig rig onto a target skeleton
(``lafan1``: the 22-joint Mixamo-style hierarchy used by LaFAN1 BVH files),
transfers the weights of every dropped joint into the nearest surviving
ancestor, and rewrites the GLB - all in pure Python, without Blender.

The mesh (positions, normals, uvs, indices, materials, textures) is copied
through untouched.
"""

from .lafan1 import LAFAN1_JOINTS, LAFAN1_PARENTS
from .collapse import collapse_to_target, CollapseReport
from .skeleton import Skeleton, label_humanoid

__all__ = [
    "LAFAN1_JOINTS",
    "LAFAN1_PARENTS",
    "collapse_to_target",
    "CollapseReport",
    "Skeleton",
    "label_humanoid",
]
