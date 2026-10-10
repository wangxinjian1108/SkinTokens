# Collapsing TokenRig output to a fixed 22-joint LaFAN1 skeleton

`scripts/rig_to_lafan1.py` turns any rig produced by TokenRig (SkinTokens) into
the fixed **22-joint skeleton used by LaFAN1 BVH files**, so that motion
retargeting can rely on a stable, name-based joint mapping.

```
python scripts/rig_to_lafan1.py --input rig.glb --output rig_22.glb --verify --map-out
python scripts/rig_to_lafan1.py --input runs/raw --output runs/rig22 --verify
```

## Why this is needed

TokenRig predicts the skeleton autoregressively, so **the joint count is chosen
per asset and is not deterministic**. Measured on this machine:

| asset | mesh vertices | joints |
|---|---|---|
| `results/test.glb` | 1 192 813 | **34** |
| `results/dbg.glb` | 1 192 813 | **34** |
| `results/dbg2.glb` | 1 192 813 | **73** |
| `results/giraffe.glb` | 62 833 | 47 |
| `Ped_Assets_output/ped1.glb` | 1 132 648 | 34 |
| `Ped_Assets_output/ped3.glb` | 1 166 300 | **64** |

Note `test.glb` / `dbg.glb` / `dbg2.glb`: **identical vertex count (same mesh),
34 vs 73 joints** - the difference is sampling (`do_sample=True`,
`temperature=1.5`, `num_beams=10`). So the total joint count is *not* a quality
signal; a 64-joint result (ped3, five fingers per hand) is a *more detailed*
rig, not a broken one.

What is stable across every asset we tested is the **topology**: one root, a
3-bone spine, neck/head, two `Shoulder -> Arm -> ForeArm -> Hand` chains and two
`UpLeg -> Leg -> Foot -> Toe` chains. The extra joints are fingers, toe tips and
occasional duplicates - none of which exist in LaFAN1.

## What the tool does

1. **Labels the skeleton from topology alone.** Joints are named `bone_0...` in
   the file, so names carry no information. The labeller uses the hierarchy plus
   rest-pose geometry:
   * root = `Hips`;
   * follow the single highest-reaching child upwards: that is the spine chain,
     and the first branch point is the chest (`Spine`/`Spine1`/`Spine2`);
   * at the chest, the child that stays closest to the spine axis is `Neck`
     (its child is `Head`), and the two children reaching furthest sideways are
     the arms;
   * the remaining children of the root are the legs;
   * everything below a `Hand` is a finger.

   Left/right follows the standard glTF convention (Y up, character faces +Z,
   so **+X is the character's left**) - the same convention Make-It-Animatable
   documents in `get_hips_transform`.

2. **Drops every joint that is not in the target skeleton, after moving its
   weights into the nearest surviving ancestor.**

   | dropped | weight goes to |
   |---|---|
   | finger joints | `Hand` |
   | toe tips | `Toe` |
   | head tips / duplicates | `Head` |

   This step matters: deleting the columns *without* re-routing is destructive.
   In `ped1`, 3.1 % of all weight mass sits on finger bones and ~3 % of the
   vertices are >50 % finger-driven (some vertices are 100 % finger-driven).
   A row that sums to 0 makes linear blend skinning collapse that vertex to the
   origin - measured hand vertices would jump ~40 % of the body height.

   This mirrors what Make-It-Animatable does in its own
   `remove_fingers_from_data`, which adds the finger weights onto the `Hand`
   bone before dropping them.

   The transfer is *exact*, not approximate: a dropped joint is a rigid child
   of the surviving ancestor, that is `W_dropped = W_ancestor . L` with `L` a
   constant rest transform, so
   `W_dropped(posed) . IBM_dropped = W_ancestor(posed) . IBM_ancestor`.
   Re-routing the weights therefore reproduces the previous deformation
   exactly, for any pose in which the dropped joint is not animated
   independently - and no LaFAN1 motion animates finger/toe-tip joints.

3. **Renormalizes** the packed rows (they already sum to 1.0 in practice) and
   sorts each vertex's 4 influences by weight.

4. **Rebuilds the node hierarchy** around the 22 retained joints, renaming them
   to the LaFAN1 names and recomputing each local transform so the **world
   transform of every retained joint is preserved**, which keeps the inverse
   bind matrices valid (they are only re-ordered).

The mesh (positions, normals, uvs, indices, materials, textures) is copied
through byte-for-byte; only `JOINTS_0`, `WEIGHTS_0`, the node tree and the skin
are rewritten. The file is streamed in place, so peak RSS stays around 290 MB
for a 1.1 M-vertex / 88 MB GLB and the output is the same size as the input.
The whole 5-asset batch takes ~7 s.

## Target skeleton

`scripts/rig_postprocess/lafan1.py` holds the 22 joints and their parents:

```
Hips
├── Spine ── Spine1 ── Spine2 ── Neck ── Head
│                       ├── LeftShoulder  ── LeftArm  ── LeftForeArm  ── LeftHand
│                       └── RightShoulder ── RightArm ── RightForeArm ── RightHand
├── LeftUpLeg  ── LeftLeg  ── LeftFoot  ── LeftToe
└── RightUpLeg ── RightLeg ── RightFoot ── RightToe
```

Target joints appear in the output in exactly this order, so the output joint
index is a stable identifier (`0 = Hips`, ..., `21 = RightToe`).

> Verify the names against your own BVH before wiring up retargeting: this list
> is the Mixamo/CMU-style set that LaFAN1 uses, but the tool cannot check it
> against a file it has never seen.

## Verification

`--verify` re-opens the produced file and reports:

| check | result on all 5 `Ped_Assets_output` rigs |
|---|---|
| `joint_count` | 22 |
| `names_match_lafan1_order` | True |
| `geometry_accessors_untouched` | True |
| `max_weight_row_sum_error` | ~1.4e-07 (float32 noise, unchanged from the input) |
| `max_joint_position_error` | ~1e-16 |

Batch run over `Ped_Assets_output`:

| file | joints | dropped | weight mass re-routed |
|---|---|---|---|
| ped1 | 34 → 22 | 12 fingers | 4.5 % |
| ped2 | 34 → 22 | 12 fingers | 2.2 % |
| ped3 | 64 → 22 | 38 fingers + 4 head/toe tips | 31.1 % |
| ped4 | 34 → 22 | 12 fingers | 4.6 % |
| ped5 | 34 → 22 | 12 fingers | 1.3 % |

ped3's 31 % is dominated by two coincident head-tip joints that the model gave
27 % of the weight mass to (the asset has long hair); re-routing it onto `Head`
is exactly right and keeps the hair attached to the head.

## Limitations

* **Fingers and toes lose articulation.** After collapsing, those vertices move
  rigidly with the hand / toe. That is inherent to targeting a 22-joint
  skeleton.
* **Coincident joints are reported, not silently fixed.** When the model emits
  a parent and child at (nearly) the same position - e.g. `ped3` has
  `bone_56~bone_57` at the ankle, 0.0007 apart - the tool lists them under
  `coincident_pairs`. Merging them would delete a joint that LaFAN1 expects
  (`LeftToe`), so it is left alone.
* If a target joint is missing from the source rig the tool fails with a clear
  message (`--non-strict` downgrades that to a warning and emits an incomplete
  rig).

## Related: bone connectivity

A separate, upstream bug in `src/rig_package/parser/bpy.py` made exported rigs
look disconnected in Blender (`sons[p].append(i)` was executed twice, so the
`connect_tail_to_unique_child` branch was dead code and every bone got a
uniform extruded tail instead of ending at its child). That is fixed in this
tree; `scripts/connect_bone_tails.py` can repair already-exported files.
