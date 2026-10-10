# scripts

Helper tools around the TokenRig / SkinTokens pipeline. Run them from the repo
root with the project virtualenv (`python scripts/<name>.py ...`).

## Rig post-processing

| script | what it does |
|---|---|
| `rig_to_lafan1.py` | Collapse any TokenRig rig onto the fixed **22-joint LaFAN1 skeleton** (drops fingers/toe tips, re-routes their weights, renames and rebuilds the hierarchy). See [docs/rig_to_lafan1.md](../docs/rig_to_lafan1.md). |
| `connect_bone_tails.py` | Repair an already-exported GLB whose bone tails do not meet their child's head (only the tail is touched; heads, hierarchy and weights are preserved). |
| `audit_rig.py` | Report how well each bone's influence matches its location, plus a bone -> owned-region table. |

## Visual inspection

| script | what it does |
|---|---|
| `render_preview.py` | Quick EEVEE preview of mesh + armature. |
| `render_skeleton.py` | Render the skeleton as real geometry (joint spheres + bone rods). |
| `render_xray.py` | X-ray render: semi-transparent shell + faint wireframe, so the generated skeleton reads clearly. |
| `render_weights.py` | Colour the mesh by its dominant bone to inspect skinning quality. |
| `render_pose.py` | Render with one bone posed - the mesh geometry is identical to the input, so this is what shows the rig actually deforms it. |

## VAE / debugging

| script | what it does |
|---|---|
| `build_vae_ckpt.py` | Rebuild `experiments/skin_vae_2_10_32768/last.ckpt` from the FSQ-CVAE weights embedded in the TokenRig checkpoint (the separate VAE checkpoint is not always available). |
| `sweep_vae_config.py`, `sweep_vae_config_real.py` | Validate the reconstructed FSQ-CVAE: different skin tokens must decode to clearly different skin fields. |
| `debug_decode.py` | Run the demo pipeline while dumping the generated skin tokens for inspection. |

## Notes

* `rig_to_lafan1.py` and `connect_bone_tails.py` are the two tools that change
  what a rig *is*; everything else is read-only diagnostics.
* The exporter in `src/rig_package/parser/bpy.py` already emits connected bones
  (see the commit "Fix bone connectivity on export"); `connect_bone_tails.py`
  is only needed for files produced before that fix.
