# SkinTokens 本地部署笔记

这台机器（NVIDIA L4 24G，驱动 550 / CUDA 12.4）上跑通 TokenRig 推理的过程、
踩过的坑、以及为批量生产准备的校验手段。原仓库 README 的步骤仍然有效，
本文只补充 README 没写、但实际会卡住的地方。

## 1. 必须的文件

| 路径 | 说明 |
| --- | --- |
| `models/grpo_1400.ckpt` | TokenRig 主模型（含 transformer / mesh_encoder / VAE 权重） |
| `models/last.ckpt` | **官方 FSQ-CVAE（SkinTokens tokenizer）**，缺了跑不了 |
| `models/Qwen3-0.6B/` | 只要 config + tokenizer 文件，权重不用（已打包进 grpo ckpt） |
| `experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt` | 软链到 `models/grpo_1400.ckpt` |
| `experiments/skin_vae_2_10_32768/last.ckpt` | 必须是官方文件（见下节） |

## 2. 最容易踩的坑：不要自己合成 VAE 配置

`grpo_1400.ckpt` 里其实自带了一份完整的 VAE 权重（252 个 `vae.*` 张量），
所以很容易想"直接从它的权重反推配置，再合成一个 `last.ckpt`"，绕过下载。
**这条路会静默产生错误结果，不要走。**

架构层面的参数（层数、宽度、head 数、FSQ 码本）可以从张量形状严格验证，
但下面两个是纯数值开关，形状上看不出来：

```jsonc
"embed_include_pi": true,   // 写成 false -> 权重整体空间错位
"use_pmpe": true            // 写成 false -> 同上
```

官方 `models/last.ckpt` 里的完整 `model_config`（用于核对）：

```json
{
  "model": {
    "in_channels": 4, "cond_channels": 3, "latent_channels": 512,
    "num_attention_heads": 12, "width_encoder": 768, "width_decoder": 768,
    "num_layers_encoder": 2, "num_layers_decoder": 10,
    "embedding_type": "frequency", "embed_frequency": 8,
    "embed_include_pi": true, "is_learned_queries": true, "use_pmpe": true,
    "FSQ_dict": {"levels": [8, 8, 8, 8, 8], "dim": 512}
  },
  "sample": {"cond_tokens": 384, "sample_tokens": 32, "compress_tokens": 32}
}
```

写错这两个开关的症状：解码出来的蒙皮权重在空间上错位——手指骨的影响区域
跑到脚上、末端骨（手腕/手指）完全没有顶点。外观上仍然"能跑通、有骨架、
有顶点组"，所以必须用第 5 节的校验脚本量化检查，肉眼看渲染图看不出来。

判定标准（`scripts/audit_rig.py`，同一份扫描件）：

| 指标 | 配置错误 | 配置正确 |
| --- | --- | --- |
| 主导骨 == 最近骨 | 39.3% | 75.4% |
| 顶点到自身骨距离中位 | 0.074（对角线） | 0.043 |
| 偏移 > 0.20 的骨头 | 4 根 | 0 根 |

## 3. 环境

```
Python 3.11.17 (uv venv)
torch 2.7.0+cu128        -> L4(sm_89) 上 torch.cuda.is_available() == True
transformers 5.19.0, diffusers 0.41.0, bpy 5.0.1
+ scipy                  <- requirements.txt 漏了，代码里 import 了，必须补装
```

安装命令（依赖走清华镜像，本项目实测很快）：

```bash
cd SkinTokens
uv venv --python 3.11 && source .venv/bin/activate
uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

网络：`huggingface.co` 在墙内不可达，走 `hf-mirror.com`
（`HF_ENDPOINT=https://hf-mirror.com`）；只取 `Qwen3-0.6B` 的 config/tokenizer 时，
ModelScope（`Qwen/Qwen3-0.6B`）可以**直连**，不需要代理。
清华 PyPI 镜像可直连；其余（GitHub release、pypi.nvidia.com、hf-mirror.com）
实测都需要代理（本机为 `127.0.0.1:1080`，由 Happy 隧道提供）。

## 4. 没有 flash-attn 也能跑（已打补丁）

代码硬编码了 `attn_implementation="flash_attention_2"`，但 L4 上完整推理
并不需要 flash-attn。改了两处（原文件备份为同目录 `.orig`）：

- `src/model/tokenrig.py`
  - `flash_attn_func` 改为三级回退：FA3 → FA2 → 纯 PyTorch SDPA（签名保持 FA3 约定：
    `(q,k,v)` 进、`(out, None)` 出、布局 `(B,L,H,D)`）
  - Qwen 主干改为 `flash_attention_2 if 可用 else sdpa`
- `src/model/skin_vae_model.py`：同样的 SDPA 回退

说明：`src/model/skin_vae/attention_processor.py` 和 Michelangelo 的 transformer
本来就自带 SDPA 回退（`use_flash3` 只在 H100 上为真），无需改动。
如果之后装了 flash-attn，补丁会自动走回 FA2/FA3，不影响。

## 5. 生成与校验

```bash
# 单人
python demo.py --input models/test.glb --output results/out.glb --use_transfer
# 批量（目录）
python demo.py --input /path/to/meshes/ --output results/ --use_transfer
```

校验（**批量前务必跑一次**）：

```bash
python scripts/audit_rig.py  results/out.glb        # 主导骨==最近骨 / 每根骨的领地偏移
python scripts/render_xray.py results/out.glb results/x.png            # 半透明外壳 + 骨架
python scripts/render_weights.py results/out.glb results/w.png         # 按主导骨上色
```

`audit_rig.py` 的三条判据：主导骨与最近骨一致率应 >70%；顶点到自身骨距离中位
应 <0.06；不应有骨头的影响区域偏移 >0.20。

## 6. 导出到 Blender 的两个已知问题

**骨尾不连**：`BpyParser.export` 里骨尾 = 关节位置 + 模型预测的长度/朝向，
不指向子关节，所以长肢段看起来"断了一截"。它是显示属性，不参与蒙皮计算
（glTF 里只有骨头静止矩阵和权重决定变形）。两种修法：

```bash
# A. 后处理已有文件（不必重跑模型）
python scripts/connect_bone_tails.py results/out.glb results/out_connected.glb
# B. 生成时直接用官方选项：在 demo.py 的 export / transfer payload 里加
#    connect_tail_to_unique_child=True
```

已确认修完后蒙皮权重不变（按坐标比对 99.98% 一致）。

**`glTF_not_exported`**：Blender 导入 glTF 会给骨骼生成一个自定义显示形状
（名为 `Icosphere`，放在 `glTF_not_exported` 集合）。它不在文件里，删掉即可，
README 的 Troubleshooting 也提到过。

## 7. 批量生产的注意事项

- **骨架是采样生成的**（`num_beams=10` + `do_sample=True`），不同输入得到的骨骼
  **数量会变**（实测同一份网格出现过 34 根和 73 根）。因此：
  - 不要硬编码 `bone_13 = 右手` 这类索引映射；
  - 按层级/位置识别关节（根 → 脊柱 → 四肢），或固定随机种子让结果可复现；
  - 下游脚本要能处理可变骨骼数。
- 各关节**并不是都占一段权重**：末端小骨（指尖、脚趾）领地面积极小，
  某些采样下甚至为 0（本次样例里 `bone_12` 就是）。短促地转它看不出变化是正常的。
- 用 Blender 的自动权重重新绑定**不可行**：这份 119 万顶点的实拍扫描件
  自身网格交叠，`Bone Heat Weighting` 会直接失败（实测 34 根全部无效）。
  批量生产必须使用模型推理出的权重。
- 动画建议：脊柱/头/肩/大臂小臂/大腿/膝/踝这些大关节可靠；
  指尖这类小骨避开。转动前先看一眼 `audit_rig.py` 的输出。

## 8. 辅助脚本一览

| 脚本 | 用途 |
| --- | --- |
| `scripts/audit_rig.py` | 骨骼质量体检：主导骨/最近骨一致率、每根骨的领地偏移 |
| `scripts/connect_bone_tails.py` | 把骨尾吸附到唯一子关节，修复"断骨"观感 |
| `scripts/render_xray.py` | 半透明外壳 + 骨架（去纹理，最直观） |
| `scripts/render_weights.py` | 按主导骨上色，检查权重分区是否干净 |
| `scripts/render_pose.py` | 摆姿势渲染，验证变形是否局部 |
| `scripts/render_skeleton.py` / `render_preview.py` | 单独渲染骨架 / 普通预览 |
| `scripts/debug_decode.py` | 跑真实流程并打印生成的 token、skin_pred 统计 |
| `scripts/sweep_vae_config.py` / `sweep_vae_config_real.py` | 扫描 VAE 配置（本次定位坑用的，已不再需要） |
| `scripts/build_vae_ckpt.py` | **已废弃**：合成 VAE ckpt（第 2 节的原因，不要用） |

## 9. 回退

`src/model/tokenrig.py.orig`、`src/model/skin_vae_model.py.orig` 是打补丁前的原文件：

```bash
cp src/model/tokenrig.py.orig src/model/tokenrig.py
cp src/model/skin_vae_model.py.orig src/model/skin_vae_model.py
```

`experiments/skin_vae_2_10_32768/last.ckpt.synthetic` 是我早先合成的错误文件，
保留仅作对照，**不要使用**。
