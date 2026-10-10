"""Find the VAE config whose decoder is actually sensitive to the skin tokens.

A correctly reconstructed FSQ-CVAE must map different skin tokens to clearly
different skin fields. We rebuild the VAE under candidate configs, load the
weights taken from the TokenRig checkpoint, and measure how much the decoded
output changes when only `z` (the skin tokens) changes.

The architecture is already pinned by shape (252/252 keys load strictly); only
value-level knobs that do not affect tensor shapes are swept here.
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.model.skin_vae_model import SkinVAEModel  # noqa: E402

BASE_MODEL = {
    "in_channels": 4,
    "cond_channels": 3,
    "latent_channels": 512,
    "num_attention_heads": 8,
    "width_encoder": 768,
    "width_decoder": 768,
    "num_layers_encoder": 2,
    "num_layers_decoder": 10,
    "embedding_type": "frequency",
    "embed_frequency": 8,
    "embed_include_pi": False,
    "is_learned_queries": True,
    "is_miche": False,
}

CANDIDATES = [
    ("baseline", {}),
    ("use_pmpe", {"use_pmpe": True}),
    ("include_pi", {"embed_include_pi": True}),
    ("is_miche", {"is_miche": True}),
    ("heads6", {"num_attention_heads": 6}),
    ("heads12", {"num_attention_heads": 12}),
    ("heads16", {"num_attention_heads": 16}),
    ("heads4", {"num_attention_heads": 4}),
]

LEVELS = [
    [8, 8, 8, 8, 8],
    [4, 8, 8, 8, 16],
    [2, 8, 8, 16, 16],
    [4, 4, 8, 16, 16],
    [2, 4, 8, 16, 32],
]


def build(model_cfg, levels, weights):
    cfg = {
        "type": "fsqc",
        "sample": {"cond_tokens": 384, "compress_tokens": 32, "sample_tokens": 32},
        "model": {**model_cfg, "FSQ_dict": {"levels": levels, "dim": 512}},
    }
    vae = SkinVAEModel(model_config=cfg, transform_config={})
    missing, unexpected = vae.load_state_dict(weights, strict=False)
    if missing or unexpected:
        return None
    return vae


def sensitivity(vae, idx, cond, cond_lat):
    vae = vae.cuda().eval()
    J = idx.shape[0]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = vae.model.FSQ.indices_to_codes(idx).reshape(J, 4, -1)
        out = vae.decode(
            z=z,
            sampled_cond=cond.repeat(J, 1, 1),
            cond_tokens=cond_lat.repeat(J, 1, 1),
        ).float().squeeze(-1)
    across = out.std(dim=0).mean().item()
    return across, z.float().std().item()


def main():
    ckpt = torch.load(REPO / "models" / "grpo_1400.ckpt", map_location="cpu", weights_only=False)
    weights = {k[len("vae."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("vae.")}

    torch.manual_seed(0)
    J = 16
    idx = torch.randint(0, 32768, (J, 4), device="cuda")
    cond = torch.randn(1, 256, 6, device="cuda")
    cond_lat = torch.randn(1, 384, 512, device="cuda")

    print(f"{'config':<22}{'levels':<22}{'z std':>9}{'across-joint std':>20}")
    results = []
    for name, over in CANDIDATES:
        for levels in (LEVELS if name == "baseline" else [LEVELS[0]]):
            vae = build({**BASE_MODEL, **over}, levels, weights)
            if vae is None:
                print(f"{name:<22}{str(levels):<22}{'LOAD FAILED':>29}")
                continue
            across, zstd = sensitivity(vae, idx, cond, cond_lat)
            results.append((across, name, levels, zstd))
            print(f"{name:<22}{str(levels):<22}{zstd:>9.4f}{across:>20.5f}")
            del vae
            torch.cuda.empty_cache()

    results.sort(reverse=True)
    print("\nbest by z-sensitivity:")
    for across, name, levels, zstd in results[:3]:
        print(f"  {name:<20} levels={levels} -> {across:.5f}")


if __name__ == "__main__":
    main()
