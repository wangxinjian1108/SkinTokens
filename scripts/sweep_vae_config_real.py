"""Sweep VAE configs using the REAL conditioning dumped from the pipeline.

/tmp/skintokens_debug_inputs.pt is produced by scripts/debug_decode.py and holds
the exact `cond`, `cond_latents` and generated token ids of a real run.

Criterion: with real conditioning, different skin tokens (i.e. different joints)
must decode to clearly different skin fields.
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.model.skin_vae_model import SkinVAEModel  # noqa: E402

DEBUG_PT = "/tmp/skintokens_debug_inputs.pt"

BASE = {
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


def build(over, weights, levels=(8, 8, 8, 8, 8)):
    cfg = {
        "type": "fsqc",
        "sample": {"cond_tokens": 384, "compress_tokens": 32, "sample_tokens": 32},
        "model": {
            **{**BASE, **over},
            "FSQ_dict": {"levels": list(levels), "dim": 512},
        },
    }
    vae = SkinVAEModel(model_config=cfg, transform_config={})
    missing, unexpected = vae.load_state_dict(weights, strict=False)
    if missing or unexpected:
        return None
    return vae.cuda().eval()


def main():
    dbg = torch.load(DEBUG_PT, weights_only=False)
    ids = dbg["inputs_ids"].to("cuda")
    cond = dbg["cond"].cuda()                    # (N, 6)
    if cond.shape[0] > 1024:
        step = cond.shape[0] // 1024
        cond = cond[::step][:1024]
    cond_lat = dbg["cond_latents"].cuda()        # (384, 512)
    vocab = dbg["tokenizer_vocab_size"]
    eos = dbg["eos"]
    split = int(torch.where(ids == eos)[0][0]) + 1
    skin = ids[split:]
    J = skin.shape[0] // 4
    idx = (skin[: J * 4] - vocab).reshape(J, 4)
    idx = idx.clamp(0, 32767)                    # the model can emit EOS inside the skin block
    print(f"real run: J={J} joints | cond {tuple(cond.shape)} | cond_latents {tuple(cond_lat.shape)}")
    print("cond_latents stats: std=%.4f absmax=%.3f" %
          (cond_lat.float().std(), cond_lat.float().abs().max()))

    ckpt = torch.load(REPO / "models" / "grpo_1400.ckpt", map_location="cpu", weights_only=False)
    weights = {k[len("vae."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("vae.")}

    H = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    cands = [(f"heads{H} baseline", {"num_attention_heads": H})]
    cands += [(f"pmpe={v}", {"num_attention_heads": H, "use_pmpe": v}) for v in (False, True)]
    cands += [(f"pi={v}", {"num_attention_heads": H, "embed_include_pi": v}) for v in (False, True)]
    cands += [(f"miche={v}", {"num_attention_heads": H, "is_miche": v}) for v in (False, True)]
    cands += [(f"levels{l}", {"num_attention_heads": H, "__levels__": l}) for l in
              ([8, 8, 8, 8, 8], [4, 8, 8, 8, 16], [2, 8, 8, 16, 16], [2, 4, 8, 16, 32],
               [16, 16, 16, 8, 1], [32, 16, 8, 4, 2], [8, 8, 8, 16, 4])]

    print(f"\n{'config':<18}{'joints-std':>12}{'spread':>10}")
    best = []
    for name, over in cands:
        levels = over.pop("__levels__", (8, 8, 8, 8, 8))
        vae = build(over, weights, levels)
        if vae is None:
            print(f"{name:<18}{'LOAD FAILED':>22}")
            continue
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = vae.model.FSQ.indices_to_codes(idx).reshape(J, 4, -1)
            out = vae.decode(
                z=z,
                sampled_cond=cond.unsqueeze(0).repeat(J, 1, 1),
                cond_tokens=cond_lat.unsqueeze(0).repeat(J, 1, 1),
            ).float().squeeze(-1)
        across = out.std(dim=0).mean().item()
        spread = (out.max(dim=0).values - out.min(dim=0).values).mean().item()
        best.append((across, name))
        print(f"{name:<18}{across:>12.5f}{spread:>10.4f}")
        del vae
        torch.cuda.empty_cache()

    best.sort(reverse=True)
    print("\ntop: ", [(n, round(v, 5)) for v, n in best[:4]])


if __name__ == "__main__":
    main()
