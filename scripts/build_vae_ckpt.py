"""Synthesize experiments/skin_vae_2_10_32768/last.ckpt from the VAE weights
embedded in the TokenRig checkpoint.

Why: the released SkinTokens repo ships the FSQ-CVAE as a separate Lightning
checkpoint (experiments/skin_vae_2_10_32768/last.ckpt). That file was not
available locally and the network proxy was down, but the TokenRig checkpoint
(grpo_1400.ckpt) already contains the full 252-key VAE state_dict. The VAE
architecture is reconstructed from those tensor shapes and validated by a
strict (missing=0, unexpected=0) state_dict load; the FSQ codebook size is
cross-checked against the TokenRig transformer embedding rows
(267 tokenizer + 32768 FSQ + 1 EOS = 33036).

Keys are stored with an extra "model." prefix because
ModelSpec.load_from_system_checkpoint strips one leading "model." from every key.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
TOKENRIG_CKPT = REPO / "models" / "grpo_1400.ckpt"
OUT_PATH = REPO / "experiments" / "skin_vae_2_10_32768" / "last.ckpt"

MODEL_CONFIG = {
    "type": "fsqc",
    "sample": {"cond_tokens": 384, "compress_tokens": 32, "sample_tokens": 32},
    "model": {
        "in_channels": 4,
        "cond_channels": 3,
        "latent_channels": 512,
        # 12, not the class default of 8: recovered by requiring the trained
        # decoder to actually respond to the skin tokens. With 8 the decoded
        # skin field is all but independent of the tokens (joint-to-joint std
        # ~1.5e-3); with 12 it is ~1.1e-1 and the weights come out sparse.
        "num_attention_heads": 12,
        "width_encoder": 768,
        "width_decoder": 768,
        "num_layers_encoder": 2,
        "num_layers_decoder": 10,
        "embedding_type": "frequency",
        "embed_frequency": 8,
        "embed_include_pi": False,
        "is_learned_queries": True,
        "is_miche": False,
        "FSQ_dict": {"levels": [8, 8, 8, 8, 8], "dim": 512},
    },
}


def main() -> None:
    ckpt = torch.load(TOKENRIG_CKPT, map_location="cpu", weights_only=False)
    state_dict = {
        "model." + k[len("vae.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("vae.")
    }
    print(f"collected {len(state_dict)} VAE tensors from {TOKENRIG_CKPT.name}")

    # Round-trip through the real loader to guarantee the file is usable.
    from src.model.skin_vae_model import SkinVAEModel

    probe = SkinVAEModel(model_config=MODEL_CONFIG, transform_config={})
    missing, unexpected = probe.load_state_dict(
        {k[len("model.") :]: v for k, v in state_dict.items()}, strict=False
    )
    if missing or unexpected:
        raise SystemExit(f"architecture mismatch: missing={missing} unexpected={unexpected}")
    print(f"validated architecture: {len(probe.state_dict())} keys, vocab_size={probe.vocab_size}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state_dict,
            "hyper_parameters": {
                "model_config": MODEL_CONFIG,
                "transform_config": {},
                "tokenizer_config": None,
            },
        },
        OUT_PATH,
    )
    print(f"wrote {OUT_PATH}")

    reloaded = SkinVAEModel.load_from_system_checkpoint(str(OUT_PATH))
    print(
        f"reload OK: vocab_size={reloaded.vocab_size} "
        f"latent_channels={reloaded.latent_channels}"
    )


if __name__ == "__main__":
    main()
