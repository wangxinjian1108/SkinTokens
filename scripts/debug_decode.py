"""Run the real demo pipeline while inspecting the generated skin tokens.

Usage: python scripts/debug_decode.py <input.glb> <output.glb> [num_beams]
"""

import os
import runpy
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import src.model.tokenrig as T  # noqa: E402

_orig_decode = T.decode


def _decode_verbose(*args, **kwargs):
    ids = kwargs.get("inputs_ids")
    tokenizer = kwargs.get("tokenizer")
    cond = kwargs.get("cond")
    cond_latents = kwargs.get("cond_latents")
    if cond is not None and cond_latents is not None:
        torch.save(
            {
                "inputs_ids": kwargs["inputs_ids"].detach().cpu(),
                "cond": cond.detach().cpu(),
                "cond_latents": cond_latents.detach().cpu(),
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "eos": tokenizer.eos,
            },
            "/tmp/skintokens_debug_inputs.pt",
        )
        print("[DEBUG] saved cond/cond_latents/tokens to /tmp/skintokens_debug_inputs.pt")
    if ids is not None and tokenizer is not None:
        ids = ids.detach()
        eos_pos = torch.where(ids == tokenizer.eos)[0]
        if eos_pos.numel():
            split = int(eos_pos[0]) + 1
            skel = ids[:split]
            skin = ids[split:]
            print("\n[DEBUG] sequence length:", ids.shape[0])
            print("[DEBUG] skeleton tokens:", skel.tolist()[:40])
            print("[DEBUG] num skin tokens:", skin.shape[0])
            if skin.numel():
                off = skin - tokenizer.vocab_size
                print("[DEBUG] skin token index range:",
                      int(off.min()), "-", int(off.max()),
                      "| unique:", int(off.unique().numel()))
                print("[DEBUG] first 24 skin indices:", off[:24].tolist())

    res = _orig_decode(*args, **kwargs)
    pred = res.get("skin_pred") if isinstance(res, dict) else None
    if pred is not None:
        p = pred.float()
        print("[DEBUG] skin_pred shape:", tuple(p.shape))
        print("[DEBUG] skin_pred stats: min=%.4f max=%.4f mean=%.4f" %
              (p.min(), p.max(), p.mean()))
        print("[DEBUG] std across joints (per point) mean: %.5f" % p.std(dim=1).mean())
        print("[DEBUG] frac >0.9: %.4f | frac <0.1: %.4f" %
              ((p > 0.9).float().mean(), (p < 0.1).float().mean()))
        top = torch.topk(p[0], k=min(4, p.shape[1])).values
        print("[DEBUG] point0 top4 values:", [round(v, 5) for v in top.tolist()])
    return res


T.decode = _decode_verbose

argv = sys.argv[1:]
inp, outp = argv[0], argv[1]
beams = argv[2] if len(argv) > 2 else "1"

sys.argv = [
    "demo.py",
    "--input", inp,
    "--output", outp,
    "--num_beams", beams,
]
runpy.run_path(str(REPO / "demo.py"), run_name="__main__")
