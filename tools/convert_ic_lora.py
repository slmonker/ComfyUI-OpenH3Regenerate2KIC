"""Convert the official Open-H3-Regenerate-2K-IC Diffusers LoRA to ComfyUI layout.

Usage:
  python tools/convert_ic_lora.py SOURCE.safetensors OUTPUT.safetensors [--dtype bf16|fp16|fp32]

The source contains 312 Diffusers/PEFT linear modules on MiniMax-H3
`transformer_ref`:
  token_refiner.refiner_blocks.N.{attn.to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2}
  transformer_blocks.N.{same}

ComfyUI fuses q/k/v into qkv_proj and orders the SwiGLU fc1 rows as [gate;value].
The conversion is exact before the optional output dtype cast.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _read_pairs(path: Path):
    out = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.endswith(".lora_A.weight"):
                base = key[:-len(".lora_A.weight")]
                bkey = base + ".lora_B.weight"
                if bkey not in f.keys():
                    raise ValueError(f"missing {bkey}")
                out[base] = (f.get_tensor(key), f.get_tensor(bkey))
    return out


def _source_to_target_prefix(prefix: str) -> str:
    if prefix.startswith("transformer.token_refiner.refiner_blocks."):
        return prefix.replace("transformer.token_refiner.refiner_blocks.", "token_refiner.blocks.", 1)
    if prefix.startswith("transformer.transformer_blocks."):
        return prefix.replace("transformer.transformer_blocks.", "blocks.", 1)
    raise ValueError(f"unexpected source prefix: {prefix}")


def _module(pairs, prefix, module):
    key = f"transformer.{prefix}.{module}"
    if key not in pairs:
        raise KeyError(f"missing LoRA module {key}")
    a, b = pairs[key]
    return a.float(), b.float()


def _block_diag_qkv(aq, bq, ak, bk, av, bv):
    rank = aq.shape[0]
    if ak.shape[0] != rank or av.shape[0] != rank:
        raise ValueError("q/k/v ranks differ")
    out = bq.shape[0]
    if bk.shape[0] != out or bv.shape[0] != out:
        raise ValueError("q/k/v output widths differ")
    a = torch.cat((aq, ak, av), dim=0)
    b = torch.zeros((3 * out, 3 * rank), dtype=torch.float32)
    b[0:out, 0:rank] = bq
    b[out:2 * out, rank:2 * rank] = bk
    b[2 * out:3 * out, 2 * rank:3 * rank] = bv
    return a, b


def convert(src_path: Path, out_path: Path, dtype_name: str):
    pairs = _read_pairs(src_path)
    if len(pairs) != 312:
        raise ValueError(f"expected 312 LoRA modules, found {len(pairs)}")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    result = {}
    modules = 0

    def put(target_prefix, module, a, b, alpha):
        result[f"diffusion_model.{target_prefix}.{module}.alpha"] = torch.tensor(alpha, dtype=torch.float32)
        result[f"diffusion_model.{target_prefix}.{module}.lora_A.weight"] = a.to(dtype).contiguous()
        result[f"diffusion_model.{target_prefix}.{module}.lora_B.weight"] = b.to(dtype).contiguous()

    prefixes = []
    for i in range(2):
        prefixes.append((f"token_refiner.refiner_blocks.{i}", f"token_refiner.blocks.{i}"))
    for i in range(50):
        prefixes.append((f"transformer_blocks.{i}", f"blocks.{i}"))

    for src_prefix, dst_prefix in prefixes:
        qa, qb = _module(pairs, src_prefix, "attn.to_q")
        ka, kb = _module(pairs, src_prefix, "attn.to_k")
        va, vb = _module(pairs, src_prefix, "attn.to_v")
        a, b = _block_diag_qkv(qa, qb, ka, kb, va, vb)
        put(dst_prefix, "attn.qkv_proj", a, b, a.shape[0])

        a, b = _module(pairs, src_prefix, "attn.to_out.0")
        put(dst_prefix, "attn.out_proj", a, b, a.shape[0])

        a, b = _module(pairs, src_prefix, "ff.net.0.proj")
        if b.shape[0] % 2:
            raise ValueError("fc1 LoRA output is not a SwiGLU pair")
        b = torch.cat((b[b.shape[0] // 2:], b[:b.shape[0] // 2]), dim=0)
        put(dst_prefix, "mlp.fc1", a, b, a.shape[0])

        a, b = _module(pairs, src_prefix, "ff.net.2")
        put(dst_prefix, "mlp.fc2", a, b, a.shape[0])
        modules += 6

    if modules != 312 or len(result) != 208 * 3:
        raise RuntimeError(f"converted {modules} source linears into {len(result)} ComfyUI tensors")

    metadata = {
        "format": "pt",
        "target_format": "ComfyUI generic LoRA (MiniMaxH3 transformer_ref)",
        "source": str(src_path.name),
        "source_components": "open_h3_regenerate_2k_ic_lora.safetensors",
        "source_scale": "alpha == rank == 384, scale 1",
        "conversion": "qkv block-diag fused (q|k|v), fc1 SwiGLU [value;gate] -> [gate;value], alpha=r*rank",
        "dtype": dtype_name,
        "source_linears": "312",
        "comfy_modules": "208",
        "base_model": "MiniMaxAI/MiniMax-H3 transformer_ref",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(result, str(out_path), metadata=metadata)
    report = {
        "source": str(src_path),
        "output": str(out_path),
        "dtype": dtype_name,
        "modules": modules,
        "source_linears": modules,
        "comfy_modules": len(result) // 3,
        "tensors": len(result),
        "size_gib": round(out_path.stat().st_size / (1024 ** 3), 3),
    }
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = ap.parse_args()
    convert(args.source, args.output, args.dtype)


if __name__ == "__main__":
    main()
