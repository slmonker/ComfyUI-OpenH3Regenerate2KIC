# ComfyUI Open-H3-Regenerate-2K-IC

ComfyUI nodes and a ready-to-run workflow for
[NVlabs/Sana · Open-H3-Regenerate-2K/IC](https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K/IC).

The workflow takes a low-resolution MiniMax-H3 draft, keeps it as `<Video 1>` in
Ref2VA context, upscales its latent with the official 3D latent upscaler, and
regenerates a 2560×1440 clip with the official `up3` sigma grid.

- GitHub: <https://github.com/slmonker/ComfyUI-OpenH3Regenerate2KIC>
- Converted ComfyUI LoRA: <https://huggingface.co/slmonker/Open-H3-Regenerate-2K-IC-ComfyUI>
- Upstream weights: <https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC>

```text
768P draft
  -> <Video 1> Ref2VA conditioning
  -> H3 VAE encode -> official 3D latent upscaler
  -> HyperFlow + LMS Sharp + IC LoRA
  -> up3 / up4 / hf8 sigma grid
  -> fast H3 VAE decode
  -> mux the draft's original audio
```

## Contents

| Path | Purpose |
|---|---|
| `nodes.py` | Four ComfyUI helper nodes |
| `workflows/open_h3_regenerate_2k_ic.json` | Main 2560×1440 workflow |
| `workflows/open_h3_regenerate_2k_ic_api.json` | API prompt payload |
| `tools/convert_ic_lora.py` | Diffusers IC LoRA → ComfyUI generic LoRA |
| `tools/build_workflow.py` | Rebuild the workflow JSON |
| `assets/silence_audio_124f.safetensors` | Official fixed-silence latent for the IC audio rows |
| `tests/test_nodes.py` | Unit tests for preprocessing and sigma grids |

## Nodes

### Open-H3 IC · Prepare Draft (124f / 24fps)

Matches the official draft preprocessing:

- samples 124 frames at 24 fps from the source video
- cover-resizes and center-crops to 1344×768
- trims or pads the original audio using its real sample rate
- requires at least 113 real frames after resampling

### Open-H3 IC · Official Prompt

Prepends the official task instruction and preserves the original MiniMax-H3
caption as `Scene description`.

### Open-H3 IC · Sigma Grid

Provides the official grids:

| Mode | Sigmas |
|---|---|
| `up3` | `0.834942, 0.696852, 0.468753, 0` |
| `up4` | `0.923077, 0.834942, 0.696852, 0.468753, 0` |
| `hf8` | `1.0, 0.99391, 0.984288, 0.966064, 0.923077, 0.834942, 0.696852, 0.468753, 0` |

### Open-H3 IC · Compose Target Latent

Combines the upscaled target video latent with the official 124-frame silence
latent. The audio stream receives `noise_mask=0`, so its clean silence rows are
preserved during sampling, matching the official IC sampler behavior.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/slmonker/ComfyUI-OpenH3Regenerate2KIC.git
```

Restart ComfyUI after installation.

The workflow also uses `LynnRealH3VAELoader` from `ComfyUI-LynnReal` for the
fast H3 VAE decoder. If you do not want that dependency, replace the fast VAE loader and
decode with the regular MiniMax-H3 video VAE.

## Required models

Place these under `ComfyUI/models/`:

```text
models/
├── diffusion_models/
│   └── minimax_h3_ref2va_bf16.safetensors
├── text_encoders/
│   └── MiniMax-H3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
├── vae/
│   └── MiniMax-H3/
│       ├── minimax_h3_video_vae_fp16.safetensors
│       └── sol_h3_vae.safetensors
├── latent_upscale_models/
│   └── minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors
└── loras/
    ├── minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors
    ├── minimax_h3_lms_v1.0_r64.safetensors
    └── minimax_h3_open_regenerate_2k_ic_comfy_bf16.safetensors
```

Download the converted IC LoRA from:

<https://huggingface.co/slmonker/Open-H3-Regenerate-2K-IC-ComfyUI>

Load the LoRAs in this exact order:

1. HyperFlow 8-step
2. LMS Sharp
3. Open-H3 Regenerate 2K IC

The other MiniMax-H3 components are available from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3).

## Workflow

1. Put the 768P MiniMax-H3 draft into `ComfyUI/input/`.
2. Open `workflows/open_h3_regenerate_2k_ic.json`.
3. Select the draft in **Load Video**.
4. Enter the original MiniMax-H3 prompt in **Open-H3 IC · Official Prompt**. An empty caption is valid.
5. Run the graph. The output is saved through **Save Video** with the draft's original audio.

The default is `up3`, CFG 1, 2560×1440, 124 frames at 24 fps.

### 1080P variant

Set both resolution-bearing nodes to `1920×1088`:

- **MiniMax H3 Reference to Video**
- **MiniMax H3 Latent Upscale With Model**

The target latent is composed from the Ref2VA latent shape, so it follows
automatically.

For 4K use `3840×2176`; this is generally impractical on a 32 GB GPU.

## Convert the official IC LoRA

The official file uses Diffusers/PEFT keys on `transformer_ref`:

```text
transformer.<module>.lora_A.weight
transformer.<module>.lora_B.weight
```

ComfyUI expects fused MiniMax-H3 keys. The converter performs:

- q/k/v → `qkv_proj` with a block-diagonal B matrix
- `ff.net.0.proj` SwiGLU rows from Diffusers `[value;gate]` to ComfyUI `[gate;value]`
- `alpha = 3 × rank` for fused qkv; `alpha = rank` otherwise
- optional BF16 output

```bash
python tools/convert_ic_lora.py \
  open_h3_regenerate_2k_ic_lora.safetensors \
  minimax_h3_open_regenerate_2k_ic_comfy_bf16.safetensors \
  --dtype bf16
```

The released converted file has:

- 312 source linear layers
- 208 fused ComfyUI modules
- 624 tensors (`alpha`, `lora_A`, `lora_B`)
- zero missing base-model key mappings

## Validation

```bash
python -m unittest discover -s tests -v
```

The included workflow was loaded by the ComfyUI frontend and converted with
`graphToPrompt()` without errors:

- 26 graph nodes
- 30 links
- 25 API prompt nodes
- zero API input-schema errors

The converted LoRA was also checked against
`minimax_h3_ref2va_bf16.safetensors`: every target base weight exists.

## Notes and limitations

- Default clip length is 124 frames (about 5.17 seconds), matching the official IC examples.
- The draft's audio is preserved; H3's generated audio is not used.
- 2K on 32 GB VRAM requires ComfyUI model offload/low-VRAM attention handling. Use the pruned INT8 Ref2VA model and 1080P if memory is tight.
- This is a community ComfyUI implementation, not an official MiniMax product.

## Acknowledgements

- [Open-H3-Regenerate-2K-IC](https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K/IC)
- [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [HyperFlow](https://github.com/Video-Rebirth/hyperflow)
- [H3 latent upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)

## License

The node code is Apache-2.0. MiniMax-H3 weights and derivative assets remain
subject to the MiniMax-H3 Community License. See `NOTICE.md`.
