# T2VA Example · Open-H3-Regenerate-2K-IC

English | [简体中文](README.zh-CN.md)

This is the full **text-to-video** example workflow:

[`regen2k_ic_T2VA.json`](../../workflows/regen2k_ic_T2VA.json)

API payload: [`regen2k_ic_T2VA_api.json`](../../workflows/regen2k_ic_T2VA_api.json)

It generates a 768P MiniMax-H3 draft from text, then automatically refines the
same clip to 2K with Open-H3-Regenerate-2K-IC.

## Pipeline

```text
T2VA prompt
  -> MiniMax-H3 Image-to-Video
  -> DMAD 4-step LoRA
  -> 1344×768 / 124 frames / 24 fps
  -> video decode + audio decode
  -> draft becomes <Video 1> in Ref2VA
  -> 3D latent upscaler
  -> HyperFlow + LMS Sharp + Open-H3 IC LoRA
  -> up3 refinement
  -> 2560×1440 final video
```

The workflow uses one source node:

**T2VA prompt · edit this text**

It feeds:

1. the raw prompt into the H3 draft generator
2. the same prompt into the IC enhancement prompt wrapper

This avoids applying `Enhance <Video 1>` to the first text-to-video stage.

## Required custom nodes

- [ComfyUI-OpenH3Regenerate2KIC](https://github.com/slmonker/ComfyUI-OpenH3Regenerate2KIC)
- `ComfyUI-LynnReal`
- `ComfyUI-Spark-H3`
- `ComfyUI-VideoHelperSuite`

The workflow uses these specific nodes:

- `OpenH3ICDraft`
- `OpenH3ICPrompt`
- `OpenH3ICSigmas`
- `OpenH3ICComposeLatent`
- `MiniMaxH3ImageToVideo`
- `MiniMaxH3ReferenceToVideo`
- `ModelAttentionBackend`
- `MiniMaxChunkFeedForward`
- `BlockSparseAttention`
- `VHS_VideoCombine`

## Required models

### Draft generation

```text
models/diffusion_models/
└── minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors/
    └── minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors

models/loras/
└── dmad_minimax_h3_4step_full_critic_comfy.safetensors
```

### IC refinement

```text
models/latent_upscale_models/
└── minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors

models/loras/
├── minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors
├── minimax_h3_lms_v1.0_r64.safetensors
└── minimax_h3_open_regenerate_2k_ic_comfy_bf16.safetensors
```

Download the converted IC LoRA from:

<https://huggingface.co/slmonker/Open-H3-Regenerate-2K-IC-ComfyUI>

### Shared components

```text
models/text_encoders/MiniMax-H3/
└── qwen3vl_32b_minimax_h3_int8_convrot.safetensors

models/vae/MiniMax-H3/
├── minimax_h3_video_vae_fp16.safetensors
└── minimax_h3_audio_vae_fp32.safetensors
```

The hybrid FL2VA+Ref2VA checkpoint and DMAD LoRA come from the MiniMax-H3 /
TaoMate-HD ecosystem. The hybrid checkpoint is not included in this repository.

## Run

1. Open `../../workflows/regen2k_ic_T2VA.json` in ComfyUI.
2. Edit **T2VA prompt · edit this text**.
3. Run the graph.
4. Outputs are written to:

```text
ComfyUI/output/OpenH3/T2VA_draft_768p_*
ComfyUI/output/OpenH3/T2VA_final_2k_*
```

## Default settings

| Stage | Setting |
|---|---|
| Draft resolution | 1344×768 |
| Draft frames | 124 |
| FPS | 24 |
| Draft sampler | Euler, 4 steps |
| Draft CFG | 1 |
| IC sampler | `up3` |
| Final resolution | 2560×1440 |

## Example prompt

```text
A cinematic medium shot of a young woman in an elegant ancient Chinese dress
standing in a sunlit bamboo forest. She looks toward the camera and says in
Mandarin: "你好，欢迎来到这片竹林。"
Gentle camera push-in, soft breeze, rustling leaves, detailed fabric and hair.

overall_soundscape: Soft wind through bamboo leaves, distant birds, and a clear natural voice.

non_diegetic_music: A restrained guzheng melody with a warm, cinematic atmosphere.
```

## Notes

- The workflow contains two output nodes: a 768P draft preview and the final 2K video.
- `BlockSparseAttention` uses the `sol-attn` selection. If it is unavailable, bypass it or use the native attention backend.
- 2K refinement is the memory-intensive stage. On 32 GB GPUs, use the INT8 hybrid checkpoint and close other GPU applications.
- This is a community workflow, not an official MiniMax ComfyUI template.
