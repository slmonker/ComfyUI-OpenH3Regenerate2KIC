# T2VA 示例 · Open-H3-Regenerate-2K-IC

[English](README.md) | 简体中文

这是完整的**文生视频**示例工作流：

[`regen2k_ic_T2VA.json`](../../workflows/regen2k_ic_T2VA.json)

API 工作流：[`regen2k_ic_T2VA_api.json`](../../workflows/regen2k_ic_T2VA_api.json)

它会先用文本生成 768P MiniMax-H3 草稿，再自动使用
Open-H3-Regenerate-2K-IC 将同一视频精修到 2K。

## 流程

```text
T2VA 提示词
  -> MiniMax-H3 Image-to-Video
  -> DMAD 4-step LoRA
  -> 1344×768 / 124 帧 / 24 fps
  -> 视频解码 + 音频解码
  -> 草稿作为 Ref2VA 的 <Video 1>
  -> 3D latent upscaler
  -> HyperFlow + LMS Sharp + Open-H3 IC LoRA
  -> up3 精修
  -> 2560×1440 最终视频
```

工作流只使用一个提示词源节点：

**T2VA prompt · edit this text**

它会同时提供：

1. 原始提示词给 H3 文生视频草稿阶段
2. 同一提示词给 IC 增强提示词包装节点

这样不会把 `Enhance <Video 1>` 误用于第一阶段文生视频。

## 必需的自定义节点

- [ComfyUI-OpenH3Regenerate2KIC](https://github.com/slmonker/ComfyUI-OpenH3Regenerate2KIC)
- `ComfyUI-LynnReal`
- `ComfyUI-Spark-H3`
- `ComfyUI-VideoHelperSuite`

工作流使用的关键节点：

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

## 必需模型

### 草稿生成

```text
models/diffusion_models/
└── minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors/
    └── minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors

models/loras/
└── dmad_minimax_h3_4step_full_critic_comfy.safetensors
```

### IC 精修

```text
models/latent_upscale_models/
└── minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors

models/loras/
├── minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors
├── minimax_h3_lms_v1.0_r64.safetensors
└── minimax_h3_open_regenerate_2k_ic_comfy_bf16.safetensors
```

从 Hugging Face 下载转换后的 IC LoRA：

<https://huggingface.co/slmonker/Open-H3-Regenerate-2K-IC-ComfyUI>

### 公共组件

```text
models/text_encoders/MiniMax-H3/
└── qwen3vl_32b_minimax_h3_int8_convrot.safetensors

models/vae/MiniMax-H3/
├── minimax_h3_video_vae_fp16.safetensors
└── minimax_h3_audio_vae_fp32.safetensors
```

hybrid FL2VA+Ref2VA 模型和 DMAD LoRA 来自 MiniMax-H3 / TaoMate-HD 生态。
hybrid 模型不在本仓库内。

## 运行

1. 在 ComfyUI 中打开 `../../workflows/regen2k_ic_T2VA.json`。
2. 修改 **T2VA prompt · edit this text**。
3. 运行工作流。
4. 输出位置：

```text
ComfyUI/output/OpenH3/T2VA_draft_768p_*
ComfyUI/output/OpenH3/T2VA_final_2k_*
```

## 默认参数

| 阶段 | 参数 |
|---|---|
| 草稿分辨率 | 1344×768 |
| 草稿帧数 | 124 |
| 帧率 | 24 fps |
| 草稿采样器 | Euler，4 步 |
| 草稿 CFG | 1 |
| IC 采样器 | `up3` |
| 最终分辨率 | 2560×1440 |

## 示例提示词

```text
A cinematic medium shot of a young woman in an elegant ancient Chinese dress
standing in a sunlit bamboo forest. She looks toward the camera and says in
Mandarin: "你好，欢迎来到这片竹林。"
Gentle camera push-in, soft breeze, rustling leaves, detailed fabric and hair.

overall_soundscape: Soft wind through bamboo leaves, distant birds, and a clear natural voice.

non_diegetic_music: A restrained guzheng melody with a warm, cinematic atmosphere.
```

## 注意事项

- 工作流包含两个输出节点：768P 草稿预览和 2K 最终视频。
- `BlockSparseAttention` 使用 `sol-attn` 选项；如果没有该节点，可绕过它或改用原生 attention。
- 2K 精修是显存占用最高的阶段。32GB 显卡建议使用 INT8 hybrid 模型，并关闭其他 GPU 程序。
- 这是社区工作流，不是 MiniMax 官方 ComfyUI 模板。
