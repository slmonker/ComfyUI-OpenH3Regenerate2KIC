# ComfyUI Open-H3-Regenerate-2K-IC

[English](README.md) | 简体中文

这是 [NVlabs/Sana · Open-H3-Regenerate-2K/IC](https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K/IC)
的 ComfyUI 节点和示例工作流仓库。

仓库包含两套示例：

1. **文生视频示例**：文本 → 768P H3 草稿 → IC 2K 精修。
2. **草稿放大示例**：已有 768P 草稿 → IC 2K 精修。

- GitHub：<https://github.com/slmonker/ComfyUI-OpenH3Regenerate2KIC>
- 转换后的 ComfyUI LoRA：<https://huggingface.co/slmonker/Open-H3-Regenerate-2K-IC-ComfyUI>
- 官方上游权重：<https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC>

## 示例工作流

| 示例 | 工作流 | 说明 |
|---|---|---|
| 文生视频 | [`workflows/regen2k_ic_T2VA.json`](workflows/regen2k_ic_T2VA.json) | 先用文本生成 1344×768 草稿，再精修到 2560×1440 |
| 草稿转 2K | [`workflows/open_h3_regenerate_2k_ic.json`](workflows/open_h3_regenerate_2k_ic.json) | 将已有 768P 草稿精修到 2560×1440 |

文生视频示例说明：

- 英文：[`examples/T2VA/README.md`](examples/T2VA/README.md)
- 中文：[`examples/T2VA/README.zh-CN.md`](examples/T2VA/README.zh-CN.md)

## 文生视频流程

```text
文本提示词
  -> MiniMax-H3 Image-to-Video + DMAD 4-step LoRA
  -> 1344×768 / 124 帧 / 24 fps 草稿
  -> 解码草稿视频和草稿音频
  -> 草稿作为 Ref2VA 的 <Video 1>
  -> 官方 3D latent upscaler
  -> HyperFlow + LMS Sharp + Open-H3 IC LoRA
  -> up3 三步精修
  -> 2560×1440 最终视频
```

T2VA 工作流只使用一个文本输入节点，同时供两个阶段使用：

- 原始文本输入到 H3 文生视频草稿阶段
- 同一文本加上官方增强前缀后输入到 IC Ref2VA 阶段

## 草稿放大流程

```text
已有 768P 草稿
  -> <Video 1> Ref2VA 条件
  -> H3 VAE 编码 -> 官方 3D latent upscaler
  -> HyperFlow + LMS Sharp + IC LoRA
  -> up3 / up4 / hf8 sigma 网格
  -> fast H3 VAE 解码
  -> 回封装草稿原音频
```

## 目录

| 路径 | 作用 |
|---|---|
| `nodes.py` | 4 个 ComfyUI 配套节点 |
| `examples/T2VA/` | 完整文生视频示例和双语说明 |
| `workflows/open_h3_regenerate_2k_ic.json` | 草稿转 2K UI 工作流 |
| `workflows/open_h3_regenerate_2k_ic_api.json` | 草稿转 2K API 工作流 |
| `tools/convert_ic_lora.py` | Diffusers IC LoRA → ComfyUI 通用 LoRA |
| `tools/build_workflow.py` | 重建草稿转 2K 工作流 |
| `assets/silence_audio_124f.safetensors` | 官方 124 帧静音 latent |
| `tests/test_nodes.py` | 预处理和 sigma 单元测试 |

## 节点说明

### Open-H3 IC · Prepare Draft (124f / 24fps)

- 从输入视频按 24 fps 提取 124 帧
- cover 缩放并中心裁剪到 1344×768
- 按真实采样率裁剪或补齐音频
- 重采样后至少需要 113 个真实帧

### Open-H3 IC · Official Prompt

自动添加官方增强任务前缀，并把原始提示词保留为 `Scene description`。

### Open-H3 IC · Sigma Grid

| 模式 | Sigmas |
|---|---|
| `up3` | `0.834942, 0.696852, 0.468753, 0` |
| `up4` | `0.923077, 0.834942, 0.696852, 0.468753, 0` |
| `hf8` | `1.0, 0.99391, 0.984288, 0.966064, 0.923077, 0.834942, 0.696852, 0.468753, 0` |

### Open-H3 IC · Compose Target Latent

把放大后的目标视频 latent 与官方 124 帧静音 latent 组合，并通过
`noise_mask=0` 固定音频流。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/slmonker/ComfyUI-OpenH3Regenerate2KIC.git
```

安装后重启 ComfyUI。

文生视频示例还需要：

- `ComfyUI-LynnReal`：attention 和 feed-forward 辅助节点
- `ComfyUI-Spark-H3`：Sol-Attn 稀疏注意力
- `ComfyUI-VideoHelperSuite`：`VHS_VideoCombine`

草稿转 2K 示例还需要 `LynnRealH3VAELoader`，用于 fast H3 VAE 解码。

## 通用模型

放到 `ComfyUI/models/`：

```text
models/
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

从 Hugging Face 下载转换后的 IC LoRA：

<https://huggingface.co/slmonker/Open-H3-Regenerate-2K-IC-ComfyUI>

IC LoRA 加载顺序必须是：

1. HyperFlow 8-step
2. LMS Sharp
3. Open-H3 Regenerate 2K IC

### 仅文生视频示例需要的模型

```text
models/
├── diffusion_models/
│   └── minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors/
│       └── minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors
├── vae/MiniMax-H3/
│   └── minimax_h3_audio_vae_fp32.safetensors
└── loras/
    └── dmad_minimax_h3_4step_full_critic_comfy.safetensors
```

hybrid FL2VA+Ref2VA 模型和 DMAD LoRA 来自 MiniMax-H3 / TaoMate-HD 生态。
hybrid 模型不在本仓库内。

## 运行文生视频示例

1. 打开 `workflows/regen2k_ic_T2VA.json`。
2. 只修改 **T2VA prompt · edit this text** 一个提示词节点。
3. 运行工作流。
4. 768P 草稿预览和 2K 最终视频输出到 `ComfyUI/output/OpenH3/`。

默认参数：

- 草稿：1344×768，124 帧，24 fps，DMAD 4 步
- IC 精修：`up3`，CFG 1，2560×1440

## 运行草稿转 2K 示例

1. 将已有 768P 草稿放入 `ComfyUI/input/`。
2. 打开 `workflows/open_h3_regenerate_2k_ic.json`。
3. 在 **Load Video** 中选择草稿。
4. 在 **Open-H3 IC · Official Prompt** 中填写原始提示词。
5. 运行工作流。

1080P 可把两个分辨率节点都设置为 `1920×1088`；4K 使用
`3840×2176`，但 32GB 单卡通常不可行。

## 转换官方 IC LoRA

```bash
python tools/convert_ic_lora.py \
  open_h3_regenerate_2k_ic_lora.safetensors \
  minimax_h3_open_regenerate_2k_ic_comfy_bf16.safetensors \
  --dtype bf16
```

转换规则：

- q/k/v 融合为 `qkv_proj`，B 使用块对角矩阵
- `ff.net.0.proj` 从 Diffusers `[value;gate]` 调整为 ComfyUI `[gate;value]`
- 融合后的 qkv 使用 `alpha = 3 × rank`
- 输出 BF16

已发布转换结果：

- 312 个源线性层
- 208 个 ComfyUI 模块
- 624 个 tensor
- 基础模型缺失映射为 0

## 验证

```bash
python -m unittest discover -s tests -v
```

草稿转 2K 工作流已在 ComfyUI 前端加载并通过 `graphToPrompt()`，
输入 schema 检查为 0 错误。转换后的 LoRA 已对照
`minimax_h3_ref2va_bf16.safetensors` 检查全部目标权重。

## 注意事项

- 默认视频长度为 124 帧，约 5.17 秒。
- 文生视频示例保留第一阶段 H3 生成的草稿音频。
- IC 阶段内部使用固定静音，输出草稿音频。
- 32GB 显存下 2K 需要模型 offload 和低显存 attention。
- 这是社区实现，不是 MiniMax 官方 ComfyUI 产品。

## 致谢

- [Open-H3-Regenerate-2K-IC](https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K/IC)
- [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [HyperFlow](https://github.com/Video-Rebirth/hyperflow)
- [H3 latent upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)

## 许可证

节点代码为 Apache-2.0。MiniMax-H3 权重及其衍生资产仍受
MiniMax-H3 Community License 约束，详见 `NOTICE.md`。
