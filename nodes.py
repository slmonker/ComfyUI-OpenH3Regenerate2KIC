"""ComfyUI companion nodes for Open-H3-Regenerate-2K-IC.

The graph stays native to ComfyUI: MiniMax H3 Ref2VA loads the draft as
<Video 1>, the official 3D latent upscaler creates the high-resolution target,
and a short HyperFlow sigma grid refines it.  This module only supplies the
pieces ComfyUI does not expose directly:

* OpenH3ICDraft: exact 24 fps / 124 frame cover-crop and source-audio trim
* OpenH3ICPrompt: the official IC task prompt
* OpenH3ICSigmas: official up3 / up4 / hf8 sigma grids
* OpenH3ICComposeLatent: target video + fixed official silence audio rows,
  with the audio stream pinned during sampling
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

TASK = (
    "Enhance <Video 1> into a high-quality version of the same video. "
    "Preserve the scene, objects, camera motion, timing, and composition of every frame. "
    "Restore fine details and remove degradation. Do not continue or change the scene."
)

SIGMAS = {
    "up3": [0.834942, 0.696852, 0.468753, 0.0],
    "up4": [0.923077, 0.834942, 0.696852, 0.468753, 0.0],
    "hf8": [1.0, 0.99391, 0.984288, 0.966064, 0.923077, 0.834942, 0.696852, 0.468753, 0.0],
}

_ASSET_DIR = Path(__file__).resolve().parent / "assets"
_SILENCE = _ASSET_DIR / "silence_audio_124f.safetensors"


def _cover_resize(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """[T,H,W,C] -> [T,height,width,C], cover resize + center crop."""
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError("OpenH3ICDraft expects IMAGE [T,H,W,C]")
    x = frames[..., :3].movedim(-1, 1).float()
    h, w = x.shape[-2:]
    scale = max(height / h, width / w)
    sh, sw = max(height, round(h * scale)), max(width, round(w * scale))
    if (sh, sw) != (h, w):
        x = F.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False, antialias=scale < 1.0)
    top, left = (sh - height) // 2, (sw - width) // 2
    x = x[..., top:top + height, left:left + width]
    return x.movedim(1, -1).clamp(0.0, 1.0).contiguous()


def _fit_audio(audio, sample_count: int):
    if audio is None:
        return {"waveform": torch.zeros((1, 2, sample_count), dtype=torch.float32), "sample_rate": 32000}
    waveform = audio["waveform"].float().clone()
    if waveform.ndim != 3:
        raise ValueError("AUDIO waveform must be [B,C,L]")
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    if waveform.shape[-1] < sample_count:
        pad = waveform.new_zeros((waveform.shape[0], waveform.shape[1], sample_count - waveform.shape[-1]))
        waveform = torch.cat((waveform, pad), dim=-1)
    elif waveform.shape[-1] > sample_count:
        waveform = waveform[..., :sample_count]
    return {"waveform": waveform, "sample_rate": int(audio["sample_rate"])}


class OpenH3ICDraft:
    """Prepare a draft exactly like the official IC preprocessing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "target_width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32}),
                "target_height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "frame_count": ("INT", {"default": 124, "min": 124, "max": 362, "step": 17}),
                "output_fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "fit": (["cover", "stretch"], {"default": "cover"}),
                "audio_mode": (["trim_pad", "passthrough", "silent"], {"default": "trim_pad"}),
            },
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("draft_frames", "draft_audio", "report")
    FUNCTION = "prepare"
    CATEGORY = "Open-H3/Regenerate 2K IC"

    def prepare(self, images, source_fps, target_width, target_height, frame_count, output_fps,
                fit="cover", audio_mode="trim_pad", audio=None):
        if images.shape[0] < 1:
            raise ValueError("The draft video contains no frames")
        if target_width % 32 or target_height % 32:
            raise ValueError("target_width and target_height must be multiples of 32")
        if output_fps != 24:
            raise ValueError("Open-H3-Regenerate-2K-IC sampling is fixed at 24 fps; set output_fps=24")
        if (int(frame_count) - 5) % 17:
            raise ValueError("frame_count must be a MiniMax-H3 valid length: 5 + 17*k (124, 141, ...)")
        source_fps = float(source_fps)
        source_count = int(images.shape[0])
        wanted = [int(i * source_fps / output_fps) for i in range(frame_count)]
        valid = sum(1 for idx in wanted if idx < source_count)
        if valid < 113:
            raise ValueError(
                f"The draft yields only {valid} real frames at 24 fps. "
                "Open-H3-Regenerate-2K-IC needs at least 113 (~4.7 s)."
            )
        indices = torch.tensor([min(idx, source_count - 1) for idx in wanted], device=images.device)
        picked = images.index_select(0, indices)
        if fit == "cover":
            frames = _cover_resize(picked, int(target_height), int(target_width))
        else:
            x = picked[..., :3].movedim(-1, 1).float()
            x = F.interpolate(x, size=(int(target_height), int(target_width)), mode="bilinear",
                              align_corners=False, antialias=True)
            frames = x.movedim(1, -1).clamp(0.0, 1.0).contiguous()
        if audio_mode == "silent" or audio is None:
            out_audio = _fit_audio(None, int(round(frames.shape[0] / float(output_fps) * 32000)))
        elif audio_mode == "passthrough":
            waveform = audio["waveform"].float().clone()
            if waveform.shape[1] == 1:
                waveform = waveform.repeat(1, 2, 1)
            out_audio = {"waveform": waveform, "sample_rate": int(audio["sample_rate"])}
        else:
            sample_count = int(round(frames.shape[0] / float(output_fps) * int(audio["sample_rate"])))
            out_audio = _fit_audio(audio, sample_count)
        report = {
            "source_frames": source_count,
            "source_fps": source_fps,
            "real_frames_at_24fps": valid,
            "output_frames": int(frames.shape[0]),
            "output_fps": int(output_fps),
            "canvas": [int(target_width), int(target_height)],
            "audio_samples": int(out_audio["waveform"].shape[-1]),
            "audio_rate": int(out_audio["sample_rate"]),
        }
        return frames, out_audio, json.dumps(report, ensure_ascii=False)


class OpenH3ICPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"caption": ("STRING", {"multiline": True, "default": ""})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "Open-H3/Regenerate 2K IC"

    def build(self, caption):
        return (TASK + "\n\nScene description: " + (caption or "").strip(),)


class OpenH3ICSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": (["up3", "up4", "hf8"], {"default": "up3"}),
                "custom_sigmas": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "report")
    FUNCTION = "build"
    CATEGORY = "Open-H3/Regenerate 2K IC"

    def build(self, sampler, custom_sigmas=""):
        if custom_sigmas.strip():
            values = [float(part.strip()) for part in custom_sigmas.split(",") if part.strip()]
            if not values or abs(values[-1]) > 1e-12:
                raise ValueError("custom_sigmas must end in 0.0")
        else:
            if sampler not in SIGMAS:
                raise ValueError("Unknown IC sampler: %s" % sampler)
            values = list(SIGMAS[sampler])
        if any(b > a + 1e-12 for a, b in zip(values, values[1:])):
            raise ValueError("custom_sigmas must be non-increasing")
        if any(v < 0.0 or v > 1.0 for v in values):
            raise ValueError("custom_sigmas values must be in [0, 1]")
        report = {"sampler": sampler if not custom_sigmas.strip() else "custom", "sigmas": values,
                  "steps": max(0, len(values) - 1)}
        return torch.tensor(values, dtype=torch.float32), json.dumps(report)


class OpenH3ICComposeLatent:
    """Build the clean target AV latent and pin its audio rows to official silence."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT",),
                "reference_latent": ("LATENT",),
                "pin_audio": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "compose"
    CATEGORY = "Open-H3/Regenerate 2K IC"

    @staticmethod
    def _fit_silence(audio: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if audio.shape[:3] != target.shape[:3]:
            raise ValueError(f"Silence latent {tuple(audio.shape)} does not match target {tuple(target.shape)}")
        if audio.shape[-1] > target.shape[-1]:
            audio = audio[..., :target.shape[-1]]
        elif audio.shape[-1] < target.shape[-1]:
            pad = audio.new_zeros(audio.shape[:-1] + (target.shape[-1] - audio.shape[-1],))
            audio = torch.cat((audio, pad), dim=-1)
        return audio

    def compose(self, video_latent, reference_latent, pin_audio=True):
        try:
            import comfy.nested_tensor as nested
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("OpenH3ICComposeLatent must run inside ComfyUI") from exc

        video = video_latent["samples"]
        if video.is_nested:
            video = video.tensors[0]
        ref = reference_latent["samples"]
        if not ref.is_nested or len(ref.tensors) != 2:
            raise ValueError("reference_latent must be the joint AV latent from MiniMax H3 Reference to Video")
        video = video.to(dtype=ref.tensors[0].dtype, device=ref.tensors[0].device)
        if tuple(video.shape) != tuple(ref.tensors[0].shape):
            raise ValueError(
                "Upscaled video latent shape %s does not match reference AV target shape %s" %
                (tuple(video.shape), tuple(ref.tensors[0].shape))
            )
        raw = load_file(str(_SILENCE), device="cpu")["audio_latents"]
        silence = raw.permute(1, 0, 2).unsqueeze(0).to(device=ref.tensors[1].device, dtype=ref.tensors[1].dtype)
        silence = self._fit_silence(silence, ref.tensors[1])

        out = reference_latent.copy()
        out.update({k: v for k, v in video_latent.items() if k != "samples"})
        out["samples"] = nested.NestedTensor((video, silence))
        if pin_audio:
            out["noise_mask"] = nested.NestedTensor((torch.ones_like(video), torch.zeros_like(silence)))
        else:
            out.pop("noise_mask", None)
        out["open_h3_ic"] = {"audio": "official_silence", "pinned": bool(pin_audio)}
        return (out,)


NODE_CLASS_MAPPINGS = {
    "OpenH3ICDraft": OpenH3ICDraft,
    "OpenH3ICPrompt": OpenH3ICPrompt,
    "OpenH3ICSigmas": OpenH3ICSigmas,
    "OpenH3ICComposeLatent": OpenH3ICComposeLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenH3ICDraft": "Open-H3 IC · Prepare Draft (124f / 24fps)",
    "OpenH3ICPrompt": "Open-H3 IC · Official Prompt",
    "OpenH3ICSigmas": "Open-H3 IC · Sigma Grid",
    "OpenH3ICComposeLatent": "Open-H3 IC · Compose Target Latent",
}
