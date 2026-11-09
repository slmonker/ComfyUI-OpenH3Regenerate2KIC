"""Build the ComfyUI UI workflow for Open-H3-Regenerate-2K-IC."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workflows" / "open_h3_regenerate_2k_ic.json"

nodes = []
links = []
next_id = 1


def node(node_type, pos, size, widgets=None, inputs=None, outputs=None, title=None, color=None):
    global next_id
    n = {
        "id": next_id,
        "type": node_type,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": len(nodes),
        "mode": 0,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "properties": {"Node name for S&R": node_type},
        "widgets_values": widgets if widgets is not None else None,
    }
    if title:
        n["title"] = title
    if color:
        n["color"] = color
        n["bgcolor"] = color
    nodes.append(n)
    next_id += 1
    return n


def socket(name, typ, link=None, widget=False, shape=None, label=None):
    value = {"name": name, "type": typ, "link": link}
    if widget:
        value["widget"] = {"name": name}
    if shape is not None:
        value["shape"] = shape
    if label:
        value["label"] = label
    return value


def output(name, typ):
    return {"name": name, "type": typ, "links": []}


def set_output(n, slot, **kwargs):
    n["outputs"][slot].update(kwargs)


def connect(src, src_slot, dst, dst_slot, typ):
    link_id = 4000 + len(links) + 1
    links.append([link_id, src["id"], src_slot, dst["id"], dst_slot, typ])
    src["outputs"][src_slot]["links"].append(link_id)
    dst["inputs"][dst_slot]["link"] = link_id
    return link_id


# 1: source draft
load = node("LoadVideo", (40, 80), (300, 110), ["draft.mp4"], outputs=[output("VIDEO", "VIDEO")],
            title="Load MiniMax-H3 768P draft")
# 2: decode frames and audio
components = node("GetVideoComponents", (380, 80), (260, 150),
                  inputs=[socket("video", "VIDEO")],
                  outputs=[output("images", "IMAGE"), output("audio", "AUDIO"),
                           output("fps", "FLOAT"), output("bit_depth", "COMBO"),
                           output("color_space", "COMBO")])
# 3: exact 124 frame / 24 fps preprocessing
draft = node("OpenH3ICDraft", (680, 80), (330, 280),
             [24.0, 1344, 768, 124, 24, "cover", "trim_pad"],
             inputs=[socket("images", "IMAGE"), socket("audio", "AUDIO", shape=7),
                     socket("source_fps", "FLOAT", widget=True),
                     socket("target_width", "INT", widget=True),
                     socket("target_height", "INT", widget=True),
                     socket("frame_count", "INT", widget=True),
                     socket("output_fps", "INT", widget=True),
                     socket("fit", "COMBO", widget=True),
                     socket("audio_mode", "COMBO", widget=True)],
             outputs=[output("draft_frames", "IMAGE"), output("draft_audio", "AUDIO"),
                      output("report", "STRING")], title="Prepare Draft (official preprocessing)")

# 4-6: models
vae = node("VAELoader", (40, 280), (320, 90),
           ["MiniMax-H3\\minimax_h3_video_vae_fp16.safetensors"],
           inputs=[socket("vae_name", "COMBO", widget=True)], outputs=[output("VAE", "VAE")],
           title="Official H3 video VAE (encode)")
fast_vae = node("LynnRealH3VAELoader", (820, 700), (360, 180),
                ["MiniMax-H3\\sol_h3_vae.safetensors", 0, 0, 0, True],
                inputs=[socket("vae_name", "COMBO", widget=True), socket("num_layers", "INT", widget=True),
                        socket("tile_size", "INT", widget=True), socket("tile_overlap", "INT", widget=True),
                        socket("compile_decoder", "BOOLEAN", widget=True)],
                outputs=[output("VAE", "VAE")], title="Fast H3 VAE decoder · official v2 24L")
clip = node("CLIPLoader", (40, 410), (360, 120),
            ["MiniMax-H3\\qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "minimax", "default"],
            inputs=[socket("clip_name", "COMBO", widget=True), socket("type", "COMBO", widget=True),
                    socket("device", "COMBO", widget=True, shape=7)], outputs=[output("CLIP", "CLIP")])
unet = node("UNETLoader", (40, 570), (360, 100),
            ["minimax_h3_ref2va_bf16.safetensors", "default"],
            inputs=[socket("unet_name", "COMBO", widget=True), socket("weight_dtype", "COMBO", widget=True)],
            outputs=[output("MODEL", "MODEL")])

# 7-9: official adapter order
lora_hf = node("LoraLoaderModelOnly", (450, 280), (330, 110),
               ["minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors", 1.0],
               inputs=[socket("model", "MODEL"), socket("lora_name", "COMBO", widget=True),
                       socket("strength_model", "FLOAT", widget=True)], outputs=[output("MODEL", "MODEL")],
               title="1 · HyperFlow 8-step")
lora_sharp = node("LoraLoaderModelOnly", (450, 420), (330, 110),
                  ["minimax_h3_lms_v1.0_r64.safetensors", 1.0],
                  inputs=[socket("model", "MODEL"), socket("lora_name", "COMBO", widget=True),
                          socket("strength_model", "FLOAT", widget=True)], outputs=[output("MODEL", "MODEL")],
                  title="2 · LMS Sharp")
lora_ic = node("LoraLoaderModelOnly", (450, 560), (330, 110),
               ["minimax_h3_open_regenerate_2k_ic_comfy_bf16.safetensors", 1.0],
               inputs=[socket("model", "MODEL"), socket("lora_name", "COMBO", widget=True),
                       socket("strength_model", "FLOAT", widget=True)], outputs=[output("MODEL", "MODEL")],
               title="3 · Open-H3 Regenerate 2K IC")

# 10: official video shift
shift = node("MiniMaxH3SigmaShift", (820, 440), (310, 100), [12.0, 3.0],
             inputs=[socket("model", "MODEL"), socket("shift_video", "FLOAT", widget=True),
                     socket("shift_audio", "FLOAT", widget=True)], outputs=[output("MODEL", "MODEL")],
             title="ModelSamplingMiniMaxH3 (12 / 3)")

# 11: prompt
prompt = node("OpenH3ICPrompt", (820, 80), (360, 200), [""],
              inputs=[socket("caption", "STRING", widget=True)], outputs=[output("prompt", "STRING")])

# 12: Ref2VA conditioning. The draft is <Video 1>.
ref2va = node("MiniMaxH3ReferenceToVideo", (1220, 80), (360, 340),
              ["", 2560, 1440, 124, "match"],
              inputs=[socket("clip", "CLIP"), socket("vae", "VAE", shape=7),
                      socket("audio_vae", "VAE", shape=7),
                      socket("ref_images.ref_image_0", "IMAGE", shape=7, label="ref_image_0"),
                      socket("ref_videos.ref_video_0", "IMAGE", shape=7, label="ref_video_0"),
                      socket("ref_video_audios.ref_video_audio_0", "AUDIO", shape=7, label="ref_video_audio_0"),
                      socket("ref_audios.ref_audio_0", "AUDIO", shape=7, label="ref_audio_0"),
                      socket("prompt", "STRING", widget=True), socket("width", "INT", widget=True),
                      socket("height", "INT", widget=True), socket("length", "INT", widget=True),
                      socket("ref_image_size", "COMBO", widget=True)],
              outputs=[output("positive", "CONDITIONING"), output("latent", "LATENT")],
              title="MiniMax H3 Ref2VA · draft = <Video 1>")

# 13: draft video latent
encode = node("VAEEncode", (1060, 470), (250, 80),
              inputs=[socket("pixels", "IMAGE"), socket("vae", "VAE")], outputs=[output("LATENT", "LATENT")])
# 14-15: official 3D latent upscaler
upscale_loader = node("LatentUpscaleModelLoader", (1060, 590), (310, 90),
                      ["minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"],
                      inputs=[socket("model_name", "COMBO", widget=True)],
                      outputs=[output("LATENT_UPSCALE_MODEL", "LATENT_UPSCALE_MODEL")])
upscale = node("MiniMaxH3LatentUpscaleWithModel", (1390, 590), (250, 120), [2560, 1440],
               inputs=[socket("model", "LATENT_UPSCALE_MODEL"), socket("samples", "LATENT"),
                       socket("width", "INT", widget=True), socket("height", "INT", widget=True)],
               outputs=[output("LATENT", "LATENT")], title="Upscale draft latent to 2560×1440")

# 16: target AV latent with pinned official silence
compose = node("OpenH3ICComposeLatent", (1670, 470), (330, 150), [True],
               inputs=[socket("video_latent", "LATENT"), socket("reference_latent", "LATENT"),
                       socket("pin_audio", "BOOLEAN", widget=True)],
               outputs=[output("LATENT", "LATENT")], title="Compose target latent")

# 17: fixed official sigmas
sigmas = node("OpenH3ICSigmas", (1670, 650), (310, 130), ["up3", ""],
              inputs=[socket("sampler", "COMBO", widget=True), socket("custom_sigmas", "STRING", widget=True)],
              outputs=[output("sigmas", "SIGMAS"), output("report", "STRING")])

# 18-21: sampling
noise = node("RandomNoise", (1670, 820), (300, 80), [42, "fixed"],
             inputs=[socket("noise_seed", "INT", widget=True)], outputs=[output("NOISE", "NOISE")])
guider = node("BasicGuider", (2010, 80), (250, 80),
              inputs=[socket("model", "MODEL"), socket("conditioning", "CONDITIONING")],
              outputs=[output("GUIDER", "GUIDER")])
sampler_select = node("KSamplerSelect", (2010, 200), (250, 80), ["euler"],
                      inputs=[socket("sampler_name", "COMBO", widget=True)],
                      outputs=[output("SAMPLER", "SAMPLER")])
sampler = node("SamplerCustomAdvanced", (2010, 320), (300, 210),
               inputs=[socket("noise", "NOISE"), socket("guider", "GUIDER"), socket("sampler", "SAMPLER"),
                       socket("sigmas", "SIGMAS"), socket("latent_image", "LATENT")],
               outputs=[output("output", "LATENT"), output("denoised_output", "LATENT")],
               title="IC 3-step refine · CFG 1")

# 22-24: decode, restore the draft audio, save
decode = node("VAEDecode", (2360, 320), (220, 80),
              inputs=[socket("samples", "LATENT"), socket("vae", "VAE")], outputs=[output("IMAGE", "IMAGE")])
create = node("CreateVideo", (2360, 440), (280, 150), [24, 8, "sRGB"],
              inputs=[socket("images", "IMAGE"), socket("audio", "AUDIO", shape=7),
                      socket("fps", "FLOAT", widget=True), socket("bit_depth", "COMBO", widget=True, shape=7),
                      socket("color_space", "COMBO", widget=True, shape=7)],
              outputs=[output("VIDEO", "VIDEO")], title="Mux refined video + draft audio")
save = node("SaveVideo", (2680, 440), (330, 180), ["video/OpenH3-Regenerate-2K-IC", "auto", "auto", "auto"],
            inputs=[socket("video", "VIDEO"), socket("filename_prefix", "STRING", widget=True),
                    socket("format", "COMFY_DYNAMICCOMBO_V3", widget=True),
                    socket("format.codec", "COMFY_DYNAMICCOMBO_V3", widget=True),
                    socket("codec", "COMFY_DYNAMICCOMBO_V3", widget=True, shape=7)],
            outputs=[output("video", "VIDEO")])

note = node("MarkdownNote", (40, 920), (660, 390), ["""# Open-H3-Regenerate-2K-IC · ComfyUI

This graph runs the **IC (in-context)** variant with native ComfyUI H3 nodes:

1. Load a 768P MiniMax-H3 draft (124 frames / about 5 s).
2. The draft becomes `<Video 1>` in **MiniMax H3 Reference to Video**.
3. The same draft is VAE-encoded and passed through the official 3D latent upscaler.
4. `up3` applies the official sigmas `0.834942, 0.696852, 0.468753, 0`.
5. The result is decoded with the official fast H3 VAE and muxed with the draft's original audio.

**Required converted LoRA:** `minimax_h3_open_regenerate_2k_ic_comfy_bf16.safetensors`
Use `tools/convert_ic_lora.py` on the official `open_h3_regenerate_2k_ic_lora.safetensors`.

**Caption:** enter the original MiniMax-H3 prompt that generated the draft. Empty is valid.

**Tip:** if VRAM is tight, replace the reference model with the pruned INT8
`minimax_h3_ref2va_pruned_int8_convrot.safetensors` and lower the target to
1920×1088. The official IC target defaults to 2560×1440.
"""], title="README / usage")
note["properties"]["Node name for S&R"] = "Note"

# Wiring
connect(load, 0, components, 0, "VIDEO")
connect(components, 0, draft, 0, "IMAGE")
connect(components, 1, draft, 1, "AUDIO")
connect(components, 2, draft, 2, "FLOAT")
connect(draft, 0, ref2va, 4, "IMAGE")
connect(draft, 0, encode, 0, "IMAGE")
connect(draft, 1, create, 1, "AUDIO")
connect(vae, 0, ref2va, 1, "VAE")
connect(vae, 0, encode, 1, "VAE")
connect(fast_vae, 0, decode, 1, "VAE")
connect(clip, 0, ref2va, 0, "CLIP")
connect(prompt, 0, ref2va, 7, "STRING")
connect(ref2va, 0, guider, 1, "CONDITIONING")
connect(ref2va, 1, compose, 1, "LATENT")
connect(unet, 0, lora_hf, 0, "MODEL")
connect(lora_hf, 0, lora_sharp, 0, "MODEL")
connect(lora_sharp, 0, lora_ic, 0, "MODEL")
connect(lora_ic, 0, shift, 0, "MODEL")
connect(shift, 0, guider, 0, "MODEL")
connect(encode, 0, upscale, 1, "LATENT")
connect(upscale_loader, 0, upscale, 0, "LATENT_UPSCALE_MODEL")
connect(upscale, 0, compose, 0, "LATENT")
connect(compose, 0, sampler, 4, "LATENT")
connect(sigmas, 0, sampler, 3, "SIGMAS")
connect(noise, 0, sampler, 0, "NOISE")
connect(guider, 0, sampler, 1, "GUIDER")
connect(sampler_select, 0, sampler, 2, "SAMPLER")
connect(sampler, 0, decode, 0, "LATENT")
connect(decode, 0, create, 0, "IMAGE")
connect(create, 0, save, 0, "VIDEO")

# UI metadata
for idx, n in enumerate(nodes):
    n["order"] = idx

workflow = {
    "last_node_id": max(n["id"] for n in nodes),
    "last_link_id": max(link[0] for link in links),
    "nodes": nodes,
    "links": links,
    "groups": [
        {"title": "1 · Draft", "bounding": [20, 0, 1000, 390], "color": "#3f789e", "font_size": 24, "flags": {}},
        {"title": "2 · Official H3 + adapters", "bounding": [20, 390, 1120, 390], "color": "#a1309b", "font_size": 24, "flags": {}},
        {"title": "3 · Ref2VA + latent upscale", "bounding": [1200, 0, 820, 760], "color": "#8a5a2b", "font_size": 24, "flags": {}},
        {"title": "4 · IC sampling + restore audio", "bounding": [1650, 0, 1380, 920], "color": "#3b7d4f", "font_size": 24, "flags": {}},
    ],
    "config": {},
    "extra": {
        "ds": {"scale": 0.72, "offset": [20, 30]},
        "open_h3_ic": {
            "upstream": "https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K/IC",
            "weights": "https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC",
            "default_sampler": "up3",
            "default_resolution": [2560, 1440],
        },
    },
    "version": 0.4,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(workflow, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(OUT)
print("nodes", len(nodes), "links", len(links))
