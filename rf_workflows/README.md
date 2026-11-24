# Renderforest production workflow: `tool.upscale`

The `ai_workflows` row with code `tool.upscale` (type `upscale`) that ai-workers-service runs for an upscale tool job.

- `tool.upscale.api.json` → the row's `config` (ComfyUI API format): **RelayVSR x4 through Hayk's
  `RelayVSRUpscaleNative` node** (2026-11-24).
- `tool.upscale.mapping.json` → the row's `mapping`. ai-workers-service fills:
  - `videoUrl`: URL of the base video (node 1, `VHS_LoadVideoPath`, which reads http/https).
  - `targetLongSide`: the long side the RelayVSR node resizes to (node 6, `target_long_side`):
    `max(targetWidth, targetHeight)`, or, when the size is not sent, the long side of a 16:9 frame at the requested
    quality (720 → 1280, 1080 → 1920). Without it the node's default 1280 is used.
  - `targetWidth` / `targetHeight`: the exact output size, computed by ai-video-api (`getUpscaleTargetSize`): the
    base's shape snapped to the nearest standard ratio, short side = the requested tier (e.g. 1280×720, 720×1280,
    960×720). Node 22 (core `ImageScale`, crop `center`) covers that size and crops the overflow at the center, so
    an off-ratio vendor frame (Seedance 480p 16:9 is 864×496) comes out exactly 16:9 with no side bars, and the
    output always matches the size ai-video-api checks (±2 px). Without them the node's default 1280×720 is used.
- `tool.upscale.full-video.*`: the previous RelayVSR graph (core `RelayVSRConditioning` → `KSampler` → `VSRDecode`
  over the whole clip, then resize), kept for rollback; it maps `videoUrl`, `targetWidth`, `targetHeight`.
- `tool.upscale.seedvr2.*`: the SeedVR2 workflow (uses this repo's `SeedVR2UltraUpscale` node), kept for rollback;
  it maps `target` (`'720p'`/`'1080p'`) instead.

## Graph

```
1 VHS_LoadVideoPath ─┬─ IMAGE ──────────────┐
                     ├─ video_info → 2 VHS_VideoInfo ── loaded_fps ──────────────────┐
                     └─ audio ─────────────────────────────────────────────────────┐ │
4 UNETLoader ── MODEL ─┐                    │                                       │ │
5 VAELoader ─── VAE ───┴→ 6 RelayVSRUpscaleNative → 23 RebatchImages 64            │ │
                          → 22 ImageScale (area, center) → 20 ImageSharpen          │ │
                          → 21 RebatchImages 4096 → 10 VHS_VideoCombine ←───────────┴─┘
```

## What changed from the full-video graph (Hayk's 2026-11-24 workflow)

- **Chunked decode**: `RelayVSRUpscaleNative` (shipped in this repo as `relayvsr_native.py`) runs the same core path
  (conditioning → 1-step euler sampler → VSR decode) but in chunks of 241 frames, crossfaded over 8 frames. Only one
  chunk's 4x frames exist at a time; the full-video graph decoded the whole clip at 4x before resizing, so its memory
  grew with the clip length. Clips up to 241 frames (~10 s at 24 fps) run in one pass, identical to
  the native nodes.
- **GPU resize**: the node resizes each chunk on the GPU to `targetLongSide` right after the decode (area when
  shrinking, bicubic when growing); node 22 then only trims the last few pixels to `targetWidth`×`targetHeight`.
- Sharpen is Hayk's new values: radius 1, sigma 0.4, alpha 0.2 (was 0.5 / 0.3).

Kept from our graph rather than Hayk's:
- Node 1 is `VHS_LoadVideoPath` (URL input) with load `format` `None` (Hayk's earlier frame fix). His workflow uses
  `AnimateDiff`, which can resize the frames to multiples of 8 and so change the aspect before the upscale.
- Seed 42 (our fixed seed, reproducible results) instead of his 40243943309965.
- Node 22 `ImageScale` to the exact target size: his graph ends at the node's long-side resize, which does not
  guarantee the size ai-video-api checks.
- No `VHS_BatchManager`: our clips are ≤30 s, and VHS requeues the prompt once per meta batch while the worker only
  tracks the first prompt.

## Requirements on the worker

- ComfyUI with the core RelayVSR support (`comfy.ldm.wan.model_vsr`), core `ImageScale`, `ImageSharpen`,
  `RebatchImages`, and ComfyUI-VideoHelperSuite.
- **This repo installed as a custom node pack** (`custom_nodes/comfyui-ultra-upscale`): it registers
  `RelayVSRUpscaleNative`. Update the checkout and restart ComfyUI before switching the row.
- Models unchanged (https://huggingface.co/Comfy-Org/RelayVSR):
  - `models/diffusion_models/relayvsr_x4_int8_convrot.safetensors` (5.2 GB)
  - `models/vae/relayvsr_flash_vae_x4_fp32.safetensors` (128 MB)

## Deploy order

1. Worker boxes: pull this repo into `custom_nodes/comfyui-ultra-upscale` and restart ComfyUI (check the log has no
   `RelayVSR Upscale (native) not loaded` line).
2. ai-workers-service with `targetLongSide` in the tool mapping data (it is ignored by the current row, so it is
   safe to ship first).
3. The `ai_workflows` row `tool.upscale`: `config` = `tool.upscale.api.json`, `mapping` = `tool.upscale.mapping.json`.

A job built before step 2 gets the node's default long side 1280; node 22 still scales to the exact target size, so
a 1080p job is upscaled to 1280 and then enlarged to 1920 (correct size, softer picture).

## Rollback

Set the row's `config` / `mapping` back to `tool.upscale.full-video.api.json` / `tool.upscale.full-video.mapping.json`.
No worker or ai-workers-service change is needed (the extra `targetLongSide` key is ignored by that mapping).

Speed (Hayk, 5 s clip, full-video graph): 480→720 ~8 s, 720→1080 ~20 s.
