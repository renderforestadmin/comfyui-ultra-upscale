"""RelayVSR Upscale (native) - IMAGE in, IMAGE out.

Same code path as the built-in nodes (RelayVSRConditioning -> 1-step sampler -> VSR Decode),
plus GPU resize to a target long side. Clips longer than chunk_len are processed in chunks
with a crossfade. No sharpen / LUT / saving: connect your own nodes and Video Combine.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils
import nodes
from comfy.ldm.wan.model_vsr import relay_keyframes, relay_latent_size


def _even(x):
    return max(2, int(round(x / 2.0)) * 2)


class RelayVSRUpscaleNative:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Load Diffusion Model: relayvsr_x4_int8_convrot (or x2)."}),
                "vae": ("VAE", {"tooltip": "Load VAE: relayvsr_flash_vae (same scale as the model)."}),
                "images": ("IMAGE", {"tooltip": "Video frames: VAE Decode of a generation, or VHS Load Video."}),
                "target_long_side": ("INT", {"default": 1920, "min": 0, "max": 16384, "step": 2,
                                             "tooltip": "Final long side: 1280 / 1920 / 2560. Vertical ok. 0 = native output size."}),
                "chunk_len": ("INT", {"default": 241, "min": 16, "max": 2401, "step": 1,
                                      "tooltip": "Clips up to this many frames run in ONE pass (identical to the native nodes)."}),
                "chunk_overlap": ("INT", {"default": 8, "min": 0, "max": 120,
                                          "tooltip": "Frames crossfaded between chunks (only for clips longer than chunk_len)."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "detail": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "upscale"
    CATEGORY = "video/upscaling"
    DESCRIPTION = "Native RelayVSR + GPU resize. IMAGE in, IMAGE out (connect Sharpen / LUT / Video Combine after it)."

    def _process(self, model, vae, images, seed, detail):
        images = images.float()
        dev = mm.get_torch_device()
        t0 = time.perf_counter()
        dm = model.model.diffusion_model
        keys = relay_keyframes(images.shape[0])
        positive = [[dm.positive_conditioning.unsqueeze(0).clone(), {"lq_images": images[keys]}]]
        lh, lw = relay_latent_size(images.shape[1], images.shape[2], dm.latent_stride)
        latent = torch.zeros([1, model.model.latent_format.latent_channels, len(keys), lh, lw],
                             device=mm.intermediate_device())
        out = nodes.common_ksampler(model, seed, 1, 1.0, "euler", "simple", positive, positive,
                                    {"samples": latent}, denoise=1.0)[0]
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Decode straight into VRAM when it fits: no CPU round trip for the huge 4x frames.
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        scale = int(getattr(vae.first_stage_model, "upscale_factor", 4) or 4)
        need = n * h * w * scale * scale * 3 * 4
        on_gpu = False
        old_dev = vae.output_device
        try:
            if dev.type == "cuda" and mm.get_free_memory(dev) > need * 1.6 + 6 * 1024 ** 3:
                vae.output_device = dev
                on_gpu = True
            decoded = vae.decode(out["samples"], vae_options={"lq_images": images, "detail": detail})
        finally:
            vae.output_device = old_dev
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        self._timing = (t1 - t0, t2 - t1, "gpu" if on_gpu else "cpu")
        return decoded.reshape(-1, *decoded.shape[-3:])

    @staticmethod
    def _resize(frames, size):
        dev = mm.get_torch_device()
        th, tw = size
        outs = []
        for i in range(0, frames.shape[0], 8):  # small GPU batches: fast, no 32-bit index limits
            x = frames[i:i + 8].to(dev, non_blocking=True).permute(0, 3, 1, 2)
            if (x.shape[2], x.shape[3]) != (th, tw):
                if th < x.shape[2]:
                    x = F.interpolate(x, size=(th, tw), mode="area")
                else:
                    x = F.interpolate(x, size=(th, tw), mode="bicubic", antialias=True)
            outs.append(x.clamp(0, 1).permute(0, 2, 3, 1).to(mm.intermediate_device()))
        del frames
        return torch.cat(outs, 0)

    def upscale(self, model, vae, images, target_long_side, chunk_len, chunk_overlap, seed, detail):
        total = images.shape[0]
        overlap = max(0, min(chunk_overlap, chunk_len // 2))
        step = chunk_len - overlap
        n_chunks = 1 + -(-max(total - chunk_len, 0) // step)
        pbar = comfy.utils.ProgressBar(n_chunks)
        size, results, pending, start, done = None, [], None, 0, 0
        while start < total:
            mm.throw_exception_if_processing_interrupted()
            end = min(start + chunk_len, total)
            is_last = end >= total
            out = self._process(model, vae, images[start:end], seed, detail)
            if size is None:
                h, w = out.shape[1], out.shape[2]
                if target_long_side > 0:
                    sc = target_long_side / max(h, w)
                    size = (_even(h * sc), _even(w * sc))
                else:
                    size = (_even(h), _even(w))
            tp = time.perf_counter()
            out = self._resize(out, size)
            ts, td, where = self._timing
            print(f"[RelayVSR Native] chunk {done + 1}/{n_chunks}: sample {ts:.1f}s | decode {td:.1f}s ({where}) | resize {time.perf_counter() - tp:.1f}s")
            if pending is not None:
                o = min(pending.shape[0], out.shape[0])
                wgt = torch.linspace(1.0 / (o + 1), o / (o + 1.0), o).view(o, 1, 1, 1)
                out[:o] = pending[:o] * (1.0 - wgt) + out[:o] * wgt
            if is_last or overlap == 0:
                results.append(out)
                pending = None
            else:
                results.append(out[:-overlap])
                pending = out[-overlap:].clone()
            done += 1
            pbar.update_absolute(done, n_chunks)
            if is_last:
                break
            start = end - overlap
        if pending is not None:
            results.append(pending)
        return (torch.cat(results, 0),)


NODE_CLASS_MAPPINGS = {"RelayVSRUpscaleNative": RelayVSRUpscaleNative}
NODE_DISPLAY_NAME_MAPPINGS = {"RelayVSRUpscaleNative": "RelayVSR Upscale (native)"}
