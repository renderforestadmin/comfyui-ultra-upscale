"""RelayVSR Upscale (native) - IMAGE in, IMAGE out.

Same code path as the built-in nodes (RelayVSRConditioning -> 1-step sampler -> VSR Decode),
plus GPU resize to a target long side. Clips longer than chunk_len are processed in chunks
with a crossfade. No LUT / saving: connect your own nodes and Video Combine.

Optional (all off by default, which reproduces the original behaviour exactly):
- target_width + target_height: cover that size and center-crop to it (same as core ImageScale
  with crop "center"), done in ONE resize from the 4x decode instead of the long-side resize.
- sharpen_radius / sharpen_sigma / sharpen_alpha: core ImageSharpen math, per chunk on the GPU.
  Sharpening is linear per frame and the crossfade is a per-pixel lerp of two frames, so
  sharpen(lerp(a, b)) == lerp(sharpen(a), sharpen(b)); the clamp is deferred until a frame's
  crossfade is done, so the result equals ImageSharpen applied after the crossfade.

Chunks are written into one preallocated output tensor (no list + torch.cat at the end), so the
batch holds a single copy of the output.
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


def _gaussian_kernel(kernel_size, sigma, device=None):
    # comfy_extras/nodes_post_processing.py gaussian_kernel
    x, y = torch.meshgrid(torch.linspace(-1, 1, kernel_size, device=device),
                          torch.linspace(-1, 1, kernel_size, device=device), indexing="ij")
    d = torch.sqrt(x * x + y * y)
    g = torch.exp(-(d * d) / (2.0 * sigma * sigma))
    return g / g.sum()


def _sharpen_kernel(radius, sigma, alpha):
    # comfy_extras/nodes_post_processing.py Sharpen.execute (kernel part), for 3 channels
    kernel_size = radius * 2 + 1
    kernel = _gaussian_kernel(kernel_size, sigma, device=mm.get_torch_device()) * -(alpha * 10)
    kernel = kernel.to(dtype=torch.float32)
    center = kernel_size // 2
    kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0
    return kernel.repeat(3, 1, 1).unsqueeze(1)


def _sharpen_nchw(x, kernel, radius):
    # Sharpen.execute on a (B, C, H, W) batch, WITHOUT the final clamp (the caller clamps).
    center = kernel.shape[-1] // 2
    kernel = kernel.to(dtype=x.dtype)
    x = F.pad(x, (radius, radius, radius, radius), "reflect")
    return F.conv2d(x, kernel, padding=center, groups=x.shape[1])[:, :, radius:-radius, radius:-radius]


def _center_crop_box(h, w, size):
    # comfy.utils.common_upscale crop="center": the (y, x) offsets that trim (h, w) to the aspect of size.
    th, tw = size
    old_aspect, new_aspect = w / h, tw / th
    y = x = 0
    if old_aspect > new_aspect:
        x = round((w - w * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((h - h * (old_aspect / new_aspect)) / 2)
    return y, x


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
            },
            "optional": {
                "target_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1,
                                         "tooltip": "With target_height: cover this size and center-crop to it exactly "
                                                    "(one resize from the 4x decode; target_long_side is ignored). 0 = off."}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1,
                                          "tooltip": "See target_width. 0 = off."}),
                "sharpen_radius": ("INT", {"default": 0, "min": 0, "max": 31, "step": 1,
                                           "tooltip": "Core ImageSharpen per chunk on the GPU. 0 = off."}),
                "sharpen_sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "sharpen_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
            },
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
    def _resize(frames, plan):
        """Resize (or crop + resize) frames on the GPU in batches of 8, optionally sharpen, and
        yield (offset, frames on the intermediate device). plan: size, crop, kernel, radius."""
        dev = mm.get_torch_device()
        th, tw = plan["size"]
        kernel, radius = plan["kernel"], plan["radius"]
        for i in range(0, frames.shape[0], 8):  # small GPU batches: fast, no 32-bit index limits
            x = frames[i:i + 8].to(dev, non_blocking=True).permute(0, 3, 1, 2)
            if plan["crop"]:
                y0, x0 = _center_crop_box(x.shape[2], x.shape[3], (th, tw))
                x = x.narrow(2, y0, x.shape[2] - y0 * 2).narrow(3, x0, x.shape[3] - x0 * 2)
            if (x.shape[2], x.shape[3]) != (th, tw):
                if th < x.shape[2]:
                    x = F.interpolate(x, size=(th, tw), mode="area")
                else:
                    x = F.interpolate(x, size=(th, tw), mode="bicubic", antialias=True)
            x = x.clamp(0, 1)
            if kernel is not None:
                x = _sharpen_nchw(x, kernel, radius)
            yield i, x.permute(0, 2, 3, 1).to(mm.intermediate_device())
            del x

    def upscale(self, model, vae, images, target_long_side, chunk_len, chunk_overlap, seed, detail,
                target_width=0, target_height=0, sharpen_radius=0, sharpen_sigma=1.0, sharpen_alpha=1.0):
        total = images.shape[0]
        overlap = max(0, min(chunk_overlap, chunk_len // 2))
        step = chunk_len - overlap
        n_chunks = 1 + -(-max(total - chunk_len, 0) // step)
        pbar = comfy.utils.ProgressBar(n_chunks)
        crop = target_width > 0 and target_height > 0
        sharpen = sharpen_radius > 0
        kernel = _sharpen_kernel(sharpen_radius, sharpen_sigma, sharpen_alpha) if sharpen else None
        result, plan, start, done = None, None, 0, 0
        while start < total:
            mm.throw_exception_if_processing_interrupted()
            end = min(start + chunk_len, total)
            is_last = end >= total
            out = self._process(model, vae, images[start:end], seed, detail)
            if out.shape[0] != end - start:
                raise RuntimeError(f"RelayVSR Native: chunk of {end - start} frames decoded to {out.shape[0]} frames")
            if plan is None:
                h, w = out.shape[1], out.shape[2]
                if crop:
                    size = (target_height, target_width)
                elif target_long_side > 0:
                    sc = target_long_side / max(h, w)
                    size = (_even(h * sc), _even(w * sc))
                else:
                    size = (_even(h), _even(w))
                plan = {"size": size, "crop": crop, "kernel": kernel, "radius": sharpen_radius}
                # One preallocated output for the whole batch: the only full-size copy held in RAM.
                result = torch.empty((total, size[0], size[1], 3), dtype=out.dtype,
                                     device=mm.intermediate_device())
            # Frames [start, start + o) overlap the previous chunk's tail, already written to result:
            # crossfade into it (same weights as before). The rest is written straight in.
            o = min(overlap, end - start) if start > 0 else 0
            wgt = torch.linspace(1.0 / (o + 1), o / (o + 1.0), o).view(o, 1, 1, 1) if o else None
            tp = time.perf_counter()
            for i, x in self._resize(out, plan):
                a, b = start + i, start + i + x.shape[0]
                if a < start + o:
                    k = min(b, start + o) - a
                    w8 = wgt[i:i + k]
                    result[a:a + k] = result[a:a + k] * (1.0 - w8) + x[:k] * w8
                    x = x[k:]
                    a += k
                if a < b:
                    result[a:b] = x
                del x
            del out
            if sharpen:
                # Deferred clamp: frames whose crossfade is done (the tail waits for the next chunk).
                result[start:end if (is_last or overlap == 0) else end - overlap].clamp_(0, 1)
            ts, td, where = self._timing
            print(f"[RelayVSR Native] chunk {done + 1}/{n_chunks}: sample {ts:.1f}s | decode {td:.1f}s ({where}) | resize {time.perf_counter() - tp:.1f}s")
            done += 1
            pbar.update_absolute(done, n_chunks)
            if is_last:
                break
            start = end - overlap
        return (result,)

NODE_CLASS_MAPPINGS = {"RelayVSRUpscaleNative": RelayVSRUpscaleNative}
NODE_DISPLAY_NAME_MAPPINGS = {"RelayVSRUpscaleNative": "RelayVSR Upscale (native)"}
