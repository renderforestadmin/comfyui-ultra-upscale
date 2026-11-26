"""RelayVSR Upscale (to file) - local video path in, finished h264 mp4 out, in ONE ComfyUI prompt.

Hayk's `RelayVSRUpscaleNative` (relayvsr_native.py, imported and reused unchanged: `_process`, `_resize`,
`_sharpen_kernel`) run as a stream instead of IMAGE in / IMAGE out:

- the source is decoded like VHS_LoadVideoPath (cv2.VideoCapture, BGR -> RGB, float32 / 255, fps from the
  container, frame size from the decoded frame so cv2's rotation handling applies exactly as in VHS), incrementally:
  at most one chunk (chunk_len frames) of source frames is held, plus one look-ahead frame;
- every chunk on Hayk's exact grid (step = chunk_len - overlap) goes through his `_process` (conditioning ->
  1-step sampler -> VSR decode) and `_resize` (center cover-crop + one GPU resize + optional sharpen);
- finished frames are crossfaded with his weights and deferred clamp (relayvsr_stream.CrossfadeStream), converted
  to uint8 exactly like VHS_VideoCombine (x * 255 + 0.5, clip) and piped as rgb24 into ffmpeg with the VHS
  video/h264-mp4 arguments; only the last `overlap` output frames are held between chunks;
- the source audio (first audio stream, if any) is muxed afterwards like VHS (aac, padded / cut to the video);
- returns VHS_VideoCombine's `{"ui": {"gifs": [...]}}` and no IMAGE output, so nothing frame-sized enters
  ComfyUI's cache. Memory is O(one chunk) whatever the clip length.

On any error or interrupt ffmpeg is killed and the partial outputs are deleted. A decoder that stops clearly before
the container's frame count raises instead of producing a short video.
"""
import os
import time

import cv2
import numpy as np
import torch

import comfy.model_management as mm
import comfy.utils
import folder_paths

from .relayvsr_native import RelayVSRUpscaleNative, _sharpen_kernel
from .relayvsr_stream import (GIB, CrossfadeStream, choose_chunk_len, effective_overlap, OutputFiles, FfmpegWriter, ReadStats, build_encode_args,
                              check_frames_read, check_frames_written, discover_ffmpeg, encode_stream,
                              estimate_chunks, expected_frames, finish_with_audio, iter_chunks, probe_has_audio,
                              ui_preview, vhs_frame_rate)

_FFMPEG = None


def _ffmpeg():
    global _FFMPEG
    if _FFMPEG is None:
        _FFMPEG = discover_ffmpeg()
        print(f"[RelayVSR to file] ffmpeg: {_FFMPEG}")
    return _FFMPEG


class _VideoReader:
    """VHS cv_frame_generator (force_rate 0, no skip / select) as a pull reader writing into a chunk buffer."""

    def __init__(self, path, chunk_len):
        self.path = path
        self.chunk_len = chunk_len
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened() or not self.cap.grab():
            self.cap.release()
            raise ValueError(f"{path} could not be loaded with cv.")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.container_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, first = self.cap.retrieve()  # VHS: frame size from the decoded frame, not the container
        if not ok or first is None:
            self.cap.release()
            raise ValueError(f"{path}: the first frame could not be decoded.")
        self.height, self.width = first.shape[0], first.shape[1]
        self._first = first
        self.buffer = None

    def next_frame(self):
        if self._first is not None:
            frame, self._first = self._first, None
            return frame
        if not self.cap.grab():
            return None
        ok, frame = self.cap.retrieve()
        if not ok or frame is None:
            return None  # treated as the end of the stream; the short-read check catches a truncated decode
        return frame

    def put(self, slot, frame):
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise RuntimeError(f"RelayVSR to file: frame size changed mid-video ({frame.shape[1]}x{frame.shape[0]}, "
                               f"first frame {self.width}x{self.height})")
        if self.buffer is None:
            self.buffer = np.empty((self.chunk_len, self.height, self.width, 3), dtype=np.float32)
        # Same as VHS: cvtColor BGR -> RGB, np.array(..., float32), torch.from_numpy(frame).div_(255)
        dst = self.buffer[slot]
        dst[...] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        torch.from_numpy(dst).div_(255)

    def move(self, src, dst, count):
        self.buffer[dst:dst + count] = self.buffer[src:src + count].copy()

    def close(self):
        self.cap.release()
        self.buffer = None


def _to_bytes(frames):
    """VHS tensor_to_bytes: clip(x * 255 + 0.5, 0, 255) -> uint8 (rgb24)."""
    arr = np.clip(frames.cpu().numpy() * (2 ** 8 - 1) + 0.5, 0, 2 ** 8 - 1).astype(np.uint8)
    return arr


class RelayVSRUpscaleToFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Load Diffusion Model: relayvsr_x4_int8_convrot (or x2)."}),
                "vae": ("VAE", {"tooltip": "Load VAE: relayvsr_flash_vae (same scale as the model)."}),
                "video": ("STRING", {"default": "", "placeholder": "local video path (link RF Download Video)"}),
                "target_width": ("INT", {"default": 1280, "min": 2, "max": 16384, "step": 2,
                                         "tooltip": "Output width (even): cover the target and center-crop to it."}),
                "target_height": ("INT", {"default": 720, "min": 2, "max": 16384, "step": 2,
                                          "tooltip": "Output height (even)."}),
                "sharpen_radius": ("INT", {"default": 1, "min": 0, "max": 31, "step": 1,
                                           "tooltip": "Core ImageSharpen per chunk on the GPU. 0 = off."}),
                "sharpen_sigma": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 10.0, "step": 0.01}),
                "sharpen_alpha": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.01}),
                "chunk_len": ("INT", {"default": 241, "min": 16, "max": 2401, "step": 1,
                                      "tooltip": "Max frames per RelayVSR pass; lowered per source size by "
                                                 "max_chunk_gb."}),
                "chunk_overlap": ("INT", {"default": 8, "min": 0, "max": 120,
                                          "tooltip": "Frames crossfaded between chunks."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "detail": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_chunk_gb": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 1024.0, "step": 0.5,
                                           "tooltip": "Cap (GiB) for one chunk's 4x decode buffer: the chunk length "
                                                      "is the largest 15k+1 (<= chunk_len) that fits. 0 = off."}),
                "crf": ("INT", {"default": 14, "min": 0, "max": 51, "step": 1}),
                "filename_prefix": ("STRING", {"default": "RelayVSR/upscaled"}),
            },
            "optional": {
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1,
                                           "tooltip": "Upscale only the first N frames. 0 = all."}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "upscale"
    CATEGORY = "video/upscaling"
    DESCRIPTION = ("RelayVSR (native) streamed from a local video to an h264 mp4 with the source audio, in one "
                   "prompt. Memory O(one chunk); no IMAGE output.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always run: a cached result would point the worker at an old (possibly cleaned up) output file.
        return float("nan")

    def upscale(self, *, model, vae, video, target_width, target_height, sharpen_radius, sharpen_sigma, sharpen_alpha,
                chunk_len, chunk_overlap, seed, detail, max_chunk_gb, crf, filename_prefix, frame_load_cap=0):
        if target_width <= 0 or target_height <= 0 or target_width % 2 or target_height % 2:
            raise ValueError(f"RelayVSR to file: target size must be positive and even, got "
                             f"{target_width}x{target_height}")
        path = (video or "").strip().strip('"')
        if path.lower().startswith(("http://", "https://")):
            raise ValueError("RelayVSR to file: `video` must be a local path (link RF Download Video before it)")
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"RelayVSR to file: video not found: {path!r}")

        ffmpeg = _ffmpeg()
        has_audio = probe_has_audio(ffmpeg, path)
        reader = _VideoReader(path, chunk_len)
        try:
            # Bound the per-chunk working set by the source size (the 4x decode of a chunk), not a fixed length.
            scale = int(getattr(vae.first_stage_model, "upscale_factor", 4) or 4)
            requested = chunk_len
            chunk_len, decode_bytes = choose_chunk_len(width=reader.width, height=reader.height, chunk_len=chunk_len,
                                                       max_chunk_gb=max_chunk_gb, scale=scale)
            reader.chunk_len = chunk_len
            print(f"[RelayVSR to file] source {reader.width}x{reader.height} @ {reader.fps:g} fps, "
                  f"{reader.container_frames} frames -> {target_width}x{target_height}: chunk_len {chunk_len} "
                  f"(requested {requested}, x{scale} decode {decode_bytes / GIB:.1f} GiB per chunk, "
                  f"cap {max_chunk_gb:g} GiB), overlap {effective_overlap(chunk_len, chunk_overlap)}")
            frame_rate = vhs_frame_rate(reader.fps)
            output_dir = folder_paths.get_output_directory()
            full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(filename_prefix,
                                                                                               output_dir)
            files = OutputFiles.allocate(full_output_folder, filename)
            try:
                frames = self._encode(model=model, vae=vae, reader=reader, files=files, ffmpeg=ffmpeg,
                                      frame_rate=frame_rate, target_width=target_width, target_height=target_height,
                                      sharpen_radius=sharpen_radius, sharpen_sigma=sharpen_sigma,
                                      sharpen_alpha=sharpen_alpha, chunk_len=chunk_len, chunk_overlap=chunk_overlap,
                                      seed=seed, detail=detail, crf=crf, frame_load_cap=frame_load_cap)
                final_name = finish_with_audio(ffmpeg=ffmpeg, files=files, source_path=path, frames=frames,
                                               frame_rate=frame_rate, has_audio=has_audio)
            except BaseException:
                files.remove()
                raise
        finally:
            reader.close()
        print(f"[RelayVSR to file] {frames} frames at {frame_rate:g} fps, "
              f"{'with' if has_audio else 'no'} audio -> {os.path.join(subfolder, final_name)}")
        return {"ui": {"gifs": [ui_preview(files=files, final_name=final_name, subfolder=subfolder,
                                           frame_rate=frame_rate)]}}

    def _encode(self, *, model, vae, reader, files, ffmpeg, frame_rate, target_width, target_height, sharpen_radius,
                sharpen_sigma, sharpen_alpha, chunk_len, chunk_overlap, seed, detail, crf, frame_load_cap):
        native = RelayVSRUpscaleNative()
        sharpen = sharpen_radius > 0
        kernel = _sharpen_kernel(sharpen_radius, sharpen_sigma, sharpen_alpha) if sharpen else None
        # RelayVSRUpscaleNative.upscale's plan with target_width / target_height set (crop = True).
        plan = {"size": (target_height, target_width), "crop": True, "kernel": kernel, "radius": sharpen_radius}
        weights = {}

        def blend(held, new, o, a):
            # Hayk: result[a:a+k] * (1.0 - w8) + x[:k] * w8, w8 = linspace(1/(o+1), o/(o+1), o)[a:a+k]; the sum is
            # stored back into his result tensor (dtype of the decode), so cast to it.
            if o not in weights:
                weights[o] = torch.linspace(1.0 / (o + 1), o / (o + 1.0), o).view(o, 1, 1, 1)
            w8 = weights[o][a:a + len(new)].to(held.device)
            return (held * (1.0 - w8) + new * w8).to(held.dtype)

        def finalize(frames):
            # Hayk's deferred clamp (only when sharpening; without it _resize already clamped).
            return frames.clamp(0, 1) if sharpen else frames

        stream = CrossfadeStream(blend, finalize, lambda parts: torch.cat(parts, 0))
        stats = ReadStats()
        n_est = estimate_chunks(expected_frames(reader.container_frames, frame_load_cap) or chunk_len,
                                chunk_len, chunk_overlap)
        pbar = comfy.utils.ProgressBar(n_est)
        writer = FfmpegWriter(build_encode_args(ffmpeg=ffmpeg, width=target_width, height=target_height,
                                                frame_rate=frame_rate, crf=crf, out_path=files.video_path),
                              files.video_path)

        def batches():
            first = True
            for chunk in iter_chunks(next_frame=reader.next_frame, put=reader.put, move=reader.move,
                                     chunk_len=chunk_len, chunk_overlap=chunk_overlap, limit=frame_load_cap,
                                     stats=stats):
                mm.throw_exception_if_processing_interrupted()
                src = torch.from_numpy(reader.buffer[:chunk.n])
                out = native._process(model, vae, src, seed, detail)
                del src
                if out.shape[0] != chunk.n:
                    raise RuntimeError(f"RelayVSR to file: chunk of {chunk.n} frames decoded to {out.shape[0]} frames")
                tp = time.perf_counter()
                for done in stream.feed(chunk, native._resize(out, plan)):
                    arr = _to_bytes(done)
                    if first:
                        self._save_first_frame(arr[0], files.png_path)
                        first = False
                    yield arr.tobytes(), arr.shape[0]
                    del done, arr
                del out
                ts, td, where = native._timing
                done_chunks = chunk.index + 1
                # n_est comes from the container's frame count; the last chunk closes the bar whatever it said.
                total_chunks = done_chunks if chunk.is_last else max(n_est, done_chunks + 1)
                print(f"[RelayVSR to file] chunk {done_chunks}/{total_chunks}: {chunk.n} frames | sample {ts:.1f}s | "
                      f"decode {td:.1f}s ({where}) | resize+write {time.perf_counter() - tp:.1f}s")
                pbar.update_absolute(done_chunks, total_chunks)

        writer.start()
        frames = encode_stream(writer, batches())
        check_frames_written(stream.emitted, stats.frames_read)
        check_frames_written(frames, stats.frames_read)
        check_frames_read(frames_read=stats.frames_read, container_count=reader.container_frames,
                          frame_load_cap=frame_load_cap, path=reader.path)
        return frames

    @staticmethod
    def _save_first_frame(rgb, path):
        # VHS_VideoCombine saves the first frame as <name>.png (its `workflow` preview entry); no prompt metadata
        # here, so the job's signed URLs are not written into the output folder.
        try:
            from PIL import Image
            Image.fromarray(rgb).save(path, compress_level=4)
        except Exception as e:
            print(f"[RelayVSR to file] could not save the first frame: {e}")


NODE_CLASS_MAPPINGS = {"RelayVSRUpscaleToFile": RelayVSRUpscaleToFile}
NODE_DISPLAY_NAME_MAPPINGS = {"RelayVSRUpscaleToFile": "RelayVSR Upscale (to file)"}
