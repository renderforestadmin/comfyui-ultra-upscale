"""Streaming bookkeeping for RelayVSR Upscale (to file): plain Python, stdlib only, no torch / ComfyUI imports.

`relayvsr_to_file.py` (the node) plugs torch / cv2 / numpy into these pieces; everything here is unit tested
without them (tests/test_relayvsr_stream.py):

- `iter_chunks`: reads source frames incrementally on the exact chunk grid of `RelayVSRUpscaleNative.upscale`
  (step = chunk_len - overlap), holding at most one chunk of source frames plus one look-ahead frame.
- `CrossfadeStream`: emits finished output frames in order, crossfading each chunk's head into the previous chunk's
  held-back tail with Hayk's weights and deferred clamp, holding back only the last `overlap` output frames.
- frame accounting (`check_frames_read`), ffmpeg discovery and command building (same args as VHS_VideoCombine with
  format video/h264-mp4), output file naming (VHS counter), and `FfmpegWriter` (kill + delete on any error).
"""

import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import namedtuple

ENCODE_ARGS = ("utf-8", "backslashreplace")
VIDEO_FORMAT = "video/h264-mp4"
VIDEO_EXTENSION = "mp4"


# ---------------------------------------------------------------------------------------------------------------
# Chunk grid (same as RelayVSRUpscaleNative.upscale)
# ---------------------------------------------------------------------------------------------------------------

def effective_overlap(chunk_len, chunk_overlap):
    """RelayVSRUpscaleNative: overlap = max(0, min(chunk_overlap, chunk_len // 2))."""
    return max(0, min(int(chunk_overlap), int(chunk_len) // 2))


def estimate_chunks(total, chunk_len, chunk_overlap):
    """Number of chunks RelayVSRUpscaleNative runs for `total` frames (for the progress bar)."""
    if total <= 0:
        return 1
    step = chunk_len - effective_overlap(chunk_len, chunk_overlap)
    return 1 + -(-max(total - chunk_len, 0) // step)


# RelayVSR puts a keyframe every RELAY_KEYFRAME_INTERVAL frames plus one on the last frame
# (comfy.ldm.wan.model_vsr.relay_keyframes); Hayk's 241 = 16 * 15 + 1 ends exactly on the keyframe grid. Shorter
# chunks keep that shape: 15k + 1 (226, 211, ..., 136, ..., 61, 46, 31, 16).
RELAY_KEYFRAME_INTERVAL = 15
MIN_CHUNK_LEN = RELAY_KEYFRAME_INTERVAL + 1
GIB = 1024 ** 3


def decode_bytes_per_frame(width, height, scale=4):
    """Bytes of one frame of the chunk's VSR decode (float32 RGB at scale x the source size)."""
    return (width * scale) * (height * scale) * 3 * 4


def choose_chunk_len(*, width, height, chunk_len, max_chunk_gb, scale=4):
    """Frames per chunk for a width x height source so the chunk's VSR decode buffer stays within max_chunk_gb GiB.

    - chunk_len (the node input) is the ceiling and is used as is when it fits (241 for sources up to ~480p at 24 GiB);
    - otherwise the largest length on the keyframe grid (15k + 1) that fits, never below MIN_CHUNK_LEN (16);
    - max_chunk_gb <= 0 turns the bound off.
    Returns (chunk_len, decode_bytes_of_that_chunk)."""
    per_frame = decode_bytes_per_frame(width, height, scale)
    chunk_len = max(1, int(chunk_len))
    if max_chunk_gb is None or max_chunk_gb <= 0:
        return chunk_len, chunk_len * per_frame
    fit = int(max_chunk_gb * GIB // per_frame)
    if chunk_len <= fit:
        return chunk_len, chunk_len * per_frame
    best = (min(chunk_len, fit) - 1) // RELAY_KEYFRAME_INTERVAL * RELAY_KEYFRAME_INTERVAL + 1
    best = min(chunk_len, max(MIN_CHUNK_LEN, best))
    return best, best * per_frame


# index: 0-based chunk number; start: global index of the chunk's first source/output frame; n: frames in the chunk;
# o: head frames to crossfade into the held tail of the previous chunk (0 for the first chunk);
# hold_from: local index from which output frames are held back for the next chunk (n for the last chunk);
# is_last: no frames follow this chunk.
Chunk = namedtuple("Chunk", "index start n o hold_from is_last")


class ReadStats:
    def __init__(self):
        self.frames_read = 0


def iter_chunks(*, next_frame, put, move, chunk_len, chunk_overlap, limit=0, stats=None):
    """Read source frames incrementally and yield one `Chunk` per RelayVSR chunk, on the same grid as
    RelayVSRUpscaleNative.upscale over the whole clip.

    - next_frame() -> a frame, or None at the end of the source.
    - put(slot, frame): store `frame` in buffer slot `slot` (0 <= slot < chunk_len).
    - move(src, dst, count): copy buffer slots [src, src + count) to [dst, dst + count).
    - limit: stop after this many source frames (0 = all).
    When a chunk is yielded, buffer slots [0, chunk.n) hold its source frames in order. The buffer holds at most
    chunk_len frames; one extra frame is read ahead to know whether the chunk is the last one (Hayk's
    `is_last = end >= total` for a source whose length is not known in advance).
    """
    if chunk_len < 1:
        raise ValueError("chunk_len must be >= 1")
    stats = stats if stats is not None else ReadStats()
    overlap = effective_overlap(chunk_len, chunk_overlap)
    pending = []  # the look-ahead frame, if one was read

    def read():
        if pending:
            return pending.pop()
        if limit and stats.frames_read >= limit:
            return None
        frame = next_frame()
        if frame is not None:
            stats.frames_read += 1
        return frame

    start, kept, index = 0, 0, 0
    while True:
        n = kept
        while n < chunk_len:
            frame = read()
            if frame is None:
                break
            put(n, frame)
            n += 1
        if n == 0:
            raise RuntimeError("RelayVSR to file: no frames could be read from the video")
        if n == kept and index > 0:
            # Cannot happen: the previous chunk was only yielded as non-last after a frame was read ahead.
            raise RuntimeError("RelayVSR to file: internal error, empty chunk after a non-last chunk")
        if n < chunk_len:
            is_last = True
        else:
            nxt = read()
            is_last = nxt is None
            if not is_last:
                pending.append(nxt)
        o = min(overlap, n) if start > 0 else 0
        hold_from = n if (is_last or overlap == 0) else n - overlap
        yield Chunk(index, start, n, o, hold_from, is_last)
        if is_last:
            return
        if overlap:
            move(n - overlap, 0, overlap)
        kept = overlap
        start += n - overlap
        index += 1


class CrossfadeStream:
    """Turn per-chunk output batches into the finished output frames, in order.

    Same result as RelayVSRUpscaleNative.upscale writing into its preallocated output:
    - output frames [0, o) of a chunk are crossfaded into the previous chunk's tail:
      blend(held[a:a+k], x[:k], o, a) = held * (1 - w) + x * w with w = linspace(1/(o+1), o/(o+1), o)[a:a+k];
    - frames before `hold_from` are finished: finalize() (the deferred clamp when sharpening) and emitted;
    - frames from `hold_from` on (the last `overlap` frames of a non-last chunk) are held back unfinished, exactly
      like Hayk's result tensor keeps them unclamped until the next chunk crossfades into them.

    Works on anything sliceable along the frame axis (torch tensors in the node, lists in the tests).
    blend(held, new, o, a) -> blended batch; finalize(batch) -> batch; concat(list of batches) -> batch.
    """

    def __init__(self, blend, finalize, concat):
        self.blend = blend
        self.finalize = finalize
        self.concat = concat
        self.held = None
        self.emitted = 0

    def held_count(self):
        return 0 if self.held is None else len(self.held)

    def feed(self, chunk, batches):
        """batches: iterable of (offset, frames) covering output frames [0, chunk.n) in order (Hayk's _resize).
        Yields finished batches in output order; keeps the chunk's tail for the next chunk."""
        if self.held_count() != chunk.o:
            raise RuntimeError(f"RelayVSR to file: chunk {chunk.index + 1} crossfades {chunk.o} frames but "
                               f"{self.held_count()} are held back")
        held, tail, pos = self.held, [], 0
        for offset, x in batches:
            if offset != pos:
                raise RuntimeError(f"RelayVSR to file: resize batch at {offset}, expected {pos}")
            b = len(x)
            pos += b
            a = offset
            if a < chunk.o:
                k = min(offset + b, chunk.o) - a
                head = self.blend(held[a:a + k], x[:k], chunk.o, a)
                yield from self._split(chunk, a, head, tail)
                x = x[k:]
                a += k
            if len(x):
                yield from self._split(chunk, a, x, tail)
        if pos != chunk.n:
            raise RuntimeError(f"RelayVSR to file: chunk {chunk.index + 1} produced {pos} frames, expected {chunk.n}")
        self.held = (tail[0] if len(tail) == 1 else self.concat(tail)) if tail else None
        if chunk.is_last and self.held is not None:
            raise RuntimeError("RelayVSR to file: frames held back after the last chunk")

    def _split(self, chunk, a, frames, tail):
        """frames are output frames [a, a + len) of the chunk: emit those before hold_from, hold the rest."""
        cut = max(0, min(len(frames), chunk.hold_from - a))
        if cut:
            done = self.finalize(frames[:cut])
            self.emitted += cut
            yield done
        if cut < len(frames):
            tail.append(frames[cut:])


# ---------------------------------------------------------------------------------------------------------------
# Frame accounting
# ---------------------------------------------------------------------------------------------------------------

def short_read_tolerance(expected):
    """Frames a read may fall short of the container's frame count (an estimate for some containers)."""
    return max(2, int(math.ceil(expected * 0.005)))


def expected_frames(container_count, frame_load_cap=0):
    """Frames the read should produce, or None when the container does not report a count."""
    if container_count is None or container_count <= 0:
        return None
    return min(container_count, frame_load_cap) if frame_load_cap > 0 else container_count


def check_frames_read(*, frames_read, container_count, frame_load_cap=0, path=""):
    """Raise when the decoder stopped clearly before the container's frame count (a truncated or unreadable
    input must never produce a "successful" short video)."""
    expected = expected_frames(container_count, frame_load_cap)
    if expected is None:
        return
    if frames_read < expected - short_read_tolerance(expected):
        raise RuntimeError(f"RelayVSR to file: short read, decoded {frames_read} of {expected} frames from "
                           f"{os.path.basename(path) or 'the video'}; the input is truncated or unreadable")


def check_frames_written(frames_written, frames_read):
    if frames_written != frames_read:
        raise RuntimeError(f"RelayVSR to file: wrote {frames_written} frames but read {frames_read}")


def vhs_frame_rate(fps):
    """VHS_LoadVideoPath's loaded_fps (with force_rate 0): 1 / (1 / fps), as VHS_VideoCombine got it."""
    if not fps or fps != fps or fps <= 0 or math.isinf(fps):
        raise RuntimeError(f"RelayVSR to file: the video reports no usable frame rate ({fps!r})")
    return 1 / (1 / fps)


# ---------------------------------------------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------------------------------------------

def ffmpeg_suitability(path):
    """VHS videohelpersuite/utils.py ffmpeg_suitability."""
    try:
        version = subprocess.run([path, "-version"], check=True, capture_output=True).stdout.decode(*ENCODE_ARGS)
    except Exception:
        return 0
    score = 0
    for criterion, points in [("libvpx", 20), ("264", 10), ("265", 3), ("svtav1", 5), ("libopus", 1)]:
        if version.find(criterion) >= 0:
            score += points
    copyright_index = version.find("2000-2")
    if copyright_index >= 0:
        copyright_year = version[copyright_index + 6:copyright_index + 9]
        if copyright_year.isnumeric():
            score += int(copyright_year)
    return score


def _loaded_vhs_ffmpeg_path(modules=None):
    """The ffmpeg_path VHS chose at startup, if VideoHelperSuite is loaded in this process."""
    for name, module in list((modules if modules is not None else sys.modules).items()):
        if module is not None and name.endswith("videohelpersuite.utils"):
            path = getattr(module, "ffmpeg_path", None)
            if path:
                return path
    return None


def discover_ffmpeg(*, environ=None, modules=None, which=shutil.which, imageio_exe=None, isfile=os.path.isfile,
                    suitability=ffmpeg_suitability):
    """The ffmpeg binary VHS_VideoCombine uses: the one VHS already picked when it is loaded, else VHS's own rules
    (VHS_FORCE_FFMPEG_PATH, imageio-ffmpeg, PATH, ./ffmpeg; the most suitable when several)."""
    environ = os.environ if environ is None else environ
    loaded = _loaded_vhs_ffmpeg_path(modules)
    if loaded:
        return loaded
    if "VHS_FORCE_FFMPEG_PATH" in environ:
        return environ["VHS_FORCE_FFMPEG_PATH"]
    paths = []
    if imageio_exe is None:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            imageio_exe = get_ffmpeg_exe()
        except Exception:
            imageio_exe = None
    if imageio_exe:
        paths.append(imageio_exe)
    if "VHS_USE_IMAGEIO_FFMPEG" in environ and imageio_exe:
        return imageio_exe
    system = which("ffmpeg")
    if system is not None:
        paths.append(system)
    for local in ("ffmpeg", "ffmpeg.exe"):
        if isfile(local):
            paths.append(os.path.abspath(local))
    if not paths:
        raise RuntimeError("RelayVSR to file: ffmpeg is required and could not be found (install imageio-ffmpeg "
                           "or put ffmpeg on PATH)")
    if len(paths) == 1:
        return paths[0]
    return max(paths, key=suitability)


def build_encode_args(*, ffmpeg, width, height, frame_rate, crf, out_path, pix_fmt="yuv420p"):
    """VHS_VideoCombine (format video/h264-mp4, 8-bit RGB input, no loop) without the metadata pass:
    rgb24 frames on stdin, tagged sRGB-primaries RGB with a bt709 transfer ("fake_trc"), libx264 to bt709 TV range."""
    return [ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-color_range", "pc", "-colorspace", "rgb", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-s", f"{width}x{height}", "-r", str(frame_rate), "-i", "-",
            "-n", "-c:v", "libx264", "-pix_fmt", pix_fmt, "-crf", str(crf),
            "-vf", "scale=out_color_matrix=bt709",
            "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            out_path]


def build_mux_args(*, ffmpeg, video_path, source_path, frames, frame_rate, out_path):
    """Copy the encoded video and add the source's first audio stream as AAC (VHS h264-mp4 audio_pass), padded with
    silence and cut to the video length (VHS: apad whole_dur = video duration + 1 s, -shortest). `1:a:0?` makes a
    source without audio produce a video-only file instead of failing."""
    min_audio_dur = frames / frame_rate + 1
    return [ffmpeg, "-v", "error", "-n", "-i", video_path, "-i", source_path,
            "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
            "-c:a", "aac", "-movflags", "use_metadata_tags",
            "-af", f"apad=whole_dur={min_audio_dur}", "-shortest", out_path]


def build_probe_args(ffmpeg, source_path):
    return [ffmpeg, "-hide_banner", "-i", source_path]


_AUDIO_STREAM = re.compile(r"^\s*Stream #\d+:\d+\S*.*?: Audio:", re.MULTILINE)


def has_audio_stream(probe_stderr):
    """Whether `ffmpeg -i <source>` lists an audio stream."""
    return bool(_AUDIO_STREAM.search(probe_stderr or ""))


def probe_has_audio(ffmpeg, source_path, run=subprocess.run):
    res = run(build_probe_args(ffmpeg, source_path), capture_output=True)
    return has_audio_stream(res.stderr.decode(*ENCODE_ARGS))


# ---------------------------------------------------------------------------------------------------------------
# Output files (VHS_VideoCombine naming)
# ---------------------------------------------------------------------------------------------------------------

def next_counter(existing_files, filename):
    """VHS_VideoCombine's counter: 1 + the largest N among `<filename>_<N><non-digits>.<ext>` files."""
    matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
    max_counter = 0
    for existing in existing_files:
        match = matcher.fullmatch(existing)
        if match:
            max_counter = max(max_counter, int(match.group(1)))
    return max_counter + 1


class OutputFiles:
    """Paths of one job's outputs, named like VHS_VideoCombine: <filename>_<counter:05>.png (first frame),
    <filename>_<counter:05>.mp4 (video), <filename>_<counter:05>-audio.mp4 (video + source audio)."""

    def __init__(self, folder, filename, counter):
        self.folder = folder
        self.filename = filename
        self.counter = counter
        base = f"{filename}_{counter:05}"
        self.png_name = f"{base}.png"
        self.video_name = f"{base}.{VIDEO_EXTENSION}"
        self.audio_name = f"{base}-audio.{VIDEO_EXTENSION}"
        self.png_path = os.path.join(folder, self.png_name)
        self.video_path = os.path.join(folder, self.video_name)
        self.audio_path = os.path.join(folder, self.audio_name)

    @classmethod
    def allocate(cls, folder, filename):
        os.makedirs(folder, exist_ok=True)
        return cls(folder, filename, next_counter(os.listdir(folder), filename))

    def remove(self, *paths):
        for path in paths or (self.png_path, self.video_path, self.audio_path):
            try:
                os.remove(path)
            except OSError:
                pass


def ui_preview(*, files, final_name, subfolder, frame_rate):
    """The VHS_VideoCombine `ui.gifs[0]` entry (save_output true): what ai-workers-service getOutputVideo reads."""
    return {
        "filename": final_name,
        "subfolder": subfolder,
        "type": "output",
        "format": VIDEO_FORMAT,
        "frame_rate": frame_rate,
        "workflow": files.png_name,
        "fullpath": os.path.join(files.folder, final_name),
    }


# ---------------------------------------------------------------------------------------------------------------
# ffmpeg subprocess
# ---------------------------------------------------------------------------------------------------------------

class FfmpegWriter:
    """One ffmpeg process fed raw frames on stdin. stderr goes to a temp file (never a pipe that could fill up and
    block the encoder). abort() kills the process and deletes the partial output."""

    def __init__(self, args, out_path, popen=subprocess.Popen):
        self.args = args
        self.out_path = out_path
        self._popen = popen
        self.proc = None
        self._stderr = None
        self.frames = 0

    def start(self):
        if os.path.exists(self.out_path):
            raise FileExistsError(f"RelayVSR to file: {self.out_path} already exists")
        self._stderr = tempfile.TemporaryFile()
        self.proc = self._popen(self.args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr)
        return self

    def _error_text(self):
        if self._stderr is None:
            return ""
        try:
            self._stderr.seek(0)
            return self._stderr.read().decode(*ENCODE_ARGS)
        except (OSError, ValueError):
            return ""

    def write(self, data, frames=1):
        try:
            self.proc.stdin.write(data)
        except (BrokenPipeError, OSError) as e:
            try:
                self.proc.wait(timeout=10)
            except Exception:
                pass
            raise RuntimeError(f"RelayVSR to file: ffmpeg stopped while encoding ({e}):\n{self._error_text()}")
        self.frames += frames

    def close(self):
        """Finish the encode; raise with ffmpeg's error output if it failed."""
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        code = self.proc.wait()
        err = self._error_text()
        self._close_stderr()
        if code != 0:
            raise RuntimeError(f"RelayVSR to file: ffmpeg exited with code {code}:\n{err}")
        if err.strip():
            print(err, end="", file=sys.stderr)
        if not os.path.isfile(self.out_path):
            raise RuntimeError(f"RelayVSR to file: ffmpeg finished but {self.out_path} is missing")

    def abort(self):
        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    self.proc.kill()
                self.proc.wait(timeout=30)
            except Exception:
                pass
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        self._close_stderr()
        try:
            os.remove(self.out_path)
        except OSError:
            pass

    def _close_stderr(self):
        if self._stderr is not None:
            try:
                self._stderr.close()
            except Exception:
                pass
            self._stderr = None


def run_checked(args, what, run=subprocess.run):
    res = run(args, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"RelayVSR to file: {what} failed (ffmpeg code {res.returncode}):\n"
                           f"{res.stderr.decode(*ENCODE_ARGS)}")
    if res.stderr:
        print(res.stderr.decode(*ENCODE_ARGS), end="", file=sys.stderr)
    return res


def encode_stream(writer, byte_batches):
    """Feed (bytes, frame_count) batches into a started FfmpegWriter and close it. On any error or interrupt
    (including one raised by the batch generator) ffmpeg is killed, the partial file deleted and the error re-raised."""
    try:
        for data, frames in byte_batches:
            writer.write(data, frames)
        writer.close()
    except BaseException:
        writer.abort()
        raise
    return writer.frames


def finish_with_audio(*, ffmpeg, files, source_path, frames, frame_rate, has_audio, run=subprocess.run):
    """After the video is encoded: mux the source audio (when it has any) into <name>-audio.mp4 and drop the
    video-only file. Returns the final file name. On failure both files are removed and the error re-raised."""
    if not has_audio:
        return files.video_name
    try:
        run_checked(build_mux_args(ffmpeg=ffmpeg, video_path=files.video_path, source_path=source_path, frames=frames,
                                   frame_rate=frame_rate, out_path=files.audio_path),
                    "muxing the source audio", run=run)
        if not os.path.isfile(files.audio_path):
            raise RuntimeError(f"RelayVSR to file: audio mux finished but {files.audio_path} is missing")
    except BaseException:
        files.remove(files.audio_path, files.video_path)
        raise
    files.remove(files.video_path)
    return files.audio_name
