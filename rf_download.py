"""RF Download Video: fetch the job's source video to a local file before the loader reads it.

VHS_LoadVideoPath streams an http(s) `video` straight from the server for the whole job (cv2 / ffmpeg keep the
connection open across meta batches). A long idle gap between batches lets the storage server reset the stream, and
the loader then treats the reset as the end of the video: the output is silently truncated. Downloading the file
first (with retries, HTTP Range resume and a Content-Length check) turns that into either a complete local file or a
clear error.

The download logic is plain Python (stdlib only, no ComfyUI imports at module level) so it can be unit tested
without ComfyUI; the node class imports folder_paths / comfy.utils lazily.
"""

import hashlib
import http.client
import logging
import os
import posixpath
import time
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger("comfyui-ultra-upscale.rf_download")

VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".mkv", ".gif", ".m4v", ".avi")
DEFAULT_EXTENSION = ".mp4"
DOWNLOAD_SUBDIR = "rf_downloads"
CHUNK_SIZE = 1024 * 1024
STALE_HOURS = 6
MAX_ATTEMPTS = 200


class DownloadError(RuntimeError):
    pass


def is_http_url(value):
    return isinstance(value, str) and value.strip().lower().startswith(("http://", "https://"))


def safe_url(url):
    """Host + path only: never log or raise with a signed query string."""
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.hostname or ''}{parts.path}"


def local_name_for(url):
    """Stable file name for a URL: hash of the URL without its query/fragment + the URL's video extension."""
    parts = urllib.parse.urlsplit(url.strip())
    key = f"{parts.scheme.lower()}://{(parts.netloc or '').lower()}{parts.path}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    ext = posixpath.splitext(urllib.parse.unquote(parts.path))[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        ext = DEFAULT_EXTENSION
    return f"rf_{digest}{ext}"


def local_path_for(url, dest_dir):
    return os.path.join(dest_dir, local_name_for(url))


def cleanup_stale(dest_dir, *, max_age_hours=STALE_HOURS, keep=(), now=None):
    """Delete our downloads (complete or partial) older than max_age_hours; never the paths in `keep`."""
    if not os.path.isdir(dest_dir):
        return []
    now = time.time() if now is None else now
    keep = {os.path.abspath(p) for p in keep}
    removed = []
    for name in os.listdir(dest_dir):
        if not name.startswith("rf_"):
            continue
        path = os.path.join(dest_dir, name)
        if os.path.abspath(path) in keep or not os.path.isfile(path):
            continue
        try:
            if now - os.path.getmtime(path) > max_age_hours * 3600:
                os.remove(path)
                removed.append(path)
        except OSError as e:
            logger.warning("[RFDownloadVideo] could not remove stale %s: %s", path, e)
    return removed


def _parse_total_from_content_range(value):
    # "bytes 100-199/1000" -> 1000 ; "bytes 100-199/*" -> None
    if not value or "/" not in value:
        return None
    total = value.rsplit("/", 1)[1].strip()
    return int(total) if total.isdigit() else None


def _attempt(url, part_path, state, *, timeout, progress):
    """One HTTP request, appending to part_path from state["offset"]; keeps state["total"] (expected size) current."""
    offset = state["offset"]
    expected_total = state["total"]
    headers = {"User-Agent": "comfyui-rf-download/1.0", "Accept-Encoding": "identity"}
    if offset > 0:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", None) or response.getcode()
        length = response.headers.get("Content-Length")
        length = int(length) if length and length.isdigit() else None
        if offset > 0 and status == 206:
            total = _parse_total_from_content_range(response.headers.get("Content-Range"))
            start = response.headers.get("Content-Range", "").split(" ", 1)[-1].split("-", 1)[0].strip()
            if start.isdigit() and int(start) != offset:
                raise DownloadError(f"server resumed at byte {start}, expected {offset}")
            mode = "ab"
        else:
            # Fresh download, or the server ignored Range (200): start over.
            total = length
            offset = 0
            mode = "wb"
        if expected_total is not None and total is not None and total != expected_total:
            raise DownloadError(f"size changed between attempts ({expected_total} -> {total} bytes)")
        expected_total = total if total is not None else expected_total
        state["total"] = expected_total
        state["offset"] = offset
        written = offset
        with open(part_path, mode) as f:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
                state["offset"] = written
                if progress is not None:
                    progress(written, expected_total)


def download_video(url, dest_dir, *, timeout=60.0, retries=5, backoff=2.0, stale_hours=STALE_HOURS, progress=None):
    """Download `url` to dest_dir and return the local path.

    - non-http(s) values are returned unchanged (local paths keep working);
    - a complete file for the same URL (query string ignored) is reused, not downloaded again;
    - streams to `<name>.<pid>.part`, resumes with HTTP Range after a dropped connection (gives up after `retries`
      consecutive attempts that added no bytes), checks the final size against Content-Length / Content-Range and renames atomically;
    - raises DownloadError (a RuntimeError) on HTTP errors, timeouts or a size mismatch, so the job fails instead of
      running on a truncated file.
    """
    if not is_http_url(url):
        return url
    url = url.strip()
    os.makedirs(dest_dir, exist_ok=True)
    final_path = local_path_for(url, dest_dir)
    part_path = f"{final_path}.{os.getpid()}.part"
    where = safe_url(url)

    cleanup_stale(dest_dir, max_age_hours=stale_hours, keep=(final_path, part_path))

    if os.path.isfile(final_path):
        os.utime(final_path, None)  # keep it fresh for the stale cleanup while it is in use
        logger.info("[RFDownloadVideo] reusing %s for %s", final_path, where)
        return final_path

    if os.path.exists(part_path):
        os.remove(part_path)

    logger.info("[RFDownloadVideo] downloading %s -> %s", where, final_path)
    started = time.time()
    state = {"offset": 0, "total": None}
    last_error = None
    failures = 0  # consecutive attempts that added no bytes; an attempt that made progress does not use a retry
    attempts = 0
    while True:
        if attempts > 0:
            if failures > retries or attempts >= MAX_ATTEMPTS:
                _remove_quietly(part_path)
                raise DownloadError(f"[RFDownloadVideo] download of {where} failed after {attempts} attempts "
                                    f"({failures} without progress): {last_error}")
            delay = min(backoff * (2 ** max(failures - 1, 0)), 30.0) if failures else 0.0
            logger.warning("[RFDownloadVideo] retry for %s at byte %d (%d/%d without progress) in %.1fs: %s",
                           where, state["offset"], failures, retries, delay, last_error)
            if delay:
                time.sleep(delay)
        attempts += 1
        before = state["offset"]
        try:
            _attempt(url, part_path, state, timeout=timeout, progress=progress)
        except urllib.error.HTTPError as e:
            if e.code == 416 and state["total"] is not None and state["offset"] == state["total"]:
                break  # already have every byte
            last_error = f"HTTP {e.code} {e.reason}"
            if 400 <= e.code < 500 and e.code not in (408, 429):
                _remove_quietly(part_path)
                raise DownloadError(f"[RFDownloadVideo] {last_error} for {where}") from None
            failures += 1
            continue
        except DownloadError as e:
            last_error = str(e)
            state = {"offset": 0, "total": None}
            _remove_quietly(part_path)
            failures += 1
            continue
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as e:
            # Connection reset / timeout / truncated body: keep what is on disk and resume from there.
            last_error = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
            state["offset"] = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            failures = 0 if state["offset"] > before else failures + 1
            continue

        offset, expected_total = state["offset"], state["total"]
        if expected_total is None or offset == expected_total:
            if expected_total is None:
                logger.warning("[RFDownloadVideo] %s sent no Content-Length; size not verified (%d bytes)",
                               where, offset)
            break
        if offset > expected_total:
            last_error = f"got {offset} bytes, more than Content-Length {expected_total}"
            state = {"offset": 0, "total": None}
            _remove_quietly(part_path)
            failures += 1
            continue
        # Clean EOF before Content-Length: the stream was cut, resume.
        last_error = f"stream ended at {offset} of {expected_total} bytes"
        failures = 0 if offset > before else failures + 1

    if not os.path.exists(part_path):
        raise DownloadError(f"[RFDownloadVideo] download of {where} produced no file")
    size = os.path.getsize(part_path)
    expected_total = state["total"]
    if expected_total is not None and size != expected_total:
        _remove_quietly(part_path)
        raise DownloadError(f"[RFDownloadVideo] size mismatch for {where}: {size} != Content-Length {expected_total}")
    if size == 0:
        _remove_quietly(part_path)
        raise DownloadError(f"[RFDownloadVideo] {where} is empty")
    os.replace(part_path, final_path)
    logger.info("[RFDownloadVideo] downloaded %s (%d bytes) in %.1fs", where, size, time.time() - started)
    return final_path


def _remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _download_dir():
    import folder_paths
    return os.path.join(folder_paths.get_temp_directory(), DOWNLOAD_SUBDIR)


class RFDownloadVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "placeholder": "https://... or a local path"}),
            },
            "optional": {
                "timeout_seconds": ("INT", {"default": 60, "min": 5, "max": 3600, "step": 1}),
                "retries": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1}),
            },
        }

    # STRING is the type of VHS_LoadVideoPath's `video` input, so the path can be linked into it.
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "download"
    CATEGORY = "Renderforest"

    def download(self, url, timeout_seconds=60, retries=5):
        if not is_http_url(url):
            return (url,)
        pbar = None
        state = {"total": None}

        def progress(done, total):
            nonlocal pbar
            if total is None:
                return
            if pbar is None or state["total"] != total:
                import comfy.utils
                pbar = comfy.utils.ProgressBar(total)
                state["total"] = total
            pbar.update_absolute(done, total)

        path = download_video(url, _download_dir(), timeout=float(timeout_seconds), retries=int(retries),
                              progress=progress)
        return (path,)

    @classmethod
    def IS_CHANGED(cls, url, **kwargs):
        # Stable for the same URL once its file is on disk, so meta-batch requeues and later prompts reuse the
        # cached output. If the file is missing (never downloaded, or removed by the stale cleanup), return a fresh
        # NaN (never equal to anything) so a cached path to a deleted file is not reused.
        if not is_http_url(url):
            return url
        try:
            if os.path.isfile(local_path_for(url, _download_dir())):
                return local_name_for(url)
        except Exception:
            pass
        return float("nan")


NODE_CLASS_MAPPINGS = {"RFDownloadVideo": RFDownloadVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"RFDownloadVideo": "RF Download Video (to local file)"}
