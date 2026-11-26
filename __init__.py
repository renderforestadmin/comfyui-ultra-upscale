import importlib

from .rtx_ultra_upscale import NODE_CLASS_MAPPINGS as _A, NODE_DISPLAY_NAME_MAPPINGS as _AD
from .detail_mix import DetailAwareMix
from .seedvr2_ultra import SeedVR2UltraUpscale

NODE_CLASS_MAPPINGS = {**_A, "DetailAwareMix": DetailAwareMix, "SeedVR2UltraUpscale": SeedVR2UltraUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {**_AD, "DetailAwareMix": "Detail-Aware Mix 🎯", "SeedVR2UltraUpscale": "SeedVR2 Ultra Upscale 🎬"}


def _register_optional(module_name):
    """Import a sibling module and merge its mappings; any failure (missing file, missing dependency, syntax or
    runtime error at import) skips only that module's nodes so the rest of the pack still loads."""
    try:
        module = importlib.import_module(f".{module_name}", __name__)
        class_mappings = module.NODE_CLASS_MAPPINGS
        display_mappings = module.NODE_DISPLAY_NAME_MAPPINGS
    except Exception as e:
        print(f"[comfyui-ultra-upscale] {module_name} not loaded: {type(e).__name__}: {e}")
        return
    NODE_CLASS_MAPPINGS.update(class_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(display_mappings)


# RF Download Video: stdlib only.
_register_optional("rf_download")
# RelayVSR Upscale (native) needs a ComfyUI with the core RelayVSR support (comfy.ldm.wan.model_vsr).
_register_optional("relayvsr_native")
# RelayVSR Upscale (to file) streams a local video through the native node's code into an mp4 (needs the native node
# above, cv2 and ffmpeg).
_register_optional("relayvsr_to_file")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
