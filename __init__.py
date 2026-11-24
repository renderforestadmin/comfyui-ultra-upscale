from .rtx_ultra_upscale import NODE_CLASS_MAPPINGS as _A, NODE_DISPLAY_NAME_MAPPINGS as _AD
from .detail_mix import DetailAwareMix
from .seedvr2_ultra import SeedVR2UltraUpscale

NODE_CLASS_MAPPINGS = {**_A, "DetailAwareMix": DetailAwareMix, "SeedVR2UltraUpscale": SeedVR2UltraUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {**_AD, "DetailAwareMix": "Detail-Aware Mix 🎯", "SeedVR2UltraUpscale": "SeedVR2 Ultra Upscale 🎬"}

# RelayVSR Upscale (native) needs a ComfyUI with the core RelayVSR support (comfy.ldm.wan.model_vsr);
# on an older ComfyUI it is skipped so the other nodes still load.
try:
    from .relayvsr_native import NODE_CLASS_MAPPINGS as _R, NODE_DISPLAY_NAME_MAPPINGS as _RD
    NODE_CLASS_MAPPINGS.update(_R)
    NODE_DISPLAY_NAME_MAPPINGS.update(_RD)
except ImportError as e:
    print(f"[comfyui-ultra-upscale] RelayVSR Upscale (native) not loaded: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
