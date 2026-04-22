"""Convert Megatron SFT checkpoint to HuggingFace format."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3rdparty" / "Megatron-LM"))

# Patch transformers 4.56 compatibility
import transformers.image_utils as _img_utils

if not hasattr(_img_utils, "pil_torch_interpolation_mapping"):
    _img_utils.pil_torch_interpolation_mapping = {}

from megatron.bridge import AutoBridge

MEGATRON_PATH = "/root/Megatron-Bridge/nemo_experiments/default/checkpoints_simpo_full/iter_0000200"
HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
HF_EXPORT_PATH = "/root/Megatron-Bridge/exports/nemotron_nano_v2_vl_simpo"

bridge = AutoBridge.from_auto_config(MEGATRON_PATH, HF_MODEL_ID, trust_remote_code=True)
bridge.export_ckpt(
    megatron_path=MEGATRON_PATH,
    hf_path=HF_EXPORT_PATH,
    show_progress=True,
)
print(f"Export done -> {HF_EXPORT_PATH}")
