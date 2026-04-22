# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings as _warnings

# Core conversion infrastructure (must always succeed)
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.t5_provider import T5ModelProvider

# Bridge registrations — each wrapped so missing HF classes don't block startup
def _safe_import(module_path: str) -> None:
    """Import a bridge module, suppressing errors from missing HF model classes."""
    try:
        __import__(module_path, fromlist=["_"])
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        _warnings.warn(f"Skipping bridge registration from {module_path}: {exc}", stacklevel=2)

_safe_import("megatron.bridge.models.bailing")
_safe_import("megatron.bridge.models.deepseek")
_safe_import("megatron.bridge.models.gemma")
_safe_import("megatron.bridge.models.gemma_vl")
_safe_import("megatron.bridge.models.glm")
_safe_import("megatron.bridge.models.glm_vl")
_safe_import("megatron.bridge.models.gpt_oss")
_safe_import("megatron.bridge.models.kimi")
_safe_import("megatron.bridge.models.kimi_vl")
_safe_import("megatron.bridge.models.llama")
_safe_import("megatron.bridge.models.llama_nemotron")
_safe_import("megatron.bridge.models.mamba.mamba_provider")
_safe_import("megatron.bridge.models.mimo.mimo_bridge")
_safe_import("megatron.bridge.models.minimax_m2")
_safe_import("megatron.bridge.models.ministral3")
_safe_import("megatron.bridge.models.mistral")
_safe_import("megatron.bridge.models.nemotron")
_safe_import("megatron.bridge.models.nemotron_vl")
_safe_import("megatron.bridge.models.nemotronh")
_safe_import("megatron.bridge.models.olmoe")
_safe_import("megatron.bridge.models.qwen")
_safe_import("megatron.bridge.models.qwen3_asr")
_safe_import("megatron.bridge.models.qwen_audio")
_safe_import("megatron.bridge.models.qwen_omni")
_safe_import("megatron.bridge.models.qwen_vl")
_safe_import("megatron.bridge.models.sarvam")
_safe_import("megatron.bridge.models.t5_provider")
_safe_import("megatron.bridge.models.hf_pretrained")

__all__ = [
    "AutoBridge",
    "MegatronMappingRegistry",
    "MegatronModelBridge",
    "GPTModelProvider",
]
