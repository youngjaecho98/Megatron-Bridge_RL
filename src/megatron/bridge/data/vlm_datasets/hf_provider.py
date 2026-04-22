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

"""
Provider that builds conversation datasets from HuggingFace datasets.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
from transformers import AutoProcessor

from megatron.bridge.data.vlm_datasets.conversation_dataset import VLMConversationDataset
from megatron.bridge.data.vlm_datasets.hf_dataset_makers import (
    make_cord_v2_dataset,
    make_cv17_dataset,
    make_default_audio_dataset,
    make_llava_video_178k_dataset,
    make_medpix_dataset,
    make_raven_dataset,
    make_rdr_dataset,
    make_rlaifv_preference_dataset,
    make_safewatch_dataset,
    make_safewatch_dpo_dataset,
    make_safewatch_sft_dataset,
)
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider


@dataclass(kw_only=True)
class HFDatasetConversationProvider(DatasetProvider):
    """DatasetProvider that builds VLM conversation datasets from HF datasets.

    This provider leverages simple maker functions that return lists of examples
    with a "conversation" schema understood by model processors. It binds a
    HuggingFace `AutoProcessor` for the specified model and selects an
    appropriate collate function for batching.
    """

    # Required to match model.seq_length (enforced by ConfigContainer.validate)
    seq_length: int

    # HF processor/model identifier (e.g., "Qwen/Qwen2.5-VL-3B-Instruct")
    hf_processor_path: str

    # Select which maker to use. Must match a function defined in makers module
    # like `make_rdr_dataset`, `make_cord_v2_dataset`, `make_medpix_dataset`, `make_cv17_dataset`.
    maker_name: str

    # Optional parameters forwarded to the selected maker (used for train split by default)
    maker_kwargs: Optional[Dict[str, Any]] = None

    # Per-split overrides: merged on top of maker_kwargs when building that split.
    # This allows different subset/split/prompt per data split (e.g. aishell "dev" vs "train").
    val_maker_kwargs: Optional[Dict[str, Any]] = None
    test_maker_kwargs: Optional[Dict[str, Any]] = None

    # Skip building specific splits (returns None for that split)
    skip_test: bool = False

    # Optional collate override. If None, inferred from processor type.
    collate_impl: Optional[Callable[[list, Any], Dict[str, torch.Tensor]]] = None

    # Keep parity with GPTDatasetConfig usage in batching utilities
    skip_getting_attention_mask_from_dataset: bool = True

    # DataloaderConfig fields are inherited (num_workers, dataloader_type, etc.)
    dataloader_type: Optional[Literal["single", "cyclic", "external"]] = "single"

    # Enable batch-level online sequence packing (dataset-level packing is available in FinetuneDatasetProvider)
    pack_sequences_in_batch: bool = False

    def _get_maker(self) -> Callable[..., List[Dict[str, Any]]]:
        registry: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
            "make_rdr_dataset": make_rdr_dataset,
            "make_cord_v2_dataset": make_cord_v2_dataset,
            "make_medpix_dataset": make_medpix_dataset,
            "make_cv17_dataset": make_cv17_dataset,
            "make_raven_dataset": make_raven_dataset,
            "make_llava_video_178k_dataset": make_llava_video_178k_dataset,
            "make_default_audio_dataset": make_default_audio_dataset,
            "make_safewatch_dataset": make_safewatch_dataset,
            "make_safewatch_sft_dataset": make_safewatch_sft_dataset,
        }
        if self.maker_name in registry:
            return registry[self.maker_name]
        alias_map = {
            "rdr": "make_rdr_dataset",
            "cord_v2": "make_cord_v2_dataset",
            "medpix": "make_medpix_dataset",
            "cv17": "make_cv17_dataset",
            "raven": "make_raven_dataset",
            "llava_video_178k": "make_llava_video_178k_dataset",
            "default_audio": "make_default_audio_dataset",
            "safewatch": "make_safewatch_dataset",
            "safewatch_sft": "make_safewatch_sft_dataset",
        }
        if self.maker_name in alias_map and alias_map[self.maker_name] in registry:
            return registry[alias_map[self.maker_name]]
        raise ValueError(f"Unknown maker_name: {self.maker_name}")

    def _build_split_dataset(
        self,
        split: str,
        target_length: int,
        processor: Any,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[VLMConversationDataset]:
        if target_length <= 0:
            return None
        maker = self._get_maker()
        kwargs = dict(self.maker_kwargs or {})
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        kwargs.setdefault("split", split)
        base_examples = maker(**kwargs)  # type: ignore[misc]
        if not isinstance(base_examples, list) or len(base_examples) == 0:
            raise ValueError(f"Maker '{self.maker_name}' returned no examples for split='{split}'")
        return VLMConversationDataset(
            base_examples=base_examples,
            target_length=target_length,
            processor=processor,
            collate_impl=self.collate_impl,
        )

    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        # Bind processor for the requested model
        processor = AutoProcessor.from_pretrained(
            self.hf_processor_path,
            trust_remote_code=is_safe_repo(
                trust_remote_code=self.trust_remote_code,
                hf_path=self.hf_processor_path,
            ),
        )

        train_ds = self._build_split_dataset("train", context.train_samples, processor)
        valid_ds = self._build_split_dataset("validation", context.valid_samples, processor, self.val_maker_kwargs)
        test_ds = (
            None
            if self.skip_test
            else self._build_split_dataset("test", context.test_samples, processor, self.test_maker_kwargs)
        )

        return train_ds, valid_ds, test_ds


@dataclass(kw_only=True)
class HFPreferencePairProvider(DatasetProvider):
    """DatasetProvider for VLM preference-pair datasets (e.g. RLAIF-V).

    Each example returned by the maker function must contain *two*
    conversation variants:

    * ``"chosen_conversation"`` — the preferred response
    * ``"rejected_conversation"`` — the dispreferred response

    The underlying :class:`VLMPreferencePairDataset` emits both variants
    per ``__getitem__`` call and the collate function stacks them as
    ``[chosen_0, …, chosen_N, rejected_0, …, rejected_N]`` along the
    batch dimension so that the SimPO forward step can split them by half.
    """

    seq_length: int
    hf_processor_path: str
    maker_name: str
    maker_kwargs: Optional[Dict[str, Any]] = None
    skip_test: bool = False
    collate_impl: Optional[Callable[[list, Any], Dict[str, torch.Tensor]]] = None
    skip_getting_attention_mask_from_dataset: bool = True
    dataloader_type: Optional[Literal["single", "cyclic", "external"]] = "single"

    def _get_maker(self) -> Callable[..., List[Dict[str, Any]]]:
        registry: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
            "make_rlaifv_preference_dataset": make_rlaifv_preference_dataset,
            "make_safewatch_dpo_dataset": make_safewatch_dpo_dataset,
        }
        if self.maker_name in registry:
            return registry[self.maker_name]
        alias_map = {
            "rlaifv": "make_rlaifv_preference_dataset",
            "safewatch": "make_safewatch_dpo_dataset",
        }
        if self.maker_name in alias_map and alias_map[self.maker_name] in registry:
            return registry[alias_map[self.maker_name]]
        raise ValueError(f"Unknown preference maker_name: {self.maker_name}")

    def _build_split_dataset(
        self,
        split: str,
        target_length: int,
        processor: Any,
    ) -> Optional["VLMPreferencePairDataset"]:
        if target_length <= 0:
            return None
        maker = self._get_maker()
        kwargs = dict(self.maker_kwargs or {})
        kwargs.setdefault("split", split)
        base_examples = maker(**kwargs)
        if not isinstance(base_examples, list) or len(base_examples) == 0:
            raise ValueError(f"Maker '{self.maker_name}' returned no examples for split='{split}'")
        return VLMPreferencePairDataset(
            base_examples=base_examples,
            target_length=target_length,
            processor=processor,
            collate_impl=self.collate_impl,
        )

    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Build preference-pair datasets for train / validation / test."""
        processor = AutoProcessor.from_pretrained(
            self.hf_processor_path,
            trust_remote_code=is_safe_repo(
                trust_remote_code=self.trust_remote_code,
                hf_path=self.hf_processor_path,
            ),
        )

        train_ds = self._build_split_dataset("train", context.train_samples, processor)
        valid_ds = self._build_split_dataset("validation", context.valid_samples, processor)
        test_ds = None if self.skip_test else self._build_split_dataset("test", context.test_samples, processor)

        return train_ds, valid_ds, test_ds


class VLMPreferencePairDataset(torch.utils.data.Dataset):
    """Dataset that yields chosen/rejected conversation pairs.

    Each ``__getitem__`` returns a dict with ``"chosen_conversation"`` and
    ``"rejected_conversation"`` keys.  The collate function processes both
    and concatenates them as ``[chosen | rejected]`` along the batch axis.
    """

    def __init__(
        self,
        base_examples: List[Dict[str, Any]],
        target_length: int,
        processor: Any,
        collate_impl: Optional[Callable[[list, Any], Dict[str, torch.Tensor]]] = None,
    ) -> None:
        assert isinstance(base_examples, list) and len(base_examples) > 0
        self._base_examples = base_examples
        self._length = int(max(0, target_length))
        self._processor = processor

        collate_key = type(processor).__name__ if processor is not None else "default"
        from megatron.bridge.data.vlm_datasets.collate import COLLATE_FNS

        base_collate = collate_impl or COLLATE_FNS.get(collate_key, COLLATE_FNS["default"])

        def _merge_visual_inputs(chosen_vi, rejected_vi):
            """Merge two visual input containers by concatenating their tensors."""
            if chosen_vi is None and rejected_vi is None:
                return None
            if chosen_vi is None:
                return rejected_vi
            if rejected_vi is None:
                return chosen_vi
            merged_kwargs = {}
            for attr_name in vars(chosen_vi):
                c_t = getattr(chosen_vi, attr_name)
                r_t = getattr(rejected_vi, attr_name, None)
                if c_t is None and r_t is None:
                    continue
                if c_t is not None and r_t is not None and isinstance(c_t, torch.Tensor):
                    merged_kwargs[attr_name] = torch.cat([c_t, r_t], dim=0)
                elif c_t is not None:
                    merged_kwargs[attr_name] = c_t
            return type(chosen_vi)(**merged_kwargs)

        def _preference_collate(batch: list) -> Dict[str, torch.Tensor]:
            chosen_batch = [{"conversation": ex["chosen_conversation"]} for ex in batch]
            rejected_batch = [{"conversation": ex["rejected_conversation"]} for ex in batch]

            chosen_out = base_collate(chosen_batch, self._processor)
            rejected_out = base_collate(rejected_batch, self._processor)

            merged: Dict[str, torch.Tensor] = {}
            for key in chosen_out:
                c_val = chosen_out[key]
                r_val = rejected_out.get(key)
                if key == "visual_inputs":
                    merged[key] = _merge_visual_inputs(c_val, r_val)
                elif c_val is None or r_val is None:
                    merged[key] = c_val
                elif isinstance(c_val, torch.Tensor) and isinstance(r_val, torch.Tensor):
                    if c_val.shape[1:] != r_val.shape[1:]:
                        max_sizes = [max(s1, s2) for s1, s2 in zip(c_val.shape[1:], r_val.shape[1:])]
                        c_padded = torch.zeros(c_val.shape[0], *max_sizes, dtype=c_val.dtype, device=c_val.device)
                        r_padded = torch.zeros(r_val.shape[0], *max_sizes, dtype=r_val.dtype, device=r_val.device)
                        c_padded[tuple(slice(0, s) for s in c_val.shape)] = c_val
                        r_padded[tuple(slice(0, s) for s in r_val.shape)] = r_val
                        merged[key] = torch.cat([c_padded, r_padded], dim=0)
                    else:
                        merged[key] = torch.cat([c_val, r_val], dim=0)
                else:
                    merged[key] = c_val
            return merged

        self.collate_fn = _preference_collate

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._length == 0:
            raise IndexError("Empty dataset")
        return self._base_examples[idx % len(self._base_examples)]
