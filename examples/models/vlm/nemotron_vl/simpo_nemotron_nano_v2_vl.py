#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
SimPO (Simple Preference Optimization) training for Nemotron Nano V2 VL.

SimPO is a reference-model-free preference optimisation method that uses
length-normalised average log-probability as the implicit reward.  This
eliminates the need for a separate reference forward pass, halving the
memory and compute cost compared to DPO.

Reference: https://arxiv.org/abs/2405.14734

Usage (single node, 8 GPUs, RLAIF-V image dataset)::

    python -m torch.distributed.run --nproc_per_node=8 \\
        examples/models/vlm/nemotron_vl/simpo_nemotron_nano_v2_vl.py \\
        --pretrained-checkpoint /path/to/sft/checkpoint

Usage (SafeWatch video DPO dataset)::

    python -m torch.distributed.run --nproc_per_node=4 \\
        examples/models/vlm/nemotron_vl/simpo_nemotron_nano_v2_vl.py \\
        --pretrained-checkpoint /path/to/sft/checkpoint \\
        --train-data /path/to/dpo_train.jsonl \\
        --eval-data /path/to/dpo_eval.jsonl

Quick test with limited data::

    python -m torch.distributed.run --nproc_per_node=4 \\
        examples/models/vlm/nemotron_vl/simpo_nemotron_nano_v2_vl.py \\
        --pretrained-checkpoint /path/to/sft/checkpoint \\
        --train-data /path/to/dpo_train.jsonl \\
        --max-preference-samples 200
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.simpo_step import make_simpo_forward_step
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe


logger: logging.Logger = logging.getLogger(__name__)

SCRIPT_DIR: Path = Path(__file__).parent.resolve()
DEFAULT_CONFIG_FILENAME: str = "nemotron_nano_v2_vl_simpo.yaml"
DEFAULT_CONFIG_FILE_PATH: Path = SCRIPT_DIR / "conf" / DEFAULT_CONFIG_FILENAME


def parse_cli_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command-line flags and return ``(namespace, hydra_overrides)``."""
    parser = argparse.ArgumentParser(
        description="SimPO training for Nemotron Nano V2 VL",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--config-file",
        type=str,
        default=str(DEFAULT_CONFIG_FILE_PATH),
        help="Path to the YAML OmegaConf override file.",
    )
    parser.add_argument(
        "--hf-model-path",
        type=str,
        default="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        help="HuggingFace model identifier for the model architecture.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        required=True,
        help="Path to a Megatron-Bridge checkpoint or HuggingFace model to initialise from.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="SimPO temperature / scaling factor (default: 2.0).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="SimPO target reward margin (default: 0.5).",
    )
    parser.add_argument(
        "--max-preference-samples",
        type=int,
        default=None,
        help="Cap on the number of preference pairs to load (useful for quick tests).",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to a JSONL file for training (e.g. SafeWatch DPO). "
        "When provided, uses the SafeWatch maker instead of RLAIF-V.",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to a JSONL file for evaluation.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def _build_simpo_config(
    *,
    hf_model_path: str,
    pretrained_checkpoint: str,
    max_preference_samples: int | None = None,
    train_data: str | None = None,
    eval_data: str | None = None,
) -> ConfigContainer:
    """Build a SimPO-ready config starting from the Nemotron VL SFT recipe.

    Key differences from the vanilla SFT config:

    * ``cross_entropy_loss_fusion`` is disabled so the model returns raw
      logits rather than fused per-token losses.
    * Lower learning rate suitable for preference tuning.
    * ``micro_batch_size`` represents the number of *pairs* — the actual
      batch fed to the model is ``2 * micro_batch_size``.
    * Uses :class:`HFPreferencePairProvider` with RLAIF-V or SafeWatch DPO.

    Args:
        hf_model_path: HuggingFace model identifier.
        pretrained_checkpoint: Path to pretrained checkpoint.
        max_preference_samples: Optional cap on the number of preference
            pairs to load from the dataset.
        train_data: Path to a JSONL training file (SafeWatch DPO).
        eval_data: Path to a JSONL evaluation file (SafeWatch DPO).

    Returns:
        A fully populated :class:`ConfigContainer`.
    """
    from megatron.bridge.recipes.nemotron_vl.nemotron_nano_v2_vl import nemotron_nano_v2_vl_12b_sft_config

    cfg = nemotron_nano_v2_vl_12b_sft_config()

    # SimPO requires raw logits — disable fused cross-entropy
    cfg.model.cross_entropy_loss_fusion = False

    from megatron.bridge import AutoBridge

    cfg.model = AutoBridge.from_hf_pretrained(hf_model_path, trust_remote_code=True).to_megatron_provider(
        load_weights=False
    )
    cfg.model.seq_length = 4096
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.cross_entropy_fusion_impl = "native"

    # Parallelism
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # VLM freeze settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = False

    # TE / transformer
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "flash"

    # Checkpoint
    cfg.checkpoint.pretrained_checkpoint = pretrained_checkpoint

    # Training — preference tuning typically needs fewer iterations and lower LR
    cfg.train.train_iters = 500
    cfg.train.global_batch_size = 16
    cfg.train.micro_batch_size = 1

    # Preference-pair dataset
    from megatron.bridge.data.vlm_datasets.hf_provider import HFPreferencePairProvider

    if train_data is not None:
        maker_kwargs: dict[str, Any] = {
            "train_path": train_data,
            "eval_path": eval_data,
        }
        if max_preference_samples is not None:
            maker_kwargs["max_samples"] = max_preference_samples

        cfg.dataset = HFPreferencePairProvider(
            seq_length=4096,
            hf_processor_path=hf_model_path,
            maker_name="make_safewatch_dpo_dataset",
            maker_kwargs=maker_kwargs,
            skip_test=True,
        )

        if eval_data is not None:
            cfg.validation.eval_iters = 10
            cfg.validation.eval_interval = 100
        else:
            cfg.validation.eval_iters = 0
            cfg.validation.eval_interval = 0
    else:
        maker_kwargs_rlaif: dict[str, int] = {}
        if max_preference_samples is not None:
            maker_kwargs_rlaif["max_samples"] = max_preference_samples

        cfg.dataset = HFPreferencePairProvider(
            seq_length=4096,
            hf_processor_path=hf_model_path,
            maker_name="make_rlaifv_preference_dataset",
            maker_kwargs=maker_kwargs_rlaif or None,
            skip_test=True,
        )

        cfg.validation.eval_iters = 0
        cfg.validation.eval_interval = 0

    return cfg


def main() -> None:
    """Entry point for SimPO training."""
    args, cli_overrides = parse_cli_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Megatron-Bridge Nemotron Nano V2 VL SimPO Training")
    logger.info("---------------------------------------------------")

    cfg = _build_simpo_config(
        hf_model_path=args.hf_model_path,
        pretrained_checkpoint=args.pretrained_checkpoint,
        max_preference_samples=args.max_preference_samples,
        train_data=args.train_data,
        eval_data=args.eval_data,
    )

    if get_rank_safe() == 0:
        cfg.print_yaml()

    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    if args.config_file and os.path.exists(args.config_file):
        logger.debug("Loading YAML overrides from: %s", args.config_file)
        yaml_overrides_omega = OmegaConf.load(args.config_file)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)

    if cli_overrides:
        logger.debug("Applying Hydra-style command-line overrides: %s", cli_overrides)
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)

    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    if get_rank_safe() == 0:
        logger.info("--- Final Merged Configuration (SimPO) ---")
        cfg.print_yaml()
        logger.info("-------------------------------------------")
        logger.info("SimPO hyper-parameters: beta=%.2f, gamma=%.2f", args.beta, args.gamma)

    simpo_forward_step = make_simpo_forward_step(beta=args.beta, gamma=args.gamma)

    finetune(config=cfg, forward_step_func=simpo_forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
