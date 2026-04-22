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

"""Nemotron Nano V2 VL finetuning recipes.

This module provides SFT, PEFT, and finetune configurations for Nemotron Nano V2 VL 12B.
"""

import os
from typing import Optional

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.data.vlm_datasets import HFDatasetConversationProvider
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora import VLMLoRA
from megatron.bridge.recipes.common import _peft_common_vlm, _sft_common_vlm
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)


_HF_PATH = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"


def _common_model_settings(cfg: ConfigContainer, hf_path: str, tp: int = 4) -> None:
    cfg.model = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = tp
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "flash"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None


def _common_train_opt_ddp(cfg: ConfigContainer, hf_path: str) -> None:
    cfg.train.train_iters = 2000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 0

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=5, lr_decay_iters=None, max_lr=2e-5, min_lr=2e-6,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = False
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.checkpoint.save_interval = 200
    cfg.mixed_precision = "bf16_mixed"


# =============================================================================
# Nemotron Nano V2 VL 12B SFT Configuration
# =============================================================================
def nemotron_nano_v2_vl_12b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Nemotron Nano V2 VL 12B."""
    cfg = _sft_common_vlm()
    _common_model_settings(cfg, _HF_PATH, tp=4)
    _common_train_opt_ddp(cfg, _HF_PATH)
    return cfg


# =============================================================================
# Nemotron Nano V2 VL 12B PEFT Configuration
# =============================================================================
def nemotron_nano_v2_vl_12b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Nemotron Nano V2 VL 12B."""
    cfg = _peft_common_vlm()

    if isinstance(peft_scheme, str) and peft_scheme.lower() == "lora":
        cfg.peft = VLMLoRA(
            target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"], dim=16, alpha=32,
        )
    elif isinstance(peft_scheme, str) and peft_scheme.lower() == "dora":
        cfg.peft = VLMLoRA(
            target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"], dim=16, alpha=32, dora=True,
        )
    else:
        cfg.peft = peft_scheme

    _common_model_settings(cfg, _HF_PATH, tp=2)
    _common_train_opt_ddp(cfg, _HF_PATH)
    return cfg


# =============================================================================
# Nemotron Nano V2 VL 12B Finetune Configuration (from pretrained checkpoint)
# =============================================================================
def nemotron_nano_v2_vl_12b_finetune_config(
    *,
    hf_model_path: str = _HF_PATH,
    pretrained_checkpoint: str = "",
    lora_on_language_model: bool = False,
    lora_on_vision_model: bool = False,
    save_checkpoint_dir: Optional[str] = None,
    tensor_parallelism: int = 4,
    seq_length: int = 4096,
    train_iters: int = 10_000,
    global_batch_size: int = 32,
    micro_batch_size: int = 1,
    lr: float = 1e-5,
    min_lr: float = 1e-6,
    lr_warmup_iters: int = 500,
    save_interval: int = 200,
    dataset_maker_name: str = "make_cord_v2_dataset",
) -> ConfigContainer:
    """Create a finetuning config for Nemotron Nano V2 VL from a pretrained checkpoint."""
    base_output_dir = os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, "default")
    checkpoint_dir = save_checkpoint_dir or os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    bridge = AutoBridge.from_hf_pretrained(hf_model_path, trust_remote_code=True)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_parallelism
    model_cfg.pipeline_model_parallel_size = 1
    model_cfg.context_parallel_size = 1
    model_cfg.sequence_parallel = False
    model_cfg.seq_length = seq_length

    opt_config, sched = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters, lr_decay_iters=None, max_lr=lr, min_lr=min_lr,
    )

    dataset_cfg = HFDatasetConversationProvider(
        seq_length=seq_length, hf_processor_path=hf_model_path,
        maker_name=dataset_maker_name, num_workers=2, dataloader_type="single",
        data_sharding=True, pin_memory=True, persistent_workers=False,
    )

    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters, eval_interval=500, eval_iters=32,
            global_batch_size=global_batch_size, micro_batch_size=micro_batch_size,
            manual_gc=True, manual_gc_interval=100, manual_gc_eval=100,
        ),
        optimizer=opt_config, scheduler=sched,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True, grad_reduce_in_fp32=True,
            overlap_grad_reduce=False, overlap_param_gather=False,
            average_in_collective=False, data_parallel_sharding_strategy="optim_grads_params",
            use_distributed_optimizer=True,
        ),
        dataset=dataset_cfg,
        logger=LoggerConfig(log_interval=10, tensorboard_dir=tensorboard_dir, log_timers_to_tensorboard=True),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE),
        checkpoint=CheckpointConfig(
            pretrained_checkpoint=pretrained_checkpoint, save_interval=save_interval,
            save=checkpoint_dir, load=checkpoint_dir, ckpt_format="torch_dist", fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=1234), mixed_precision="bf16_mixed",
    )

    if lora_on_language_model:
        if lora_on_vision_model:
            cfg.peft = VLMLoRA(
                target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"], dim=16, alpha=32,
            )
        else:
            cfg.peft = VLMLoRA(
                target_modules=["*language_model*.linear_qkv", "*language_model*.linear_proj",
                                "*language_model*.linear_fc1", "*language_model*.linear_fc2"],
                dim=16, alpha=32, freeze_vision_model=False, freeze_vision_projection=False,
            )
        cfg.optimizer.lr = 5e-5
        cfg.optimizer.min_lr = 5e-6
        cfg.model.tensor_model_parallel_size = 2

    return cfg
