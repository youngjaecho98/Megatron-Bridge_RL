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

"""SimPO (Simple Preference Optimization) forward step for VLM training.

SimPO eliminates the reference model by using length-normalised average
log-probability as the implicit reward, halving the memory and compute
cost compared to DPO.

Reference: https://arxiv.org/abs/2405.14734
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Iterable

import torch
import torch.nn.functional as F
from megatron.core.models.gpt import GPTModel
from megatron.core.utils import get_model_config

from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.training.vlm_step import get_batch


logger: logging.Logger = logging.getLogger(__name__)


def _compute_simpo_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    beta: float,
    gamma: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute SimPO loss from a batch of ``[chosen | rejected]`` pairs.

    The batch dimension is assumed to contain ``2N`` entries where the first
    ``N`` are *chosen* and the last ``N`` are *rejected* responses.

    SimPO reward:  r(y) = (beta / |y|) * sum_t log p(y_t | y_{<t}, x)
    SimPO loss:    -log sigmoid( r(chosen) - r(rejected) - gamma )

    Args:
        logits: Raw model logits ``[2N, seq_len, vocab]``.
        labels: Token ids for next-token targets ``[2N, seq_len]``.
        loss_mask: Binary mask ``[2N, seq_len]`` (1 = valid token).
        beta: Temperature / scaling factor.
        gamma: Target reward margin.

    Returns:
        Tuple of (scalar loss, metrics dict for logging).
    """
    vocab_size = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    gather_labels = labels.clamp(min=0, max=vocab_size - 1)
    per_token_log_probs = torch.gather(log_probs, dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)

    # Mask out padding / prompt tokens and compute per-sequence average log-prob
    masked_log_probs = per_token_log_probs * loss_mask
    seq_log_probs = masked_log_probs.sum(dim=-1)  # [2N]
    seq_lengths = loss_mask.sum(dim=-1).clamp(min=1)  # [2N]
    avg_log_probs = seq_log_probs / seq_lengths  # length-normalised

    half = logits.size(0) // 2
    chosen_rewards = beta * avg_log_probs[:half]
    rejected_rewards = beta * avg_log_probs[half:]

    reward_margin = chosen_rewards - rejected_rewards - gamma
    loss = -F.logsigmoid(reward_margin).mean()

    with torch.no_grad():
        reward_acc = (chosen_rewards > rejected_rewards).float().mean()
        chosen_reward_mean = chosen_rewards.mean()
        rejected_reward_mean = rejected_rewards.mean()

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    metrics: dict[str, torch.Tensor] = {
        "simpo loss": reporting_loss,
        "reward accuracy": torch.cat([reward_acc.view(1), torch.ones(1, device=loss.device)]),
        "chosen reward": torch.cat([chosen_reward_mean.view(1), torch.ones(1, device=loss.device)]),
        "rejected reward": torch.cat([rejected_reward_mean.view(1), torch.ones(1, device=loss.device)]),
    }

    return loss, num_tokens, metrics


def _simpo_loss_func(
    loss_mask: torch.Tensor,
    output_tensor: torch.Tensor,
    *,
    beta: float,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Loss function called by the Megatron training loop.

    When ``cross_entropy_loss_fusion`` is disabled the model returns raw
    logits of shape ``[batch, seq_len, vocab]``.  We shift them by one
    position to align predictions with labels, then compute SimPO loss.
    """
    logits = output_tensor.float()

    batch_size, seq_len, vocab_size = logits.shape
    shift_logits = logits[:, :-1, :].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous()

    # Build shifted labels from the original token ids that were fed via loss_mask's companion
    # The labels tensor is stored in _simpo_labels by the forward step
    labels = _simpo_labels_storage.get("labels")
    if labels is None:
        raise RuntimeError("SimPO labels not found — _simpo_labels_storage was not populated by forward step")
    shift_labels = labels[:, 1:].contiguous()

    return _compute_simpo_loss(
        shift_logits,
        shift_labels,
        shift_mask,
        beta=beta,
        gamma=gamma,
    )


# Thread-local-ish storage to pass labels from forward step to loss function.
# Megatron's pipeline schedule calls the loss function separately from the
# forward function, so we need a side channel for the labels tensor.
_simpo_labels_storage: dict[str, torch.Tensor] = {}


def _simpo_forward_step(
    state: GlobalState,
    data_iterator: Iterable,
    model: GPTModel,
    *,
    beta: float,
    gamma: float,
) -> tuple[torch.Tensor, partial]:
    """Forward step that returns raw logits and a SimPO loss function.

    Requires ``cross_entropy_loss_fusion = False`` in the model config so
    that logits are returned instead of fused per-token losses.
    """
    timers = state.timers
    straggler_timer = state.straggler_timer
    config = get_model_config(model)
    pg_collection = get_pg_collection(model)
    use_mtp = (getattr(config, "mtp_num_layers", None) or 0) > 0

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            cu_seqlens,
            max_seqlen,
            visual_inputs,
        ) = get_batch(data_iterator, state.cfg, use_mtp, pg_collection=pg_collection)
    timers("batch-generator").stop()

    # Stash labels for the loss function (Megatron calls it separately)
    _simpo_labels_storage["labels"] = labels

    forward_args = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": None,  # SimPO needs raw logits, not fused loss
    }

    if visual_inputs is not None:
        forward_args.update(visual_inputs.normalized_for_model())

    with straggler_timer:
        output_tensor = model(**forward_args)
        if isinstance(output_tensor, tuple):
            output_tensor = output_tensor[0]

    loss_function = partial(
        _simpo_loss_func,
        loss_mask,
        beta=beta,
        gamma=gamma,
    )

    return output_tensor, loss_function


def make_simpo_forward_step(
    *,
    beta: float = 2.0,
    gamma: float = 0.5,
) -> partial:
    """Create a SimPO forward step function with bound hyper-parameters.

    Args:
        beta: SimPO temperature / scaling factor. Higher values make the
            optimisation more sensitive to reward differences.
        gamma: Target reward margin between chosen and rejected responses.

    Returns:
        A callable matching the ``FourArgForwardStep`` protocol, suitable
        for passing to :func:`megatron.bridge.training.finetune.finetune`.
    """
    return partial(_simpo_forward_step, beta=beta, gamma=gamma)
