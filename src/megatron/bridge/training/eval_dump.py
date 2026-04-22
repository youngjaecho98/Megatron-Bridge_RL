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

"""Small utilities for dumping lightweight eval artifacts during training.

Supports two modes controlled by ``cfg.logger.eval_dump_mode``:

* ``"teacher_forced"`` (default, legacy) — feeds GT user+assistant tokens and reads
  argmax at supervised positions.
* ``"generate"`` — feeds only the user prompt and runs autoregressive greedy
  decoding to produce a free-form response.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from megatron.core import parallel_state
from megatron.core.transformer import MegatronModule
from megatron.core.utils import unwrap_model
from transformers import AutoProcessor

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.eval_dump_interval_parse import intervals_presentation_for_span
from megatron.bridge.training.state import GlobalState
from megatron.bridge.utils.common_utils import get_rank_safe

_PROCESSOR_CACHE: dict[str, Any] = {}


def _get_cached_processor(hf_path: str) -> Any:
    cached = _PROCESSOR_CACHE.get(hf_path)
    if cached is not None:
        return cached
    proc = AutoProcessor.from_pretrained(hf_path, trust_remote_code=True)
    _PROCESSOR_CACHE[hf_path] = proc
    return proc


def _conversation_from_gt_record(
    gt_obj: Dict[str, Any],
    *,
    include_assistant: bool = True,
) -> Optional[dict]:
    """Build a Nemotron Nano V2 VL collate example from a SafeWatch-style JSONL record.

    Args:
        gt_obj: A single JSONL record with ``messages`` list.
        include_assistant: If False, strip assistant turns so the conversation
            ends after the last user turn (used for autoregressive generation).
    """
    if not isinstance(gt_obj.get("messages"), list) or not gt_obj["messages"]:
        return None

    example: dict = {"conversation": []}
    for msg in gt_obj["messages"]:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("user", "assistant"):
            continue
        if not include_assistant and role == "assistant":
            continue
        if isinstance(content, list):
            parts: List[Dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                t = item.get("type")
                if t == "text" and isinstance(item.get("text"), str):
                    parts.append({"type": "text", "text": item["text"]})
                elif t == "video":
                    p = item.get("path") or item.get("video") or item.get("url")
                    if isinstance(p, str) and p:
                        parts.append({"type": "video", "path": p})
                elif t == "image":
                    p = item.get("image") or item.get("path") or item.get("url")
                    if isinstance(p, str) and p:
                        parts.append({"type": "image", "image": p})
            if parts:
                example["conversation"].append({"role": role, "content": parts})
        elif isinstance(content, str) and role == "assistant":
            example["conversation"].append({"role": role, "content": [{"type": "text", "text": content}]})

    if not example["conversation"]:
        return None
    return example


def _teacher_forced_supervised_span(
    mod: Any,
    batch: dict[str, Any],
    tok: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """Greedy decode over supervised positions; returns (pred_text, gt_text) or ("","") if none."""
    images = batch.get("pixel_values")
    num_image_tiles = None

    visual_inputs = batch.get("visual_inputs")
    if visual_inputs is not None:
        vi_dict = visual_inputs.normalized_for_model()
        images = vi_dict.get("images", images)
        num_image_tiles = vi_dict.get("num_image_tiles")

    if images is not None:
        images = images.cuda() if not images.is_cuda else images
    if num_image_tiles is not None:
        num_image_tiles = num_image_tiles.cuda() if not num_image_tiles.is_cuda else num_image_tiles

    input_ids = batch["input_ids"].cuda()
    position_ids = batch["position_ids"].cuda()
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.cuda()
    labels = batch["labels"].cuda()
    loss_mask = batch["loss_mask"].cuda()

    forward_args = {
        "images": images,
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": None,
        "loss_mask": None,
        "runtime_gather_output": True,
    }
    if num_image_tiles is not None:
        forward_args["num_image_tiles"] = num_image_tiles

    was_training = mod.training
    mod.eval()
    try:
        with torch.no_grad():
            raw_out = mod(**forward_args)
    finally:
        if was_training:
            mod.train()

    logits = raw_out[0] if isinstance(raw_out, tuple) else raw_out
    if not torch.is_tensor(logits):
        return None, None

    if logits.dim() != 3:
        return None, None
    batch_size = int(logits.size(0))
    if input_ids.dim() != 2 or input_ids.size(0) != batch_size:
        return None, None

    llava = getattr(mod, "llava_model", None)
    image_token_index = getattr(llava, "image_token_index", None)
    img_seq_len = getattr(llava, "img_seq_len", None)
    if image_token_index is None or img_seq_len is None:
        return None, None
    img_seq_len = int(img_seq_len)
    if img_seq_len <= 0:
        return None, None

    num_patches = batch.get("num_patches")
    num_image_tiles = None
    if torch.is_tensor(num_patches):
        num_image_tiles = num_patches.detach().to(torch.int).view(-1).tolist()

    image_token_mask = input_ids == int(image_token_index)
    num_images_per_sample = torch.sum(image_token_mask, dim=-1)

    if num_image_tiles is None:
        total_images = int(num_images_per_sample.sum().item())
        num_image_tiles_t = torch.ones(total_images, dtype=torch.int, device=input_ids.device)
    else:
        num_image_tiles_t = torch.tensor(num_image_tiles, dtype=torch.int, device=input_ids.device)

    image_token_mask_lens = image_token_mask.int().clone()
    if int(num_images_per_sample.sum().item()) != int(num_image_tiles_t.numel()):
        num_image_tiles_t = torch.ones(int(num_images_per_sample.sum().item()), dtype=torch.int, device=input_ids.device)

    if num_image_tiles_t.numel() > 0:
        image_token_mask_lens[image_token_mask] = num_image_tiles_t * img_seq_len - 1
    new_position_ids = torch.cumsum((image_token_mask_lens + 1), dim=-1) - 1
    batch_indices, non_image_indices = torch.where(~image_token_mask)
    text_position_ids = new_position_ids[batch_indices, non_image_indices]

    if logits.size(1) <= int(text_position_ids.max().item()):
        print(
            f"[eval_dump] rank={get_rank_safe()} skip: combined logits seq too short for mapped positions; "
            f"logits.shape={tuple(logits.shape)} max_text_pos={int(text_position_ids.max().item())}",
            flush=True,
        )
        return None, None

    text_logits = logits[batch_indices, text_position_ids, :]
    text_pred_ids = torch.argmax(text_logits, dim=-1)

    sup_text = loss_mask.bool() & (labels != -100)
    sup_b, sup_t = torch.where(sup_text & (~image_token_mask))
    if sup_b.numel() == 0:
        return "", ""

    flat_index = torch.full_like(input_ids, -1, dtype=torch.long)
    flat_index[batch_indices, non_image_indices] = torch.arange(text_pred_ids.numel(), device=input_ids.device)
    sup_flat = flat_index[sup_b, sup_t]
    valid = sup_flat >= 0
    sup_flat = sup_flat[valid]
    gt_ids = labels[sup_b[valid], sup_t[valid]]
    pr_ids = text_pred_ids[sup_flat]

    gt_text = tok.decode(gt_ids.detach().cpu().tolist(), skip_special_tokens=True)
    pred_text = tok.decode(pr_ids.detach().cpu().tolist(), skip_special_tokens=True)
    return pred_text, gt_text


def _autoregressive_generate(
    mod: Any,
    batch: dict[str, Any],
    tok: Any,
    max_new_tokens: int = 4096,
) -> Optional[str]:
    """Greedy autoregressive generation from a prompt-only batch (no GT assistant tokens)."""
    images = batch.get("pixel_values")
    num_image_tiles = None

    visual_inputs = batch.get("visual_inputs")
    if visual_inputs is not None:
        vi_dict = visual_inputs.normalized_for_model()
        images = vi_dict.get("images", images)
        num_image_tiles = vi_dict.get("num_image_tiles")

    if images is not None:
        images = images.cuda() if not images.is_cuda else images
    if num_image_tiles is not None:
        num_image_tiles = num_image_tiles.cuda() if not num_image_tiles.is_cuda else num_image_tiles

    input_ids = batch["input_ids"].cuda()  # (1, prompt_len)
    position_ids = batch["position_ids"].cuda()

    eos_token_id = getattr(tok, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tok.convert_tokens_to_ids("</s>")

    was_training = mod.training
    mod.eval()
    generated_ids: list[int] = []
    try:
        with torch.no_grad():
            cur_input_ids = input_ids
            cur_position_ids = position_ids
            cur_images = images
            cur_num_image_tiles = num_image_tiles

            for _ in range(max_new_tokens):
                forward_args: dict[str, Any] = {
                    "images": cur_images,
                    "input_ids": cur_input_ids,
                    "position_ids": cur_position_ids,
                    "attention_mask": None,
                    "labels": None,
                    "loss_mask": None,
                    "runtime_gather_output": True,
                }
                if cur_num_image_tiles is not None:
                    forward_args["num_image_tiles"] = cur_num_image_tiles

                raw_out = mod(**forward_args)
                logits = raw_out[0] if isinstance(raw_out, tuple) else raw_out
                if not torch.is_tensor(logits):
                    break

                next_token_logits = logits[:, -1, :]
                next_token_id = int(torch.argmax(next_token_logits, dim=-1).item())

                if next_token_id == eos_token_id:
                    break
                generated_ids.append(next_token_id)

                next_ids = torch.tensor([[next_token_id]], device=input_ids.device, dtype=input_ids.dtype)
                cur_input_ids = torch.cat([cur_input_ids, next_ids], dim=1)
                seq_len = cur_input_ids.size(1)
                cur_position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    finally:
        if was_training:
            mod.train()

    if not generated_ids:
        return ""
    return tok.decode(generated_ids, skip_special_tokens=True)


def _collate_prompt_only(
    example: dict,
    processor: Any,
) -> dict[str, Any]:
    """Collate a user-only conversation into a prompt batch (no labels/loss_mask needed)."""
    from megatron.bridge.data.vlm_datasets.collate import NemotronVLVisualInputs
    from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import adjust_image_tokens

    is_video = example["conversation"][0]["content"][0]["type"] == "video"
    if is_video:
        from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import (
            maybe_path_or_url_to_data_urls,
            pil_image_from_base64,
        )

        video_path = example["conversation"][0]["content"][0]["path"]
        image_urls, metadata = maybe_path_or_url_to_data_urls(
            video_path, fps=0, nframe=10, nframe_max=-1,
        )
        frames = [[pil_image_from_base64(u) for u in image_urls]]

        prompt = processor.apply_chat_template(
            [example["conversation"]],
            tokenize=False,
            add_generation_prompt=True,
        )
        batch = processor(
            text=prompt,
            videos=frames,
            videos_kwargs={"video_metadata": metadata},
            return_tensors="pt",
        )
    else:
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        batch = processor.apply_chat_template(
            [example["conversation"]],
            tokenize=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )

    img_start_token_id = 131073
    img_end_token_id = 131074
    dummy_loss_mask = torch.zeros_like(batch["input_ids"], dtype=torch.float)
    adjusted = adjust_image_tokens(
        {"input_ids": batch["input_ids"], "loss_mask": dummy_loss_mask},
        batch["num_patches"],
        img_start_token_id,
        img_end_token_id,
    )

    if is_video:
        video_token_id = processor.tokenizer.convert_tokens_to_ids("<video>")
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        adjusted["input_ids"] = torch.where(
            adjusted["input_ids"] == video_token_id, image_token_id, adjusted["input_ids"],
        )

    batch["input_ids"] = adjusted["input_ids"]

    if "position_ids" not in batch:
        bs, sl = batch["input_ids"].shape
        batch["position_ids"] = torch.arange(sl, device=batch["input_ids"].device).unsqueeze(0).expand(bs, -1)

    key = "pixel_values_videos" if is_video else "pixel_values"
    pv = batch[key].to(torch.bfloat16)
    batch[key] = pv
    total_tiles = pv.shape[0]
    num_image_tiles = torch.ones(total_tiles, dtype=torch.int)
    batch["visual_inputs"] = NemotronVLVisualInputs(pixel_values=pv, num_image_tiles=num_image_tiles)

    return batch


def maybe_dump_eval_gt_pred_json(
    *,
    state: GlobalState,
    model: list[MegatronModule],
    cfg: ConfigContainer,
) -> None:
    """Optionally dump pred JSON (includes ``gt_record`` and ``ground_truth``) during validation.

    This is intentionally best-effort and rank-local:
    - Only runs when `cfg.logger.eval_dump_dir` and `cfg.logger.eval_gt_jsonl_path` are set.
    - Only one rank writes: pipeline last stage and data-parallel rank 0 within that partition.
    - Uses a teacher-forced greedy decode over supervised positions (loss_mask==1) by running
      `LLaVAModel.forward(..., labels=None)` to obtain logits.

    Notes:
    - This is not a full beam-search / autoregressive generation implementation.
    - For SafeWatch-style JSON targets, prefer comparing the decoded supervised span as text.
    - Uses ``runtime_gather_output=True`` so tensor-parallel logits are gathered before argmax.
    - Reads up to ``cfg.logger.eval_dump_max_records`` non-empty JSON lines (``None`` = entire file).
    - Writes one JSONL per eval step: ``{eval_dump_dir}/iter_{step:07d}/records.jsonl`` (one JSON object per line).
    """

    dump_dir = getattr(cfg.logger, "eval_dump_dir", None)
    gt_path = getattr(cfg.logger, "eval_gt_jsonl_path", None)
    if not dump_dir or not gt_path:
        return

    if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
        return

    if parallel_state.get_data_parallel_rank() != 0:
        return

    gt_file = Path(gt_path)
    if not gt_file.is_file():
        return

    max_records = getattr(cfg.logger, "eval_dump_max_records", 1)
    if max_records is not None and max_records < 1:
        print(f"[eval_dump] rank={get_rank_safe()} skip: eval_dump_max_records must be >= 1 or null", flush=True)
        return

    out_root = Path(dump_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    hf_path = getattr(cfg.dataset, "hf_processor_path", None)
    if not hf_path:
        print(f"[eval_dump] rank={get_rank_safe()} skip: no hf_processor_path", flush=True)
        return

    eval_mode = getattr(cfg.logger, "eval_dump_mode", "teacher_forced")
    use_generate = eval_mode == "generate"

    processor = _get_cached_processor(str(hf_path))
    from megatron.bridge.data.vlm_datasets.collate import nemotron_nano_v2_vl_collate_fn

    tok = getattr(processor, "tokenizer", processor)
    mod = unwrap_model(model)[0]

    step_dir_name = f"iter_{int(state.train_state.step):07d}"
    iter_out_dir = out_root / step_dir_name
    iter_out_dir.mkdir(parents=True, exist_ok=True)
    records_path = iter_out_dir / "records.jsonl"
    n_written = 0

    max_new_tokens = int(getattr(cfg.logger, "eval_generate_max_tokens", 4096))

    print(
        f"[eval_dump] rank={get_rank_safe()} mode={eval_mode} max_records={max_records}"
        + (f" max_new_tokens={max_new_tokens}" if use_generate else ""),
        flush=True,
    )

    with records_path.open("w", encoding="utf-8") as out_fp, gt_file.open("r", encoding="utf-8") as f:
        rec_idx = 0
        for line in f:
            if max_records is not None and rec_idx >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                gt_obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[eval_dump] rank={get_rank_safe()} skip line {rec_idx}: invalid JSON", flush=True)
                rec_idx += 1
                continue
            if not isinstance(gt_obj, dict):
                rec_idx += 1
                continue

            if use_generate:
                example = _conversation_from_gt_record(gt_obj, include_assistant=False)
                if example is None:
                    rec_idx += 1
                    continue
                try:
                    batch = _collate_prompt_only(example, processor)
                except Exception as e:
                    print(
                        f"[eval_dump] rank={get_rank_safe()} skip record {rec_idx}: collate error: {e}",
                        flush=True,
                    )
                    rec_idx += 1
                    continue
                pred_text = _autoregressive_generate(mod, batch, tok, max_new_tokens=max_new_tokens)
                if pred_text is None:
                    rec_idx += 1
                    continue

                # Also get GT text from the assistant turn for comparison
                gt_example = _conversation_from_gt_record(gt_obj, include_assistant=True)
                gt_text = ""
                if gt_example:
                    for turn in gt_example["conversation"]:
                        if turn["role"] == "assistant":
                            for part in turn["content"]:
                                if part.get("type") == "text":
                                    gt_text = part["text"]
                                    break
                            break

                payload = {
                    "iteration": int(state.train_state.step),
                    "consumed_train_samples": int(state.train_state.consumed_train_samples),
                    "jsonl_record_index": rec_idx,
                    "gt_jsonl": str(gt_file),
                    "gt_record": gt_obj,
                    "prediction": {
                        "supervised_span_text": pred_text,
                        "note": "Autoregressive greedy generation from user prompt only.",
                        "intervals": intervals_presentation_for_span(pred_text),
                    },
                    "ground_truth": {
                        "supervised_span_text": gt_text,
                        "intervals": intervals_presentation_for_span(gt_text),
                    },
                }
            else:
                example = _conversation_from_gt_record(gt_obj)
                if example is None:
                    print(
                        f"[eval_dump] rank={get_rank_safe()} skip record {rec_idx}: "
                        "unsupported schema (expected messages[])",
                        flush=True,
                    )
                    rec_idx += 1
                    continue

                batch = nemotron_nano_v2_vl_collate_fn([example], processor)
                pred_text, gt_text = _teacher_forced_supervised_span(mod, batch, tok)
                if pred_text is None:
                    rec_idx += 1
                    continue

                payload = {
                    "iteration": int(state.train_state.step),
                    "consumed_train_samples": int(state.train_state.consumed_train_samples),
                    "jsonl_record_index": rec_idx,
                    "gt_jsonl": str(gt_file),
                    "gt_record": gt_obj,
                    "prediction": {
                        "supervised_span_text": pred_text,
                        "note": "Teacher-forced greedy decode over supervised positions (loss_mask==1); "
                        "not full generation.",
                        "intervals": intervals_presentation_for_span(pred_text),
                    },
                    "ground_truth": {
                        "supervised_span_text": gt_text,
                        "intervals": intervals_presentation_for_span(gt_text),
                    },
                }

            out_fp.write(json.dumps(payload, ensure_ascii=False))
            out_fp.write("\n")
            out_fp.flush()
            n_written += 1
            rec_idx += 1

            if use_generate and n_written % 10 == 0:
                print(f"[eval_dump] rank={get_rank_safe()} generated {n_written}/{max_records or 'all'}...", flush=True)

    if n_written:
        print(f"[eval_dump] rank={get_rank_safe()} wrote {n_written} record(s) to {records_path}", flush=True)
    else:
        try:
            records_path.unlink(missing_ok=True)
        except OSError:
            pass
