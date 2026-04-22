#!/usr/bin/env python3
"""
HF transformers generate()-based evaluation for SafeWatch guardrail models.

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/hf_generate_eval.py \
    --model-path ./exports/nemotron_nano_v2_vl_sft_510_hf \
    --eval-jsonl /gpfs/public/artifacts/SafeWatch-Bench-200K/sft_jsonl/sft_eval_v2.jsonl \
    --output-dir ./results/nemotron_nano_v2_vl_safewatch_datav2/eval_hf_generate_510/iter_0000510 \
    --max-new-tokens 4096
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append("/opt/venv/lib/python3.12/site-packages")

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3rdparty" / "Megatron-LM"))

HF_MODEL_PATH = ""


HF_BASE_MODEL = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"


def load_model(model_path: str, device: str = "cuda"):
    print(f"[load] loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(HF_BASE_MODEL, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(HF_BASE_MODEL, trust_remote_code=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load] params: {n_params:,}")
    return model, tokenizer, processor


def load_video_frames(video_path: str):
    """Load video frames and metadata using the Megatron-Bridge nemotron_vl_utils."""
    from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import (
        maybe_path_or_url_to_data_urls,
        pil_image_from_base64,
    )

    image_urls, metadata = maybe_path_or_url_to_data_urls(
        video_path, fps=0, nframe=10, nframe_max=-1,
    )
    frames = [pil_image_from_base64(u) for u in image_urls]
    return frames, metadata


def generate_one(
    model, tokenizer, processor, gt_obj: dict, device: str, max_new_tokens: int,
) -> tuple[str | None, str]:
    """Generate prediction for one SafeWatch eval record. Returns (pred_text, gt_text)."""
    messages = gt_obj.get("messages", [])
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return None, ""

    user_content = user_msgs[0].get("content", [])
    if isinstance(user_content, str):
        user_content = [{"type": "text", "text": user_content}]

    gt_text = ""
    for m in messages:
        if m.get("role") == "assistant":
            c = m.get("content", "")
            gt_text = c if isinstance(c, str) else ""
            break

    is_video = any(item.get("type") == "video" for item in user_content)

    conversation = [{"role": "user", "content": user_content}]

    if is_video:
        video_item = next(i for i in user_content if i.get("type") == "video")
        video_path = video_item.get("video") or video_item.get("path") or video_item.get("url", "")

        frames, metadata = load_video_frames(video_path)

        prompt = processor.apply_chat_template(
            [conversation], tokenize=False, add_generation_prompt=True,
        )

        if metadata:
            inputs = processor(
                text=prompt,
                videos=frames,
                videos_kwargs={"video_metadata": metadata},
                return_tensors="pt",
            )
        else:
            inputs = processor(
                text=prompt,
                videos=frames,
                return_tensors="pt",
            )
        inputs = inputs.to(device)

        generated_ids = model.generate(
            pixel_values_videos=inputs.pixel_values_videos,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    else:
        prompt = processor.apply_chat_template(
            [conversation], tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(text=prompt, return_tensors="pt")
        inputs = inputs.to(device)

        generated_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_len = inputs.input_ids.shape[1]
    new_ids = generated_ids[0, prompt_len:]
    pred_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return pred_text, gt_text


def run_eval(
    model, tokenizer, processor,
    eval_jsonl: str,
    output_dir: str,
    max_new_tokens: int = 4096,
    max_records: int | None = None,
    iteration: int = 510,
    device: str = "cuda",
):
    from megatron.bridge.training.eval_dump_interval_parse import intervals_presentation_for_span

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    records_file = out_path / "records.jsonl"

    n_written = 0
    t0 = time.time()

    with records_file.open("w", encoding="utf-8") as out_fp, open(eval_jsonl, "r") as f:
        for idx, line in enumerate(f):
            if max_records is not None and idx >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                gt_obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            try:
                pred_text, gt_text = generate_one(
                    model, tokenizer, processor, gt_obj, device, max_new_tokens,
                )
            except Exception as e:
                print(f"[eval] skip record {idx}: {e}", flush=True)
                continue

            if pred_text is None:
                continue

            payload = {
                "iteration": iteration,
                "consumed_train_samples": 0,
                "jsonl_record_index": idx,
                "gt_jsonl": eval_jsonl,
                "gt_record": gt_obj,
                "prediction": {
                    "supervised_span_text": pred_text,
                    "note": "HF transformers greedy generation from user prompt only.",
                    "intervals": intervals_presentation_for_span(pred_text),
                },
                "ground_truth": {
                    "supervised_span_text": gt_text,
                    "intervals": intervals_presentation_for_span(gt_text),
                },
            }
            out_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
            out_fp.flush()
            n_written += 1

            if n_written % 10 == 0:
                elapsed = time.time() - t0
                speed = elapsed / n_written
                total = max_records or 916
                eta = (total - n_written) * speed
                print(
                    f"[eval] {n_written}/{total} | {elapsed:.0f}s | "
                    f"{speed:.1f}s/rec | ETA {eta/60:.0f}min",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"[eval] done: {n_written} records in {elapsed:.0f}s -> {records_file}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--iteration", type=int, default=510)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model, tokenizer, processor = load_model(args.model_path, args.device)
    run_eval(
        model, tokenizer, processor,
        eval_jsonl=args.eval_jsonl,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        max_records=args.max_records,
        iteration=args.iteration,
        device=args.device,
    )


if __name__ == "__main__":
    main()
