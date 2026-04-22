#!/usr/bin/env python3
"""
vLLM-based evaluation for SafeWatch guardrail models.

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/vllm_generate_eval.py \
    --model-path ./exports/nemotron_nano_v2_vl_sft_510_hf \
    --eval-jsonl /gpfs/public/artifacts/SafeWatch-Bench-200K/sft_jsonl/sft_eval_v2.jsonl \
    --output-dir ./results/nemotron_nano_v2_vl_safewatch_datav2/eval_vllm_generate_510/iter_0000510 \
    --max-new-tokens 4096
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3rdparty" / "Megatron-LM"))

from vllm import LLM, SamplingParams
from vllm.multimodal.utils import encode_video_base64
from transformers import AutoProcessor


def load_video_frames(video_path: str):
    """Load video frames using nemotron_vl_utils."""
    from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import (
        maybe_path_or_url_to_data_urls,
        pil_image_from_base64,
    )
    image_urls, metadata = maybe_path_or_url_to_data_urls(
        video_path, fps=0, nframe=10, nframe_max=-1,
    )
    frames = [pil_image_from_base64(u) for u in image_urls]
    return frames, metadata


def build_prompt(gt_obj: dict, processor) -> tuple[dict | None, str]:
    """Build vLLM prompt from a SafeWatch JSONL record. Returns (vllm_input, gt_text)."""
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

        prompt_text = processor.apply_chat_template(
            [conversation], tokenize=False, add_generation_prompt=True,
        )

        if metadata:
            inputs = processor(
                text=prompt_text,
                videos=frames,
                videos_kwargs={"video_metadata": metadata},
                return_tensors="pt",
            )
        else:
            inputs = processor(
                text=prompt_text,
                videos=frames,
                return_tensors="pt",
            )

        pixel_values_videos = inputs.get("pixel_values_videos")
        mm_data = {}
        if pixel_values_videos is not None:
            mm_data["pixel_values_videos"] = pixel_values_videos

        return {
            "prompt": prompt_text,
            "multi_modal_data": mm_data,
        }, gt_text
    else:
        prompt_text = processor.apply_chat_template(
            [conversation], tokenize=False, add_generation_prompt=True,
        )
        return {"prompt": prompt_text}, gt_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--iteration", type=int, default=510)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of prompts to batch before calling llm.generate()")
    args = parser.parse_args()

    hf_base = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"

    print(f"[init] loading vLLM model from {args.model_path} ...", flush=True)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    )
    print("[init] model loaded", flush=True)

    processor = AutoProcessor.from_pretrained(hf_base, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
    )

    from megatron.bridge.training.eval_dump_interval_parse import intervals_presentation_for_span

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    records_file = out_path / "records.jsonl"

    gt_records: list[dict] = []
    with open(args.eval_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                gt_records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if args.max_records is not None:
        gt_records = gt_records[: args.max_records]

    print(f"[eval] {len(gt_records)} records to process", flush=True)

    n_written = 0
    n_skip = 0
    t0 = time.time()

    with records_file.open("w", encoding="utf-8") as out_fp:
        batch_inputs = []
        batch_meta = []

        for idx, gt_obj in enumerate(gt_records):
            try:
                vllm_input, gt_text = build_prompt(gt_obj, processor)
            except Exception as e:
                print(f"[eval] skip {idx}: prompt build error: {e}", flush=True)
                n_skip += 1
                continue

            if vllm_input is None:
                n_skip += 1
                continue

            batch_inputs.append(vllm_input)
            batch_meta.append((idx, gt_obj, gt_text))

            if len(batch_inputs) >= args.batch_size or idx == len(gt_records) - 1:
                try:
                    outputs = llm.generate(
                        batch_inputs,
                        sampling_params=sampling_params,
                    )
                except Exception as e:
                    print(f"[eval] batch generate error: {e}", flush=True)
                    for inp in batch_inputs:
                        n_skip += 1
                    batch_inputs.clear()
                    batch_meta.clear()
                    continue

                for out, (rec_idx, gt_obj_i, gt_text_i) in zip(outputs, batch_meta):
                    pred_text = out.outputs[0].text

                    payload = {
                        "iteration": args.iteration,
                        "consumed_train_samples": 0,
                        "jsonl_record_index": rec_idx,
                        "gt_jsonl": args.eval_jsonl,
                        "gt_record": gt_obj_i,
                        "prediction": {
                            "supervised_span_text": pred_text,
                            "note": "vLLM greedy generation from user prompt only.",
                            "intervals": intervals_presentation_for_span(pred_text),
                        },
                        "ground_truth": {
                            "supervised_span_text": gt_text_i,
                            "intervals": intervals_presentation_for_span(gt_text_i),
                        },
                    }
                    out_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    n_written += 1

                elapsed = time.time() - t0
                total = len(gt_records)
                speed = elapsed / n_written if n_written else 0
                eta = (total - n_written - n_skip) * speed
                print(
                    f"[eval] {n_written}/{total} | {elapsed:.0f}s | "
                    f"{speed:.1f}s/rec | ETA {eta/60:.0f}min | skip={n_skip}",
                    flush=True,
                )

                batch_inputs.clear()
                batch_meta.clear()

    elapsed = time.time() - t0
    print(f"[eval] done: {n_written} records in {elapsed:.0f}s -> {records_file}", flush=True)


if __name__ == "__main__":
    main()
