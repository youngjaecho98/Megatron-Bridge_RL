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
Built-in maker functions that transform HuggingFace datasets into
conversation-style examples consumable by VLM processors.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import concatenate_datasets, load_dataset

from megatron.bridge.data.vlm_datasets.token_utils import json2token
from megatron.bridge.utils.common_utils import resolve_path


def make_rdr_dataset(
    path_or_dataset: str = "quintend/rdr-items", split: str = "train", **kwargs
) -> List[Dict[str, Any]]:
    """Load and preprocess the RDR dataset for image-to-text fine-tuning.

    Returns a list of examples with a "conversation" field that includes an image and text.
    """
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example):
        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": "Describe this image."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example["text"]}],
                },
            ],
        }

    return [format(example) for example in dataset]


def make_cord_v2_dataset(
    path_or_dataset: str = "naver-clova-ix/cord-v2", split: str = "train", **kwargs
) -> List[Dict[str, Any]]:
    """Load and preprocess the CORD-V2 dataset for image-to-text fine-tuning."""
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example):
        ground_truth = json.loads(example["ground_truth"])
        if "gt_parses" in ground_truth:
            assert isinstance(ground_truth["gt_parses"], list)
            gt_jsons = ground_truth["gt_parses"]
        else:
            assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
            gt_jsons = [ground_truth["gt_parse"]]

        text = random.choice([json2token(gt_json, sort_json_key=True) for gt_json in gt_jsons])

        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": "Describe this image."},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": text}]},
            ],
        }

    return [format(example) for example in dataset]


def make_medpix_dataset(
    path_or_dataset: str = "mmoukouba/MedPix-VQA", split: str = "train", **kwargs
) -> List[Dict[str, Any]]:
    """Load and preprocess the MedPix dataset for image-to-text fine-tuning."""
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example):
        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image_id"]},
                        {"type": "text", "text": example["question"]},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]},
            ],
        }

    return [format(example) for example in dataset]


def make_raven_dataset(
    path_or_dataset: str = "HuggingFaceM4/the_cauldron",
    subset: str = "raven",
    split: str = "train",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load and preprocess the Raven subset from the Cauldron dataset.

    This subset follows the IDEFICS-style layout where each sample contains:
    - ``images``: a (possibly empty) list of PIL images
    - ``texts``: a list of conversation dictionaries. For Raven, ``texts[0]``
      is a *single* turn stored as a dictionary with two keys::

          {"user": "<question>", "assistant": "<answer>"}

      Only the first element is used.  The ``user`` string is taken as the
      user prompt, and ``assistant`` is the ground-truth answer.

    Conversation building policy:
    1. All images are placed at the beginning of the user turn followed by the
       textual prompt.
    2. The assistant turn contains the answer text.

    Examples missing either images or the required fields are filtered out.
    """
    if split != "train":
        raise ValueError("Raven dataset only supports train split. Please set `train.eval_iters=0`.")
    dataset = load_dataset(path_or_dataset, subset, split=split)

    def format(example):
        images = example.get("images", [])
        texts = example.get("texts", [])
        if not images or not texts or not isinstance(texts[0], dict):
            return None

        user_prompt = texts[0].get("user")
        assistant_answer = texts[0].get("assistant")
        if user_prompt is None or assistant_answer is None:
            return None

        user_content: List[Dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        user_content.append({"type": "text", "text": user_prompt})

        assistant_content = [{"type": "text", "text": assistant_answer}]

        return {
            "conversation": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        }

    formatted = (format(example) for example in dataset)
    # Filter out any None values from malformed rows.
    return [ex for ex in formatted if ex is not None]


def make_llava_video_178k_dataset(
    video_root_path: str,
    path_or_dataset: str = "lmms-lab/LLaVA-Video-178K",
    subsets: str | List[str] = "0_30_s_nextqa",
    split: str = "open_ended",
) -> List[Dict[str, Any]]:
    """Load and preprocess a subset of the *LLaVA-Video-178K* dataset.

    Each row contains:
    - ``video``: path or URL to the MP4 file.
    - ``conversations``: a **two-turn** list::

          [{"from": "human", "value": "<video>\n<question>"},
           {"from": "gpt",   "value": "<answer>"}]

      We map this schema to our internal multimodal conversation format:

      User turn  →  [video, user prompt]
      Assistant  →  answer text

    Note:
        Video files are assumed to be pre-downloaded and stored locally in the
        ``video_root_path`` directory. Rows with missing videos or empty
        conversations are filtered out from the final output.

    Args:
        video_root_path: Root directory where video files are stored locally.
        path_or_dataset: HF dataset path or local cache dir.
        subsets: Single subset name or list of the dataset's directory-style
            subsets to load.
        split: Split to load from the dataset. Note that "train" is automatically
            mapped to "open_ended".

    Returns:
        A list of dicts each containing a ``conversation`` field ready for
        downstream VLM processors.
    """
    if isinstance(subsets, str):
        subsets = [subsets]

    if split == "train":
        split = "open_ended"
    elif split in ("validation", "test"):
        raise ValueError("LLaVA-Video-178K dataset only supports train split. Please set `train.eval_iters=0`.")
    individual_datasets = [load_dataset(path_or_dataset, subset, split=split) for subset in subsets]
    dataset = concatenate_datasets(individual_datasets)

    # FIXME: right now we assume the video files are pre-downloaded and stored in the video_root_path
    # we need to modify this to download the video files from the hub if they are not present in the video_root_path

    def clean_prompt(val: str) -> str:
        # Remove placeholder tokens such as <image> or <video>
        val = val.replace("<image>", "").replace("<video>", "").strip()
        return val.lstrip("\n").rstrip()

    def format(example):
        video = example.get("video")
        convs = example.get("conversations", [])
        if video in (None, "") or not convs:
            return None

        conversation: List[Dict[str, Any]] = []

        first_human_handled = False
        for turn in convs:
            role = turn.get("from")
            value = turn.get("value", "")
            if not value:
                continue
            if role == "human":
                content: List[Dict[str, Any]] = []
                if not first_human_handled:
                    abs_path = resolve_path(Path(video_root_path) / video)
                    content.append({"type": "video", "path": str(abs_path)})
                    first_human_handled = True
                content.append({"type": "text", "text": clean_prompt(value)})
                conversation.append({"role": "user", "content": content})
            elif role == "gpt":
                conversation.append({"role": "assistant", "content": [{"type": "text", "text": value.strip()}]})

        if not conversation:
            return None

        return {"conversation": conversation}

    formatted = (format(ex) for ex in dataset)
    return [ex for ex in formatted if ex is not None]


def make_default_audio_dataset(
    path_or_dataset: str,
    subset: str | None = None,
    split: str = "train",
    audio_column: str = "audio",
    text_column: str = "text",
    prompt: str = "Transcribe the audio clip.",
    remove_text_spaces: bool = True,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load and preprocess a HuggingFace audio dataset for audio-to-text fine-tuning.

    Formats each example into a conversation with an audio user turn and a text assistant turn.
    Works with any HF dataset that has audio and text columns.
    """
    dataset = load_dataset(path_or_dataset, subset, split=split)
    try:
        all_columns = dataset.column_names
    except Exception:
        first_example = dataset[0] if len(dataset) > 0 else {}
        all_columns = list(first_example.keys()) if isinstance(first_example, dict) else []
    if hasattr(dataset, "remove_columns"):
        columns_to_remove = [col for col in all_columns if col not in [audio_column, text_column]]
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)

    def format_example(example):
        text = example[text_column]
        if remove_text_spaces:
            text = text.replace(" ", "")
        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": "placeholder"},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                },
            ],
            "audio": (example[audio_column]["array"], example[audio_column]["sampling_rate"]),
        }

    return [format_example(example) for example in dataset]


def make_cv17_dataset(
    path_or_dataset: str = "ysdede/commonvoice_17_tr_fixed", split: str = "train", **kwargs
) -> List[Dict[str, Any]]:
    """Load and preprocess the CommonVoice 17 dataset for audio-to-text fine-tuning."""
    dataset = load_dataset(path_or_dataset, split=split)
    # Be robust to simple list-like datasets used in tests without `column_names` attr
    try:
        all_columns = dataset.column_names  # type: ignore[attr-defined]
    except Exception:
        first_example = dataset[0] if len(dataset) > 0 else {}
        all_columns = list(first_example.keys()) if isinstance(first_example, dict) else []
    if hasattr(dataset, "remove_columns"):
        columns_to_remove = [col for col in all_columns if col not in ["audio", "transcription"]]
        dataset = dataset.remove_columns(columns_to_remove)

    def format(example):
        return {
            "conversation": [
                {"role": "user", "content": "<|audio_1|>Transcribe the Turkish audio clip."},
                {"role": "assistant", "content": example["transcription"]},
            ],
            "audio": (example["audio"]["array"], example["audio"]["sampling_rate"]),
        }

    return [format(example) for example in dataset]


def make_rlaifv_preference_dataset(
    path_or_dataset: str = "openbmb/RLAIF-V-Dataset",
    split: str = "train",
    max_samples: int | None = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load the RLAIF-V preference dataset for VLM alignment.

    Each row contains an image, a question, and chosen/rejected text
    responses.  The function returns examples with
    ``chosen_conversation`` and ``rejected_conversation`` keys suitable
    for :class:`VLMPreferencePairDataset`.

    Dataset schema (``openbmb/RLAIF-V-Dataset``):
        - ``image``: PIL image
        - ``question``: user prompt
        - ``chosen``: preferred response text
        - ``rejected``: dispreferred response text

    Args:
        path_or_dataset: HF dataset path.
        split: Dataset split to load.
        max_samples: Optional cap on number of examples.

    Returns:
        List of preference-pair example dicts.
    """
    if split in ("validation", "test"):
        raise ValueError("RLAIF-V dataset only has a train split. Set eval_iters=0.")

    dataset = load_dataset(path_or_dataset, split=split)

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    def format_pair(example: Dict[str, Any]) -> Dict[str, Any] | None:
        image = example.get("image")
        question = example.get("question", "")
        chosen_text = example.get("chosen", "")
        rejected_text = example.get("rejected", "")
        if not chosen_text or not rejected_text:
            return None

        user_content = []
        if image is not None:
            user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": question})

        chosen_conversation = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": chosen_text}]},
        ]
        rejected_conversation = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": rejected_text}]},
        ]

        return {
            "chosen_conversation": chosen_conversation,
            "rejected_conversation": rejected_conversation,
        }

    formatted = (format_pair(ex) for ex in dataset)
    return [ex for ex in formatted if ex is not None]


def make_safewatch_dpo_dataset(
    train_path: str | None = None,
    eval_path: str | None = None,
    split: str = "train",
    max_samples: int | None = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Load a SafeWatch-Bench DPO JSONL dataset for video preference learning.

    Each JSONL line contains:
        - ``video``: path to ``.mp4`` file
        - ``messages``: HF-style conversation with ``{"type": "video", ...}``
        - ``chosen`` / ``rejected``: response text strings

    The function converts these into ``chosen_conversation`` /
    ``rejected_conversation`` dicts consumable by the Nemotron VL collate
    function (which expects ``{"type": "video", "path": "..."}``).

    Args:
        train_path: Path to the training JSONL file.
        eval_path: Path to the evaluation JSONL file.
        split: Which split to load (``"train"`` or ``"validation"``).
        max_samples: Optional cap on number of examples.

    Returns:
        List of preference-pair example dicts.
    """
    if split in ("train",):
        jsonl_path = train_path
    elif split in ("validation", "eval"):
        jsonl_path = eval_path
    else:
        raise ValueError(f"Unknown split '{split}'. Use 'train' or 'validation'.")

    if jsonl_path is None:
        raise ValueError(f"No data path provided for split='{split}'")

    examples: List[Dict[str, Any]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(row)
            if max_samples is not None and len(examples) >= max_samples:
                break

    def _convert_content(content_list: list, video_path: str) -> list:
        """Normalise HF message content to the format expected by collate."""
        converted = []
        for item in content_list:
            if item.get("type") == "video":
                converted.append({"type": "video", "path": video_path})
            else:
                converted.append(item)
        return converted

    results: List[Dict[str, Any]] = []
    for row in examples:
        video_path = row.get("video", "")
        chosen_text = row.get("chosen", "")
        rejected_text = row.get("rejected", "")
        if not chosen_text or not rejected_text or not video_path:
            continue

        messages = row.get("messages", [])
        if not messages:
            continue

        user_msg = messages[0]
        user_content = _convert_content(user_msg.get("content", []), video_path)

        chosen_conversation = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": chosen_text}]},
        ]
        rejected_conversation = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": rejected_text}]},
        ]

        results.append({
            "chosen_conversation": chosen_conversation,
            "rejected_conversation": rejected_conversation,
        })

    return results


def make_safewatch_sft_dataset(
    train_path: str | None = None,
    eval_path: str | None = None,
    split: str = "train",
    max_samples: int | None = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Load a SafeWatch-Bench SFT JSONL dataset for video supervised finetuning.

    Each JSONL line contains:
        - ``video``: path to ``.mp4`` file
        - ``messages``: HF-style ``[user, assistant]`` conversation

    The function converts these into ``{"conversation": [...]}`` dicts
    consumable by the Nemotron VL collate function.

    Args:
        train_path: Path to the training JSONL file.
        eval_path: Path to the evaluation JSONL file.
        split: Which split to load (``"train"`` or ``"validation"``).
        max_samples: Optional cap on number of examples.

    Returns:
        List of conversation example dicts.
    """
    if split in ("train",):
        jsonl_path = train_path
    elif split in ("validation", "eval"):
        jsonl_path = eval_path
    else:
        raise ValueError(f"Unknown split '{split}'. Use 'train' or 'validation'.")

    if jsonl_path is None:
        raise ValueError(f"No data path provided for split='{split}'")

    raw: List[Dict[str, Any]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw.append(json.loads(line))
            if max_samples is not None and len(raw) >= max_samples:
                break

    results: List[Dict[str, Any]] = []
    for row in raw:
        video_path = row.get("video", "")
        messages = row.get("messages", [])
        if not video_path or len(messages) < 2:
            continue

        conversation = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                converted = []
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "video":
                            converted.append({"type": "video", "path": video_path})
                        else:
                            converted.append(item)
                else:
                    converted.append({"type": "text", "text": str(content)})
                conversation.append({"role": "user", "content": converted})
            elif role == "assistant":
                text = content if isinstance(content, str) else str(content)
                conversation.append({"role": "assistant", "content": [{"type": "text", "text": text}]})

        if len(conversation) >= 2:
            results.append({"conversation": conversation})

    return results


def make_safewatch_dataset(
    jsonl_path: Optional[str] = None,
    train_jsonl_path: Optional[str] = None,
    eval_jsonl_path: Optional[str] = None,
    split: str = "train",
    max_examples: Optional[int] = None,
    require_video_exists: bool = True,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Load SafeWatch SFT JSONL into conversation examples for Nemotron Nano V2 VL.

    Supports both the HF-style ``messages`` schema and legacy LLaVA-style ``conversations``
    with a top-level ``video`` field. Normalises video paths via :func:`resolve_path`.

    Args:
        jsonl_path: Explicit JSONL path (fallback for both splits).
        train_jsonl_path: Train JSONL path (used when ``split="train"``).
        eval_jsonl_path: Eval/validation JSONL path.
        max_examples: Optional cap for quick experiments.
        require_video_exists: If True, drops rows whose referenced mp4 does not exist on disk.
    """
    from megatron.bridge.utils.common_utils import resolve_path as _resolve

    split_norm = str(split)
    if split_norm == "train":
        path = train_jsonl_path or jsonl_path
    elif split_norm in ("validation", "valid", "eval", "test"):
        path = eval_jsonl_path or jsonl_path
    else:
        path = jsonl_path
    if not path:
        raise ValueError(f"SafeWatch maker: no JSONL path for split={split_norm!r}")
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"SafeWatch JSONL not found: {path_obj}")

    def _norm_user_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        t = item.get("type")
        if t == "text" and isinstance(item.get("text"), str):
            return {"type": "text", "text": item["text"]}
        if t == "image":
            for key in ("image", "path", "url"):
                v = item.get(key)
                if isinstance(v, str) and v:
                    return {"type": "image", "image": v}
            return None
        if t == "video":
            for key in ("path", "video", "url"):
                v = item.get(key)
                if isinstance(v, str) and v:
                    p = str(_resolve(Path(v)))
                    if require_video_exists and not Path(p).is_file():
                        return None
                    return {"type": "video", "path": p}
            return None
        return None

    def _assistant_blocks(msg: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        content = msg.get("content")
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, list):
            blocks = [
                {"type": "text", "text": p["text"]}
                for p in content
                if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
            ]
            return blocks or None
        return None

    def _from_messages(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        messages = obj.get("messages")
        if not isinstance(messages, list) or not messages:
            return None
        conversation: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "user":
                raw_parts = msg.get("content")
                if not isinstance(raw_parts, list):
                    continue
                user_content = []
                for item in raw_parts:
                    normalized = _norm_user_item(item)
                    if normalized is None:
                        return None
                    user_content.append(normalized)
                if not user_content:
                    return None
                conversation.append({"role": "user", "content": user_content})
            elif role == "assistant":
                blocks = _assistant_blocks(msg)
                if not blocks:
                    return None
                conversation.append({"role": "assistant", "content": blocks})
        if len(conversation) < 2:
            return None
        return {"conversation": conversation}

    out: List[Dict[str, Any]] = []
    with path_obj.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL at %s:%d", path_obj, line_idx + 1)
                continue
            if not isinstance(obj, dict):
                continue

            example = _from_messages(obj)
            if example is None:
                convs = obj.get("conversations")
                video = obj.get("video")
                if not isinstance(video, str) or not video:
                    vids = obj.get("videos")
                    if isinstance(vids, list) and vids and isinstance(vids[0], str):
                        video = vids[0]
                if not (isinstance(convs, list) and isinstance(video, str) and video):
                    continue

                mp4_path = str(_resolve(Path(video)))
                if require_video_exists and not Path(mp4_path).is_file():
                    continue

                conversation_legacy: List[Dict[str, Any]] = []
                first_human = False
                for turn in convs:
                    if not isinstance(turn, dict):
                        continue
                    role = turn.get("from")
                    value = turn.get("value", "")
                    if not isinstance(value, str) or not value:
                        continue
                    if role == "human":
                        parts: List[Dict[str, Any]] = []
                        if not first_human:
                            parts.append({"type": "video", "path": mp4_path})
                            first_human = True
                        text = value.replace("<image>", "").replace("<video>", "").strip().lstrip("\n").rstrip()
                        parts.append({"type": "text", "text": text})
                        conversation_legacy.append({"role": "user", "content": parts})
                    elif role == "gpt":
                        conversation_legacy.append(
                            {"role": "assistant", "content": [{"type": "text", "text": value.strip()}]}
                        )

                if len(conversation_legacy) < 2:
                    continue
                example = {"conversation": conversation_legacy}

            out.append(example)
            if max_examples is not None and len(out) >= int(max_examples):
                break

    if not out:
        raise ValueError(f"SafeWatch maker produced zero examples from {path_obj}")
    return out
