# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Logic for saving file format

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from datacreek.utils.format_converter import (
    to_alpaca,
    to_chatml,
    to_fine_tuning,
    to_hf_dataset,
    to_jsonl,
)
from datacreek.utils.llm_processing import convert_to_conversation_format


def _format_pairs(qa_pairs: List[Dict[str, str]], fmt: str) -> Any:
    """Return formatted data for in-memory conversion."""

    if fmt == "jsonl":
        return "\n".join(json.dumps(p, ensure_ascii=False) for p in qa_pairs)
    if fmt == "alpaca":
        return json.dumps(
            [
                {"instruction": p["question"], "input": "", "output": p["answer"]}
                for p in qa_pairs
            ],
            indent=2,
        )
    if fmt in {"ft", "chatml"}:
        messages_key = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": p["question"]},
                {"role": "assistant", "content": p["answer"]},
            ]
            for p in qa_pairs
        ]
        if fmt == "ft":
            return json.dumps([{"messages": m} for m in messages_key], indent=2)
        return json.dumps([{"messages": m} for m in messages_key], indent=2)

    raise ValueError(f"Unknown format type: {fmt}")


def convert_format(
    input_data: Any,
    output_path: Optional[str],
    format_type: str,
    config: Optional[Dict[str, Any]] = None,
    storage_format: str = "json",
) -> Any:
    """Convert data to different formats

    Args:
        input_path: Path to the input file
        output_path: Path to save the output
        format_type: Output format (jsonl, alpaca, ft, chatml)
        config: Configuration dictionary
        storage_format: Storage format, either "json" or "hf" (Hugging Face dataset)

    Returns:
        Path to the output file or directory
    """
    supported = {"jsonl", "alpaca", "ft", "chatml"}
    if format_type not in supported:
        raise ValueError(f"Unknown format type: {format_type}")

    if isinstance(input_data, str):
        if os.path.exists(input_data):
            with open(input_data, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(input_data)
    else:
        data = input_data

    # Extract data based on known structures
    # Try to handle the case where we have QA pairs or conversations
    if "qa_pairs" in data:
        qa_pairs = data.get("qa_pairs", [])
    elif "filtered_pairs" in data:
        qa_pairs = data.get("filtered_pairs", [])
    elif "conversations" in data:
        conversations = data.get("conversations", [])
        qa_pairs = []
        for conv in conversations:
            if len(conv) >= 3 and conv[1]["role"] == "user" and conv[2]["role"] == "assistant":
                qa_pairs.append({"question": conv[1]["content"], "answer": conv[2]["content"]})
    else:
        # If the file is just an array of objects, check if they look like QA pairs
        if isinstance(data, list):
            qa_pairs = []
            for item in data:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    qa_pairs.append(item)
        else:
            raise ValueError("Unrecognized data format - expected QA pairs or conversations")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # When using HF dataset storage format
    if storage_format == "hf":
        # Prepare data in list-of-dict structure
        if format_type == "jsonl":
            formatted_pairs = qa_pairs
        else:
            formatted_pairs = json.loads(_format_pairs(qa_pairs, format_type))

        # Save as HF dataset (Arrow format)
        if output_path:
            return to_hf_dataset(formatted_pairs, output_path)
        else:
            raise ValueError("HF dataset export requires output path")

    # Standard JSON file storage format
    else:
        if format_type == "jsonl" and output_path:
            return to_jsonl(qa_pairs, output_path)
        if format_type == "alpaca" and output_path:
            return to_alpaca(qa_pairs, output_path)
        if format_type == "ft" and output_path:
            return to_fine_tuning(qa_pairs, output_path)
        if format_type == "chatml" and output_path:
            return to_chatml(qa_pairs, output_path)

        return _format_pairs(qa_pairs, format_type)
