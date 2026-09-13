"""Build single-turn Tong Xiangyu SFT samples from extracted dialogues."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


TONG_NAMES = {
    "佟湘玉",
    "湘玉",
    "佟掌柜",
    "佟掌柜的",
    # The supplied screenplay uses 掌柜 as an explicit speaker label.
    "掌柜",
}
UNKNOWN_NAMES = {"UNKNOWN", "未知", "不明"}


def canonicalize_speaker(speaker: str) -> str:
    name = re.sub(r"\s+", "", speaker)
    if name in TONG_NAMES:
        return "佟湘玉"
    if name.upper() == "UNKNOWN" or name in UNKNOWN_NAMES:
        return "UNKNOWN"
    return speaker.strip()


def build_sample(turns: list[dict[str, object]], index: int) -> dict[str, str] | None:
    current = turns[index]
    if index == 0 or current["speaker"] != "佟湘玉":
        return None

    previous = turns[index - 1]
    if previous["speaker"] == "佟湘玉":
        return None
    if previous["speaker"] == "UNKNOWN":
        return None
    if previous["scene_id"] != current["scene_id"]:
        return None

    return {
        "instruction": str(previous["content"]),
        "input": "",
        "output": str(current["content"]),
    }


def read_dialogues(path: Path) -> list[dict[str, object]]:
    required = {"scene_id", "turn_id", "speaker", "content"}
    rows: list[dict[str, object]] = []

    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是合法 JSON: {exc}") from exc
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError(f"第 {line_number} 行缺少中间数据字段")
            if not isinstance(row["speaker"], str) or not isinstance(row["content"], str):
                raise ValueError(f"第 {line_number} 行 speaker/content 必须是字符串")

            normalized = dict(row)
            normalized["speaker"] = canonicalize_speaker(row["speaker"])
            normalized["content"] = row["content"].strip()
            rows.append(normalized)

    return rows


def similarity_signature(sample: dict[str, str]) -> str:
    text = sample["instruction"] + "\u0000" + sample["output"]
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text, flags=re.UNICODE).lower()


def _candidate_bucket_keys(signature: str) -> Iterable[tuple[str, int]]:
    prefix = signature[:8]
    length_bucket = len(signature) // 12
    for bucket in range(max(0, length_bucket - 1), length_bucket + 2):
        yield prefix, bucket


def deduplicate_samples(
    samples: Iterable[dict[str, str]], similarity_threshold: float
) -> tuple[list[dict[str, str]], int]:
    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold 必须在 (0, 1] 范围内")

    kept: list[dict[str, str]] = []
    exact_seen: set[str] = set()
    buckets: dict[tuple[str, int], list[str]] = defaultdict(list)
    removed = 0

    for sample in samples:
        signature = similarity_signature(sample)
        if signature in exact_seen:
            removed += 1
            continue

        is_similar = False
        if similarity_threshold < 1.0:
            candidates: list[str] = []
            for key in _candidate_bucket_keys(signature):
                candidates.extend(buckets.get(key, ()))
            is_similar = any(
                SequenceMatcher(None, signature, candidate).ratio()
                >= similarity_threshold
                for candidate in candidates
            )

        if is_similar:
            removed += 1
            continue

        kept.append(sample)
        exact_seen.add(signature)
        own_key = (signature[:8], len(signature) // 12)
        buckets[own_key].append(signature)

    return kept, removed


def build_dataset(turns: list[dict[str, object]]) -> list[dict[str, str]]:
    samples = []
    for index in range(len(turns)):
        sample = build_sample(turns, index)
        if sample is not None:
            samples.append(sample)
    return samples


def write_jsonl(rows: Iterable[dict[str, str]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="对话 JSONL")
    parser.add_argument("--output", type=Path, required=True, help="SFT JSONL")
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.96,
        help="高度相似去重阈值；设为 1 时仅删除规范化后完全重复项",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    turns = read_dialogues(args.input)
    samples = build_dataset(turns)
    samples, removed = deduplicate_samples(samples, args.similarity_threshold)
    count = write_jsonl(samples, args.output)

    if count == 0:
        raise ValueError("没有构造出训练样本，请检查角色名称与场景划分")

    print(f"输入对话: {len(turns)} 条")
    print(f"输出样本: {count} 条")
    print(f"删除重复或高度相似样本: {removed} 条")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
