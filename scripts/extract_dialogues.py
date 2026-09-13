"""Extract speaker-labelled dialogue lines from a normalized screenplay TXT."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


LINE_PATTERN = re.compile(r"^\s*([^：:\n]{1,20})[：:]\s*(.+?)\s*$")
SPEAKER_ACTION_PATTERN = re.compile(r"\s*[（(【\[].*?[）)】\]]\s*")
CHAPTER_PATTERN = re.compile(
    r"^第[〇零一二三四五六七八九十百两0-9]+(?:回|集)(?:\s|$)"
)
BRACKETED_SCENE_PATTERN = re.compile(r"^[【\[].+[】\]]$")
NAMED_SCENE_PATTERN = re.compile(r"^(?:场景|地点|时间)[：:].+")


def clean_speaker(speaker: str) -> str:
    """Remove stage directions attached to a speaker label."""
    return SPEAKER_ACTION_PATTERN.sub("", speaker).strip()


def is_scene_boundary(line: str) -> bool:
    """Return whether a non-dialogue line explicitly marks a new scene."""
    return bool(
        CHAPTER_PATTERN.match(line)
        or BRACKETED_SCENE_PATTERN.match(line)
        or NAMED_SCENE_PATTERN.match(line)
    )


def extract_dialogues(lines: Iterable[str]) -> list[dict[str, object]]:
    """Extract dialogue turns and preserve explicit scene boundaries.

    Stage directions are ignored because they commonly occur between two
    adjacent spoken turns. Chapters and explicit scene/location/time markers
    advance the scene id so samples cannot cross those boundaries.
    """
    turns: list[dict[str, object]] = []
    scene_number = 1
    turn_id = 0

    for raw_line in lines:
        line = raw_line.strip()
        if is_scene_boundary(line):
            if turn_id > 0:
                scene_number += 1
                turn_id = 0
            continue

        match = LINE_PATTERN.match(line) if line else None

        if match is None:
            continue

        speaker = clean_speaker(match.group(1))
        content = match.group(2).strip()
        if not speaker or not content:
            continue

        turn_id += 1
        turns.append(
            {
                "scene_id": f"scene_{scene_number:04d}",
                "turn_id": turn_id,
                "speaker": speaker,
                "content": content,
            }
        )

    return turns


def write_jsonl(rows: Iterable[dict[str, object]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 剧本文本")
    parser.add_argument("--output", type=Path, required=True, help="对话 JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    with args.input.open("r", encoding="utf-8", errors="strict") as handle:
        turns = extract_dialogues(handle)

    count = write_jsonl(turns, args.output)
    if count == 0:
        raise ValueError("没有提取到对话，请检查文本格式和编码")

    scene_count = len({turn["scene_id"] for turn in turns})
    print(f"已提取 {count} 条对话，涉及 {scene_count} 个保守场景片段")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
