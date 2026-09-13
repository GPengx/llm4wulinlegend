"""Validate SFT JSONL structure and conservative single-turn constraints."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REQUIRED_FIELDS = {"instruction", "input", "output"}
OUTPUT_SPEAKER_PREFIX = re.compile(
    r"^\s*(?:佟湘玉|湘玉|佟掌柜(?:的)?|掌柜)\s*[：:]"
)


def validate_sample(sample: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(sample, dict):
        return ["sample must be a JSON object"]

    if not REQUIRED_FIELDS.issubset(sample):
        errors.append("missing required fields")
    if not isinstance(sample.get("instruction"), str):
        errors.append("instruction must be a string")
    if sample.get("input") != "":
        errors.append("input must be an empty string")
    if not isinstance(sample.get("output"), str):
        errors.append("output must be a string")

    instruction = sample.get("instruction")
    output = sample.get("output")
    if isinstance(instruction, str):
        if not instruction.strip():
            errors.append("instruction is empty")
        if "\n" in instruction or "\r" in instruction:
            errors.append("instruction must be single-line")
    if isinstance(output, str):
        if not output.strip():
            errors.append("output is empty")
        if "\n" in output or "\r" in output:
            errors.append("output must be single-line")
        if OUTPUT_SPEAKER_PREFIX.match(output):
            errors.append("output must not contain a speaker prefix")

    return errors


def sample_signature(sample: dict[str, object]) -> str | None:
    instruction = sample.get("instruction")
    output = sample.get("output")
    if not isinstance(instruction, str) or not isinstance(output, str):
        return None
    return re.sub(r"\s+", "", instruction + "\u0000" + output)


def validate_file(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    seen: dict[str, int] = {}
    count = 0

    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                errors.append(f"第 {line_number} 行: empty line")
                continue
            count += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"第 {line_number} 行: invalid JSON ({exc.msg})")
                continue

            for error in validate_sample(sample):
                errors.append(f"第 {line_number} 行: {error}")

            if isinstance(sample, dict):
                signature = sample_signature(sample)
                if signature is not None:
                    if signature in seen:
                        errors.append(
                            f"第 {line_number} 行: duplicate of line {seen[signature]}"
                        )
                    else:
                        seen[signature] = line_number

    if count == 0:
        errors.append("数据集为空")
    return count, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="待校验 SFT JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    count, errors = validate_file(args.input)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"校验失败：{len(errors)} 个错误", file=sys.stderr)
        raise SystemExit(1)

    print(f"校验通过：{count} 条样本")


if __name__ == "__main__":
    main()
