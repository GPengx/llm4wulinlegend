"""Detect the source encoding and normalize a text file to UTF-8."""

from __future__ import annotations

import argparse
from pathlib import Path


COMMON_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def read_text_with_detection(path: Path) -> tuple[str, str]:
    """Decode *path* without silently dropping or replacing characters."""
    raw = path.read_bytes()

    for encoding in COMMON_ENCODINGS:
        try:
            return raw.decode(encoding, errors="strict"), encoding
        except UnicodeDecodeError:
            continue

    try:
        from charset_normalizer import from_bytes
    except ImportError as exc:  # pragma: no cover - only reached without deps
        raise UnicodeError(
            "无法使用常见编码解码；请安装 charset-normalizer 后重试"
        ) from exc

    match = from_bytes(raw).best()
    if match is None or match.encoding is None:
        raise UnicodeError("无法识别 TXT 文件编码")

    return str(match), match.encoding


def normalize_text(text: str) -> str:
    """Normalize line endings and remove NUL/BOM characters."""
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\x00", "")
        .replace("\ufeff", "")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="源 TXT 文件")
    parser.add_argument("--output", type=Path, required=True, help="UTF-8 输出文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    text, encoding = read_text_with_detection(args.input)
    clean_text = normalize_text(text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(clean_text, encoding="utf-8", newline="\n")

    print(f"检测到原始编码: {encoding}")
    print(f"已写入 UTF-8 文件: {args.output}")
    print(f"字符数: {len(clean_text)}")


if __name__ == "__main__":
    main()

