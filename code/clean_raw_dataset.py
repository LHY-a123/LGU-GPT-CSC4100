#!/usr/bin/env python3
"""
Utility to clean raw CUHK(SZ) JSONL dumps.

Rules (tuned for school sites):
1. Deduplicate by URL.
2. Drop entries with empty/very short content (<120 chars after removing whitespace).
3. Drop entries whose content is >85% ASCII (often link/email directories).
4. Drop entries whose content contains >50% digits (usually目录或编号列表).
5. Drop navigation/TOC pages (lots of nav tokens + short body).
6. Remove `meta_description` and persist cleaned entries to data/cleaned/.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


CONTENT_MIN_LEN = 120
ASCII_RATIO_THRESHOLD = 0.85
DIGIT_RATIO_THRESHOLD = 0.30
NAV_TOKEN_THRESHOLD = 5
NAV_TOKENS: Set[str] = {
    "概览",
    "学院新闻",
    "分页",
    "下一页",
    "上一页",
    "查看更多",
    "页面",
    "了解更多",
    "活动预告",
    "学院手册",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw CUHKSZ JSONL pages.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more JSONL files under data/raw/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cleaned"),
        help="Directory to store cleaned JSONL files (default: data/cleaned).",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] {path}:{line_no} JSON decode error: {exc}")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def content_without_ws(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def is_ascii_heavy(text: str) -> bool:
    if not text:
        return False
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return ascii_chars / len(text) >= ASCII_RATIO_THRESHOLD


def is_digit_heavy(text: str) -> bool:
    if not text:
        return False
    digit_chars = sum(1 for ch in text if ch.isdigit())
    return digit_chars / len(text) >= DIGIT_RATIO_THRESHOLD


def looks_like_nav_page(content: str) -> bool:
    hits = sum(token in content for token in NAV_TOKENS)
    return hits >= NAV_TOKEN_THRESHOLD and len(content) < 1000


def clean_records(records: Iterable[Dict]) -> List[Dict]:
    seen_urls: Set[str] = set()
    cleaned: List[Dict] = []

    for record in records:
        url = record.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        content = (record.get("content") or "").strip()
        meta_desc = (record.get("meta_description") or "").strip()
        
        # 合并 meta_description 到 content
        if meta_desc:
            content = f"{meta_desc}\n\n{content}"
        
        compact = content_without_ws(content)
        if len(compact) < CONTENT_MIN_LEN:
            continue
        if is_ascii_heavy(compact):
            continue
        if is_digit_heavy(compact):
            continue
        if looks_like_nav_page(content):
            continue

        record = dict(record)
        record["content"] = content
        record.pop("meta_description", None)
        cleaned.append(record)

    return cleaned


def derive_output_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / input_path.name


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0

    for input_str in args.inputs:
        input_path = Path(input_str)
        if not input_path.exists():
            print(f"[WARN] Skip missing file: {input_path}")
            continue
        records = list(read_jsonl(input_path))
        cleaned = clean_records(records)
        output_path = derive_output_path(input_path, args.output_dir)
        with output_path.open("w", encoding="utf-8") as file:
            for record in cleaned:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[INFO] cleaned {input_path} -> {output_path} "
            f"({len(cleaned)} / {len(records)} records kept)"
        )
        total_in += len(records)
        total_out += len(cleaned)

    if total_in:
        kept_ratio = total_out / total_in * 100
        print(f"[INFO] Overall: kept {total_out}/{total_in} ({kept_ratio:.1f}%)")
    else:
        print("[WARN] No valid input records processed.")


if __name__ == "__main__":
    main()

