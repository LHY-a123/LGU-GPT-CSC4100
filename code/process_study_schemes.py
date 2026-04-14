#!/usr/bin/env python3
"""
将 study_schemes 目录下的 PDF 解析为结构化 JSONL，以便训练校情 LLM。
输出按 chunk（文本或表格行）组织，带上院系/方案/页码等元数据。
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pdfplumber


TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "intersection_y_tolerance": 5,
    "intersection_x_tolerance": 5,
    "snap_tolerance": 3,
}


@dataclass
class Chunk:
    content: str
    chunk_type: str  # "text" | "table_row"
    faculty: str
    scheme: str
    pdf_path: str
    page_number: int
    table_idx: int | None = None
    row_idx: int | None = None

    def to_dict(self) -> dict:
        data = {
            "content": self.content,
            "chunk_type": self.chunk_type,
            "faculty": self.faculty,
            "scheme": self.scheme,
            "pdf_path": self.pdf_path,
            "page_number": self.page_number,
        }
        if self.table_idx is not None:
            data["table_idx"] = self.table_idx
        if self.row_idx is not None:
            data["row_idx"] = self.row_idx
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured chunks from study scheme PDFs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/study_schemes"),
        help="存放 study scheme PDF 的根目录。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/study_schemes_processed/chunks.jsonl"),
        help="输出 JSONL 路径。",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=800,
        help="单个文本 chunk 最多字符数，超出将切分。",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=30,
        help="低于该字符数的文本块将被丢弃，避免噪声。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印解析统计，不写文件。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 个 PDF，方便调试。",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    # 合并多余空白
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int) -> List[str]:
    if not text:
        return []
    pieces: List[str] = []
    buffer: List[str] = []
    current_len = 0
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        para_len = len(paragraph)
        if current_len + para_len <= max_chars:
            buffer.append(paragraph)
            current_len += para_len + 2
            continue
        if buffer:
            pieces.append("\n\n".join(buffer).strip())
        if para_len > max_chars:
            for i in range(0, para_len, max_chars):
                pieces.append(paragraph[i : i + max_chars])
            buffer = []
            current_len = 0
        else:
            buffer = [paragraph]
            current_len = para_len
    if buffer:
        pieces.append("\n\n".join(buffer).strip())
    return pieces


def table_rows_to_chunks(
    rows: Sequence[Sequence[str]],
    faculty: str,
    scheme: str,
    pdf_path: str,
    page_number: int,
    table_idx: int,
) -> Iterable[Chunk]:
    cleaned_rows: List[List[str]] = [
        [cell.strip() if isinstance(cell, str) else "" for cell in row]
        for row in rows
    ]
    if not cleaned_rows:
        return []
    header = cleaned_rows[0]
    has_header = any(cell for cell in header)
    data_rows = cleaned_rows[1:] if has_header else cleaned_rows
    for row_idx, row in enumerate(data_rows, start=1 if has_header else 0):
        pairs = []
        for col_idx, value in enumerate(row):
            col_name = header[col_idx] if has_header else f"列{col_idx+1}"
            if not value and not col_name:
                continue
            pairs.append(f"{col_name or f'列{col_idx+1}'}：{value.strip()}")
        if not pairs:
            continue
        sentence = f"{scheme}（{faculty}）第{page_number}页表格："
        sentence += "；".join(pairs)
        yield Chunk(
            content=sentence,
            chunk_type="table_row",
            faculty=faculty,
            scheme=scheme,
            pdf_path=pdf_path,
            page_number=page_number,
            table_idx=table_idx,
            row_idx=row_idx,
        )


def extract_chunks_from_pdf(
    pdf_path: Path, max_chars: int, min_chars: int
) -> List[Chunk]:
    faculty = pdf_path.parent.name
    scheme = pdf_path.stem
    chunks: List[Chunk] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_number = page.page_number
            text = clean_text(page.extract_text(layout=True) or "")
            for text_chunk in chunk_text(text, max_chars=max_chars):
                if len(text_chunk) < min_chars:
                    continue
                chunks.append(
                    Chunk(
                        content=text_chunk,
                        chunk_type="text",
                        faculty=faculty,
                        scheme=scheme,
                        pdf_path=str(pdf_path),
                        page_number=page_number,
                    )
                )
            tables = page.extract_tables(table_settings=TABLE_SETTINGS) or []
            for table_idx, table in enumerate(tables):
                chunks.extend(
                    table_rows_to_chunks(
                        table,
                        faculty=faculty,
                        scheme=scheme,
                        pdf_path=str(pdf_path),
                        page_number=page_number,
                        table_idx=table_idx,
                    )
                )
    return chunks


def iter_pdfs(root: Path, limit: int | None) -> Iterable[Path]:
    pdfs = sorted(root.rglob("*.pdf"))
    if limit is not None:
        pdfs = pdfs[:limit]
    for pdf in pdfs:
        yield pdf


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pdf_files = list(iter_pdfs(args.input_dir, args.limit))
    if not pdf_files:
        logging.error("未在 %s 找到 PDF 文件", args.input_dir)
        return
    logging.info("待处理 PDF：%d", len(pdf_files))

    total_chunks = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        logging.info("Dry-run 模式，将仅统计但不写文件。")
    else:
        output_fp = args.output.open("w", encoding="utf-8")

    try:
        for pdf_path in pdf_files:
            chunks = extract_chunks_from_pdf(
                pdf_path=pdf_path, max_chars=args.max_chars, min_chars=args.min_chars
            )
            total_chunks += len(chunks)
            logging.info(
                "解析 %s -> %d chunks", pdf_path.relative_to(args.input_dir), len(chunks)
            )
            if not args.dry_run:
                for chunk in chunks:
                    json.dump(chunk.to_dict(), output_fp, ensure_ascii=False)
                    output_fp.write("\n")
    finally:
        if not args.dry_run:
            output_fp.close()

    logging.info(
        "共处理 %d 个 PDF，生成 %d 个 chunks。输出：%s",
        len(pdf_files),
        total_chunks,
        args.output if not args.dry_run else "dry-run 未写出",
    )


if __name__ == "__main__":
    main()


