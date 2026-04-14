#!/usr/bin/env python3
"""
使用 DeepSeek API 从 cleaned 课程数据并行生成 QA Dataset。

流程：
1. 读取 cleaned JSONL（默认 data/cleaned/cuhksz_courses.jsonl）
2. 按 API key 数量切片课程集合，ProcessPoolExecutor 并行调用 DeepSeek
3. 将 QA 输出写入 data/qa/*.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from openai import OpenAI
from tqdm import tqdm

QA_MODEL = "deepseek-reasoner"
QA_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
SLEEP_SECONDS = 1.5
MAX_RETRIES = 3
DEFAULT_API_KEYS = [
    "sk-2db26a5af44b43a7af0b3e4a55cd29e6",
    "sk-8ce77f4caa9b45bfbf741a368be18840",
    "sk-42892705f5a642b18ff533680c380162",
    "sk-ad2fffdb5d194d0b95ecab1e6b37b370",
    "sk-3930f5f79ff1450eb5c6b1ac582bc246",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 cleaned 课程数据生成 QA Dataset（DeepSeek 并行）"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/cleaned/cuhksz_courses.jsonl"),
        help="cleaned JSONL 输入路径（默认：data/cleaned/cuhksz_courses.jsonl）",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="可选：采样课程数量（默认：全部）",
    )
    return parser.parse_args()


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] {file_path}:{line_no} 解析失败：{exc}")
    return records


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def extract_course_content(course: Dict[str, Any]) -> str:
    content = (course.get("content") or "").strip()
    if content:
        return content
    parts: List[str] = []
    info_parts: List[str] = []
    if course.get("code"):
        info_parts.append(f"课程代码：{course['code']}")
    if course.get("title"):
        info_parts.append(f"课程名称：{course['title']}")
    if course.get("level"):
        info_parts.append(f"学历层次：{course['level']}")
    if course.get("school"):
        info_parts.append(f"所属学院：{course['school']}")
    if course.get("credits"):
        info_parts.append(f"学分：{course['credits']}")
    if course.get("grading"):
        info_parts.append(f"评分方式：{course['grading']}")
    if course.get("teaching_mode"):
        info_parts.append(f"教学方式：{course['teaching_mode']}")
    if course.get("term"):
        info_parts.append(f"开课学期：{course['term']}")
    if course.get("category"):
        info_parts.append(f"课程类别：{course['category']}")
    if info_parts:
        parts.append("\n".join(info_parts))
    if course.get("description"):
        parts.append(f"\n课程描述：\n{course['description']}")
    return "\n\n".join(parts).strip()


def generate_qa_with_llm(
    course: Dict[str, Any],
    client: OpenAI,
    model: str,
    max_retries: int,
) -> List[Dict[str, Any]]:
    system_prompt = (
        "你是一名对香港中文大学（深圳）课程信息非常了解的学术顾问。"
        "请根据提供的课程信息，生成2-3个高质量的问答对。"
        "问题应该涵盖课程的基本信息、课程内容和学习要求等方面。"
        "答案必须完全基于提供的课程信息，不要编造内容。"
    )
    content = extract_course_content(course)
    if not content:
        return []
    user_prompt = f"""请根据以下课程信息生成2-3个问答对：

{content}

要求：
1. 问题应该自然、信息密度高，涵盖课程代码、名称、所属学院、学分、课程描述等关键信息
2. 答案必须完全基于提供的课程信息
3. 每个答案长度在50-200字之间
4. 输出 JSON 格式：
{{
  "qas": [
    {{
      "question": "问题文本",
      "answer": "答案文本",
      "answer_type": "descriptive|procedural|comparative|eligibility|other",
      "confidence_note": "说明答案依据"
    }}
  ]
}}"""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
            )
            raw_text = response.choices[0].message.content.strip()
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].lstrip("`").startswith("json"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()
            data = json.loads(raw_text)
            qas = data.get("qas", [])
            validated: List[Dict[str, Any]] = []
            for qa in qas:
                if isinstance(qa, dict) and qa.get("question") and qa.get("answer"):
                    validated.append(
                        {
                            "question": qa["question"].strip(),
                            "answer": qa["answer"].strip(),
                            "answer_type": qa.get("answer_type", "descriptive"),
                            "confidence_note": qa.get("confidence_note", "").strip(),
                        }
                    )
            return validated
        except Exception as exc:
            if attempt < max_retries:
                wait = attempt * 2
                print(
                    f"[WARN] 生成 QA 失败（{course.get('code', 'unknown')}，第{attempt}次）：{exc}，{wait}s 后重试..."
                )
                time.sleep(wait)
            else:
                print(
                    f"[ERROR] 生成 QA 失败（{course.get('code', 'unknown')}）：达到最大重试次数"
                )
                return []
    return []


def split_courses(courses: Sequence[Dict[str, Any]], parts: int) -> List[List[Dict[str, Any]]]:
    if parts <= 0:
        return [list(courses)]
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(parts)]
    for idx, course in enumerate(courses):
        buckets[idx % parts].append(course)
    return buckets


def process_courses_with_api_key(
    courses: List[Dict[str, Any]],
    api_key: str,
    qa_model: str,
    qa_base_url: str,
    sleep_seconds: float,
    max_retries: int,
) -> List[str]:
    if not courses:
        return []
    client = OpenAI(api_key=api_key, base_url=qa_base_url)
    lines: List[str] = []
    for course in courses:
        if not course.get("code") and not course.get("title"):
            continue
        qas = generate_qa_with_llm(course, client, qa_model, max_retries)
        for qa in qas:
            entry = {
                "question": qa["question"],
                "answer": qa["answer"],
                "answer_type": qa.get("answer_type", "descriptive"),
                "confidence_note": qa.get("confidence_note", ""),
                "source": {
                    "code": course.get("code"),
                    "title": course.get("title"),
                    "url": course.get("url"),
                    "school": course.get("school"),
                },
            }
            lines.append(json.dumps(entry, ensure_ascii=False))
        time.sleep(sleep_seconds)
    return lines


def load_qa_api_keys() -> List[str]:
    env_value = os.getenv("DEEPSEEK_API_KEYS")
    if env_value:
        parsed = [key.strip() for key in env_value.split(",") if key.strip()]
        if parsed:
            return parsed
    if DEFAULT_API_KEYS:
        print("[INFO] 使用脚本内置的 DeepSeek API keys")
        return DEFAULT_API_KEYS
    raise RuntimeError("未提供 DeepSeek API key，请通过 --qa-api-keys 或环境变量设置。")


def save_qa_dataset(
    courses: List[Dict[str, Any]],
    output_path: Path,
    api_keys: Sequence[str],
) -> None:
    ensure_parent_dir(output_path)
    worker_count = len(api_keys)
    if worker_count == 0:
        raise RuntimeError("需要至少一个 DeepSeek API key 才能生成 QA 数据。")
    chunks = split_courses(courses, worker_count)
    print(f"[INFO] 使用 {worker_count} 个进程并行生成 QA 数据")
    total_qas = 0
    with output_path.open("w", encoding="utf-8") as fp:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    process_courses_with_api_key,
                    chunk,
                    api_key,
                    QA_MODEL,
                    QA_BASE_URL,
                    SLEEP_SECONDS,
                    MAX_RETRIES,
                )
                for chunk, api_key in zip(chunks, api_keys)
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="并行生成 QA",
                unit="worker",
            ):
                lines = future.result()
                for line in lines:
                    fp.write(line + "\n")
                total_qas += len(lines)
    print(f"[INFO] QA 数据集已保存：{output_path}，共 {total_qas} 条问答对")


def maybe_sample_courses(
    courses: List[Dict[str, Any]],
    sample_size: Optional[int],
) -> List[Dict[str, Any]]:
    if not sample_size or sample_size >= len(courses):
        return courses
    random.seed(42)
    sampled = random.sample(courses, sample_size)
    print(f"[INFO] 采样 {sample_size} 门课程（共 {len(courses)}）用于生成 QA")
    return sampled


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        print(f"[ERROR] 输入文件不存在：{args.input}")
        sys.exit(1)
    courses = read_jsonl(args.input)
    if not courses:
        print(f"[ERROR] 输入文件 {args.input} 没有有效课程记录")
        sys.exit(1)
    courses_for_qa = maybe_sample_courses(courses, args.sample_size)
    api_keys = load_qa_api_keys()
    qa_output = Path("data/qa") / args.input.name
    save_qa_dataset(courses_for_qa, qa_output, api_keys=api_keys)
    print("[INFO] QA 生成完成！")


if __name__ == "__main__":
    main()


