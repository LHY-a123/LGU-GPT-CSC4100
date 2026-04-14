#!/usr/bin/env python3
"""
将 cuhksz_courses.json 转换为 cleaned & RAG 数据

功能：
1. 使用本地 vLLM 翻译（英文 -> 简体中文）并进行繁简转换
2. 复用 rule-base 过滤逻辑生成 cleaned JSONL
3. 基于 cleaned 记录构建 RAG JSONL
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from openai import OpenAI
from tqdm import tqdm

try:
    import zhconv
except ImportError:
    print("[WARN] zhconv 未安装，将无法进行繁简转换。请运行: pip install zhconv")
    zhconv = None


CONTENT_MIN_LEN = 120
ASCII_RATIO_THRESHOLD = 0.85
DIGIT_RATIO_THRESHOLD = 0.30
NAV_TOKEN_THRESHOLD = 5
NAV_TOKENS = {
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

LOCAL_MODEL = "Qwen/Qwen2.5-14B-Instruct"
LOCAL_BASE_URL = "http://localhost:8000/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将课程数据翻译为简体中文并生成 cleaned / RAG 数据集"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/cuhksz_courses.json"),
        help="输入的课程 JSON 文件路径（默认：data/cuhksz_courses.json）",
    )
    return parser.parse_args()


def load_courses(input_path: Path) -> List[Dict[str, Any]]:
    """加载课程 JSON 文件"""
    print(f"[INFO] 加载课程数据：{input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        courses = json.load(f)
    print(f"[INFO] 共加载 {len(courses)} 门课程")
    return courses


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def is_mostly_english(text: str) -> bool:
    """检测文本是否主要是英文"""
    if not text:
        return False
    # 统计中文字符和英文字符数量
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    total_chars = chinese_chars + english_chars
    if total_chars == 0:
        return False
    # 如果英文占比超过70%，认为是英文文本
    return english_chars / total_chars > 0.7


def traditional_to_simplified(text: str) -> str:
    """将繁体中文转换为简体中文"""
    if not text or zhconv is None:
        return text
    try:
        return zhconv.convert(text, 'zh-cn')
    except Exception as e:
        print(f"[WARN] 繁简转换失败: {e}")
        return text


def simplify_field(value: Optional[str]) -> str:
    if value is None:
        return ""
    return traditional_to_simplified(value.strip())


def translate_to_chinese(
    text: str,
    client: Optional[OpenAI],
    model: str = "Qwen/Qwen2.5-14B-Instruct",
    max_retries: int = 3,
) -> str:
    """将英文文本翻译为简体中文"""
    if not text or not client:
        return text
    
    if not is_mostly_english(text):
        # 如果不是英文，只做繁简转换
        return traditional_to_simplified(text)
    
    # 使用本地 vLLM 翻译
    prompt = f"""请将以下英文课程描述翻译成简体中文。要求：
1. 翻译准确、流畅
2. 保持专业术语的准确性
3. 使用简体中文
4. 只输出翻译结果，不要添加任何解释

英文内容：
{text}"""
    
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一名专业的翻译助手，擅长将英文课程描述翻译成简体中文。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            translated = response.choices[0].message.content.strip()
            # 移除可能的代码块标记
            if translated.startswith("```"):
                lines = translated.splitlines()
                if lines[0].lstrip("`").startswith(("中文", "翻译", "简体")):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                translated = "\n".join(lines).strip()
            return translated
        except Exception as e:
            if attempt < max_retries:
                wait = attempt * 2
                print(f"[WARN] 翻译失败（第{attempt}次）：{e}，{wait}s 后重试...")
                time.sleep(wait)
            else:
                print(f"[ERROR] 翻译失败：达到最大重试次数，返回原文")
                return text
    
    return text


def translate_course_record(
    course: Dict[str, Any],
    client: OpenAI,
    model: str,
) -> Dict[str, Any]:
    """使用本地 vLLM 翻译课程字段"""
    translated: Dict[str, Any] = dict(course)
    translated["title"] = simplify_field(course.get("title"))
    translated["level"] = simplify_field(course.get("level"))
    translated["school"] = simplify_field(course.get("school"))
    translated["grading"] = simplify_field(course.get("grading"))
    translated["teaching_mode"] = simplify_field(course.get("teaching_mode"))
    translated["term"] = simplify_field(course.get("term"))
    translated["category"] = simplify_field(course.get("category"))
    
    description = (course.get("description") or "").strip()
    if description:
        if is_mostly_english(description):
            translated["description"] = translate_to_chinese(description, client, model)
        else:
            translated["description"] = traditional_to_simplified(description)
    else:
        translated["description"] = ""
    return translated


def build_course_content(course: Dict[str, Any]) -> str:
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


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: Path) -> None:
    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def course_to_rag_document(course: Dict[str, Any]) -> Dict[str, Any]:
    """将清洗后的课程数据转换为 RAG 文档"""
    content = (course.get("content") or "").strip()
    word_count = len(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', content))
    title_parts = []
    if course.get("code"):
        title_parts.append(course["code"])
    if course.get("title"):
        title_parts.append(course["title"])
    final_title = " - ".join(title_parts) if title_parts else "未知课程"
    return {
        "title": final_title,
        "content": content,
        "word_count": word_count,
        "url": course.get("url", ""),
        "metadata": {
            "code": course.get("code"),
            "level": course.get("level"),
            "school": course.get("school"),
            "credits": course.get("credits"),
            "category": course.get("category"),
        },
    }


def save_rag_documents(
    courses: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """保存 RAG 文档为 JSONL 格式"""
    print(f"[INFO] 生成 RAG 文档：{output_path}")
    ensure_parent_dir(output_path)
    
    with output_path.open("w", encoding="utf-8") as f:
        for course in tqdm(courses, desc="转换课程", unit="门"):
            # 跳过没有描述且信息不足的课程
            if not course.get("description") and not course.get("code"):
                continue
            
            doc = course_to_rag_document(course)
            if doc["content"].strip():  # 确保内容不为空
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    print(f"[INFO] RAG 文档已保存：{output_path}")


def clean_translated_record(
    record: Dict[str, Any],
    seen_ids: Set[str],
) -> Optional[Dict[str, Any]]:
    dedup_key = record.get("code") or record.get("title") or record.get("url")
    if not dedup_key or dedup_key in seen_ids:
        return None
    content = (record.get("content") or "").strip()
    compact = content_without_ws(content)
    if len(compact) < CONTENT_MIN_LEN:
        return None
    if is_ascii_heavy(compact):
        return None
    if is_digit_heavy(compact):
        return None
    if looks_like_nav_page(content):
        return None
    cleaned_record = dict(record)
    cleaned_record["content"] = content
    cleaned_record["word_count"] = len(
        re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', content)
    )
    seen_ids.add(dedup_key)
    return cleaned_record


def translate_and_clean_courses(
    courses: List[Dict[str, Any]],
    translator_client: OpenAI,
    local_model: str,
    cleaned_output: Path,
) -> List[Dict[str, Any]]:
    print(f"[INFO] 使用本地 vLLM 翻译课程并生成 cleaned 数据：{cleaned_output}")
    ensure_parent_dir(cleaned_output)
    cleaned_records: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    kept = 0
    with cleaned_output.open("w", encoding="utf-8") as fp:
        for course in tqdm(courses, desc="翻译课程", unit="门"):
            translated = translate_course_record(course, translator_client, local_model)
            translated["content"] = build_course_content(translated)
            cleaned = clean_translated_record(translated, seen_ids)
            if not cleaned:
                continue
            fp.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            fp.flush()
            cleaned_records.append(cleaned)
            kept += 1
    print(f"[INFO] 清洗后课程数：{kept} / {len(courses)}")
    return cleaned_records


def main() -> None:
    args = parse_args()
    
    # 加载课程数据
    courses = load_courses(args.input)
    
    # 初始化本地 vLLM client（翻译）
    print(f"[INFO] 使用本地 vLLM 翻译模型：{LOCAL_MODEL}")
    try:
        import requests
        health_url = LOCAL_BASE_URL.replace("/v1", "/health")
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            print(f"[INFO] 本地 vLLM 服务运行正常")
        else:
            print(f"[WARN] 本地 vLLM 服务可能异常（状态码：{response.status_code}）")
    except ImportError:
        print("[INFO] 跳过本地 vLLM 健康检查（requests 未安装）")
    except Exception as exc:
        print(f"[ERROR] 无法连接到本地 vLLM 服务：{exc}")
        print("请先启动服务，例如：")
        print("  python -m vllm.entrypoints.openai.api_server \\")
        print(f"    --model {LOCAL_MODEL} \\")
        print("    --port 8000")
        sys.exit(1)
    translator_client = OpenAI(api_key="empty", base_url=LOCAL_BASE_URL)
    
    input_filename = args.input.stem
    cleaned_output = Path("data/cleaned") / f"{input_filename}.jsonl"
    cleaned_courses = translate_and_clean_courses(
        courses,
        translator_client,
        LOCAL_MODEL,
        cleaned_output,
    )
    
    # 生成 RAG 文档（基于 cleaned 数据）
    rag_output = Path("data/rag") / f"{input_filename}.jsonl"
    save_rag_documents(cleaned_courses, rag_output)
    
    print("[INFO] cleaned 与 RAG 数据已生成")


if __name__ == "__main__":
    main()

