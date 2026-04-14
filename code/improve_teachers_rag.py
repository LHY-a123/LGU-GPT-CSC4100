#!/usr/bin/env python3
"""
专门针对 data/cuhksz_teachers.jsonl 的 RAG 优化脚本：

目标：
- 从教师 / 学者页面中，生成更适合检索和问答的 title 与 content
- 保留较长上下文（包含官网页面 + 个人网站抓取内容）
- 跳过无效的导航页、列表页等

约定：
- 标题格式：中文名 英文名 研究方向1；研究方向2
  - 英文名或研究方向缺失时可省略对应部分，但仍保持整体简洁
  - 如：张三 John Zhang 人工智能；机器学习
- content：
  - 正常整理为可读性较好的中文说明文本
  - 论文标题、出版物列表等需要**保持英文原文**（不要翻译成中文）
  - 可以删除导航、版权等垃圾文本，去重、规整段落

依赖：
- 本地 vLLM OpenAI 接口（默认 http://localhost:8000/v1）
- 模型示例：Qwen/Qwen2.5-14B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import requests
except ImportError:
    requests = None

from openai import OpenAI
from tqdm import tqdm


# ============================================================================
# 参数解析
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="优化 cuhksz_teachers.jsonl 的 RAG 标题与内容（教师专用）"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="输入 JSONL 文件路径（例如：data/cuhksz_teachers.jsonl）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 RAG JSONL 文件路径（默认：data/rag/同名文件）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="模型名称（默认：Qwen/Qwen2.5-14B-Instruct）",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API base URL（默认：http://localhost:8000/v1）",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="API 调用最大重试次数（默认：3）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="并行处理线程数（默认：8）",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="采样记录数量（用于测试，默认：全部）",
    )
    return parser.parse_args()


# ============================================================================
# 文件 I/O
# ============================================================================


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件"""
    records: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] {file_path}:{line_no} JSON解析失败：{exc}", file=sys.stderr)
    return records


def derive_output_path(input_path: Path) -> Path:
    """根据输入文件路径推导输出路径（放到 data/rag/ 下）"""
    filename = input_path.name
    return Path("data/rag") / filename


# ============================================================================
# 内容截断 / 质量判断 / 清理
# ============================================================================


def truncate_content(content: str, max_length: int = 4000) -> str:
    """截断内容，保留前 max_length 字符"""
    if len(content) <= max_length:
        return content
    return content[:max_length] + "……（内容已截断）"


def is_low_quality_content(content: str) -> bool:
    """
    判断是否为明显无效内容（导航、纯菜单、异常乱码等）
    """
    content_stripped = content.strip()
    if not content_stripped:
        return True

    total_length = len(content_stripped)

    # 太短基本没信息
    if total_length < 80:
        return True

    # 明显乱码（Unicode 替换符过多）
    fffd_count = content.count("\ufffd")
    total_chars = len(content)
    if total_chars > 0 and fffd_count / total_chars > 0.03:
        return True

    # 导航 / 菜单占比极高
    nav_keywords = [
        "登录",
        "注册",
        "Sub Main",
        "My Portal",
        "校园生活",
        "招生",
        "就业指导",
        "校园地图",
        "版权所有",
        "粤ICP备",
    ]
    nav_hits = sum(1 for kw in nav_keywords if kw in content)
    if nav_hits >= 6 and total_length < 800:
        return True

    # 行级重复率极高也视为低质量
    lines = [l.strip() for l in content_stripped.split("\n") if l.strip()]
    if len(lines) >= 10:
        unique = len(set(lines))
        if unique / len(lines) < 0.25:
            return True

    return False


def clean_content_basic(content: str) -> str:
    """
    基础清洗：移除导航/版权等垃圾文本、简单去重
    不做任何翻译。
    """
    import re

    cleaned = content.replace("\ufffd", "")

    nav_patterns = [
        r"登录\s*\|\s*注册",
        r"Sub Main",
        r"My Portal",
        r"校园生活",
        r"校园地图",
        r"版权所有.+?粤公网安备",
        r"粤ICP备\d+号",
        r"上一页",
        r"下一页",
        r"末页",
        r"尾页",
        r"分页\s*当前页",
    ]
    for pattern in nav_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)

    # 合并多余空行
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # 行级去重（避免重复菜单/段落）
    lines = [l.strip() for l in cleaned.split("\n")]
    lines = [l for l in lines if l]

    dedup_lines: List[str] = []
    seen = set()
    for line in lines:
        key = line
        if len(key) >= 12:
            if key in seen:
                continue
            seen.add(key)
        dedup_lines.append(line)

    cleaned = "\n".join(dedup_lines)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def improve_content_with_llm(
    content: str,
    client: OpenAI,
    model: str,
) -> str:
    """
    使用 LLM 优化教师页面内容：
    - 不翻译英文论文标题、期刊名等
    - 以中文为主做结构和摘要整理
    """
    # 先做基础清洗
    cleaned = clean_content_basic(content)
    if len(cleaned) < 80:
        return cleaned

    # 只让 LLM 处理前一部分，后半部分直接丢弃，避免长篇论文列表全部保留
    content_to_process = cleaned[:3000] if len(cleaned) > 3000 else cleaned

    system_prompt = (
        "你是一名负责整理高校教师简介的编辑助手。\n"
        "现在有一段混合了中文和英文的教师/学者页面内容，请你：\n"
        "1. 删除网页导航、菜单、版权等与教师个人无关的内容；\n"
        "2. 删除明显重复的段落；\n"
        "3. 用中文整理出一段结构清晰、可读性好的介绍文本，重点包括：教育背景、工作经历、研究方向、代表性成果等；\n"
        "4. 注意：论文题目、期刊名称、书籍名称等**一律保持英文原文**，不要翻译成中文，也不要随意改写；\n"
        "5. 如果有很长的论文/出版物列表，只需保留少量具有代表性的条目，或用一两句中文进行概括，不要完整逐条罗列所有论文；\n"
        "6. 英文简介段落如果本身是论文/著作标题列表，也保持英文原文，不必翻译；\n"
        "7. 不要杜撰信息，只能基于提供的文本进行整理；\n"
        "8. 输出只包含整理后的正文内容，不要添加任何说明性的前后缀。"
    )

    user_prompt = f"请整理下面这段教师/学者相关的页面内容：\n\n{content_to_process}\n\n整理后的内容："

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            timeout=15,
        )
        improved = (resp.choices[0].message.content or "").strip()
        if not improved:
            return cleaned

        return improved
    except Exception:
        return cleaned


# ============================================================================
# 标题生成（教师专用）
# ============================================================================


def generate_teacher_title(
    content: str,
    original_title: str,
    teacher_info: Dict[str, Any] | None,
    client: OpenAI,
    model: str,
) -> Tuple[str, bool]:
    """
    生成教师专用标题：
    - 标题格式：中文名 英文名 研究方向1；研究方向2
    - 如果无法可靠识别教师信息或内容无效，则返回 ("SKIP", False)
    """
    # 如果整体内容很差，直接跳过
    if is_low_quality_content(content):
        return ("SKIP", False)

    # 从 teacher_info 中拿到一些 hint
    zh_name_hint = ""
    en_name_hint = ""
    research_hint = ""
    if teacher_info:
        zh_name_hint = str(teacher_info.get("name") or "").strip()
        research_hint = str(teacher_info.get("research_areas") or "").strip()

    system_prompt = (
        "你是一名高校人事信息整理专家，负责为教师/学者生成统一格式的标题。\n"
        "根据给定的教师页面内容，提取：\n"
        "1. 中文姓名（必填，如果无法确定，返回 SKIP）；\n"
        "2. 英文姓名（如果有，则提取；没有可以省略）；\n"
        "3. 1-2 个最核心的研究方向或学科领域，使用中文简洁表达；\n"
        "并按以下格式输出标题：\n"
        "中文名 英文名 研究方向1；研究方向2\n"
        "注意：\n"
        "- 英文名紧跟在中文名后面，中间用空格分隔；\n"
        "- 如果没有英文名，就只写中文名和研究方向；\n"
        "- 研究方向用中文，1到2个，用中文分号 '；' 分隔；\n"
        "- 标题中不要包含学校名称、学院名称、职务头衔等冗余信息；\n"
        "- 如果无法确定这是某一位具体教师/学者（例如只是导航页、列表页），请只输出 'SKIP'。"
    )

    hints_part = ""
    if zh_name_hint or research_hint:
        hints_part = (
            f"\n\n可用的辅助信息（可能来自结构化字段，可用于校验）：\n"
            f"- 可能的姓名: {zh_name_hint}\n"
            f"- 可能的研究方向: {research_hint}\n"
        )

    truncated = truncate_content(content, 3000)
    user_prompt = (
        f"原始页面标题：{original_title}\n"
        f"{hints_part}\n"
        f"下面是该页面的主要内容（可能包含中文和英文）：\n\n"
        f"{truncated}\n\n"
        "请根据以上信息，生成一个符合要求的标题。如果无法确定具体教师，请输出 'SKIP'。"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            timeout=5,
        )
        title = (resp.choices[0].message.content or "").strip()
        title = title.strip('"').strip("'").strip()

        if not title:
            return ("SKIP", False)
        if title.upper() == "SKIP" or title.startswith("SKIP"):
            return ("SKIP", False)
        if len(title) < 2:
            return ("SKIP", False)

        # 简单限制长度
        if len(title) > 80:
            title = title[:80] + "..."

        return (title, True)
    except Exception:
        return ("SKIP", False)


# ============================================================================
# 单条记录处理
# ============================================================================


def process_single_record(
    record: Dict[str, Any],
    client: OpenAI,
    model: str,
    max_retries: int,
    output_path: Path,
    file_lock: threading.Lock,
) -> bool:
    """
    处理单条教师记录：优化内容 + 生成标题 + 写入 RAG 格式
    """
    content = record.get("content", "") or ""
    original_title = record.get("title", "") or ""
    url = record.get("url", "") or ""
    teacher_info = record.get("teacher_info") or {}

    # teacher_info.name 完全为空，且 original_title 明显是导航类时，直接跳过
    name_field = str(teacher_info.get("name") or "").strip()
    if not name_field and ("最新图片" in original_title or "入学申请" in original_title):
        return False

    if not content or len(content.strip()) < 50:
        return False
    if is_low_quality_content(content):
        return False

    # 多次重试 LLM 生成，防止偶发失败
    improved_content = None
    for _ in range(max_retries):
        improved_content = improve_content_with_llm(content, client, model)
        if improved_content and len(improved_content.strip()) >= 80:
            break
    if not improved_content or len(improved_content.strip()) < 80:
        return False

    # 再次用 LLM 生成教师专用标题
    title = "SKIP"
    is_valid = False
    for _ in range(max_retries):
        title, is_valid = generate_teacher_title(
            improved_content,
            original_title,
            teacher_info,
            client,
            model,
        )
        if is_valid:
            break
    if not is_valid or title == "SKIP":
        return False

    word_count = len(improved_content.split())

    rag_record = {
        "title": title,
        "content": improved_content,
        "word_count": word_count,
        "url": url,
    }

    with file_lock:
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rag_record, ensure_ascii=False) + "\n")
            f.flush()

    return True


# ============================================================================
# 批量处理
# ============================================================================


def improve_teachers_rag(
    records: List[Dict[str, Any]],
    output_path: Path,
    client: OpenAI,
    model: str,
    max_retries: int = 2,
    max_workers: int = 32,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 过滤掉明显不含教师信息的记录（简单预筛）
    filtered: List[Dict[str, Any]] = []
    for r in records:
        c = (r.get("content") or "").strip()
        if not c:
            continue
        if len(c) < 80:
            continue
        # 教师页面通常有“教授/博士/讲席教授”等关键词
        if not any(
            kw in c
            for kw in ["教授", "讲席", "博士", "Distinguished Professor", "Nobel"]
        ):
            # 但如果 teacher_info.name 非空，也保留
            ti = r.get("teacher_info") or {}
            if not str(ti.get("name") or "").strip():
                continue
        filtered.append(r)

    print(f"[INFO] 原始记录：{len(records)} 条，预筛后：{len(filtered)} 条")

    file_lock = threading.Lock()
    output_path.touch()

    processed = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_record,
                r,
                client,
                model,
                max_retries,
                output_path,
                file_lock,
            ): r
            for r in filtered
        }

        with tqdm(total=len(futures), desc="处理教师页面", unit="条") as pbar:
            completed = 0
            for future in as_completed(futures):
                try:
                    ok = future.result(timeout=30)
                    if ok:
                        processed += 1
                    else:
                        skipped += 1
                except Exception as e:
                    skipped += 1
                    if completed < 5:
                        print(f"[DEBUG] 处理失败: {e}", file=sys.stderr)
                completed += 1
                pbar.update(1)

                if completed % 10 == 0:
                    size = output_path.stat().st_size if output_path.exists() else 0
                    print(
                        f"[INFO] 已完成 {completed}/{len(futures)} 条，"
                        f"成功 {processed} 条，文件大小 {size} 字节",
                        file=sys.stderr,
                    )

    final_size = output_path.stat().st_size if output_path.exists() else 0
    final_lines = (
        sum(1 for _ in output_path.open("r", encoding="utf-8"))
        if output_path.exists() and final_size > 0
        else 0
    )

    print(f"[INFO] 最终写入 {processed} 条，跳过 {skipped} 条")
    print(f"[INFO] 输出文件：{output_path}，大小：{final_size} 字节，行数：{final_lines}")


# ============================================================================
# vLLM 健康检查 & 主函数
# ============================================================================


def check_vllm_health(base_url: str) -> bool:
    """检查 vLLM 服务健康状态"""
    if not requests:
        return True
    try:
        health_url = base_url.replace("/v1", "/health")
        resp = requests.get(health_url, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def main() -> None:
    args = parse_args()

    print(f"[INFO] 读取教师数据：{args.input}")
    records = read_jsonl(args.input)
    print(f"[INFO] 共读取 {len(records)} 条记录")

    if args.sample_size and args.sample_size < len(records):
        import random

        random.seed(42)
        records = random.sample(records, args.sample_size)
        print(f"[INFO] 采样 {args.sample_size} 条记录用于处理")

    base_url = args.base_url
    api_key = "empty"

    print(f"[INFO] 使用本地 vLLM API：{base_url}")
    print(f"[INFO] 使用模型：{args.model}")

    if check_vllm_health(base_url):
        print("[INFO] vLLM 服务运行正常")
    else:
        print("[WARN] vLLM 服务可能未正常运行", file=sys.stderr)
        print(
            "      请确保已启动，例如：\n"
            f"      python -m vllm.entrypoints.openai.api_server \\\n"
            f"        --model {args.model} \\\n"
            f"        --port 8000",
            file=sys.stderr,
        )

    client = OpenAI(api_key=api_key, base_url=base_url)

    if args.output:
        output_path = args.output
    else:
        output_path = derive_output_path(args.input)

    if output_path.exists():
        print(f"[INFO] 输出文件已存在，将覆盖：{output_path}")
        output_path.unlink()

    improve_teachers_rag(
        records,
        output_path,
        client,
        args.model,
        max_retries=args.max_retries,
        max_workers=args.max_workers,
    )

    print("[INFO] 教师 RAG 优化完成")


if __name__ == "__main__":
    main()


