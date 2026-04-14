#!/usr/bin/env python3
"""
改进 RAG 数据的 title，使其真正反映文本主题

功能：
1. 读取 cleaned 数据（JSONL 格式）
2. 使用本地 vLLM API 根据 content 生成更好的 title
3. 过滤低质量内容（导航菜单、列表页面等）
4. 保存到 data/rag 目录（流式输出）

要求：
- 本地运行 vLLM 服务（默认：http://localhost:8000/v1）
- 使用 Qwen/Qwen2.5-14B-Instruct 模型（或通过 --model 指定）
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
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
# 配置和参数解析
# ============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="改进 RAG 数据的 title，使其真正反映文本主题"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="输入的 cleaned JSONL 文件路径（例如：data/cleaned/career_cuhk_edu_cn.jsonl）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出的 RAG JSONL 文件路径（默认：data/rag/xxx.jsonl）",
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
        default=3,
        help="API 调用最大重试次数（默认：3）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="并行处理线程数（默认：16）",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="采样记录数量（用于测试，默认：全部）",
    )
    parser.add_argument(
        "--no-improve-content",
        action="store_true",
        help="不改进内容，只优化标题（默认：同时改进内容和标题）",
    )
    return parser.parse_args()


# ============================================================================
# 文件 I/O
# ============================================================================

def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件"""
    records = []
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
    """根据输入文件路径推导输出路径（从 cleaned 到 rag）"""
    filename = input_path.name
    return Path("data/rag") / filename


# ============================================================================
# 内容处理和过滤
# ============================================================================

def truncate_content(content: str, max_length: int = 3000) -> str:
    """截断内容，保留前 max_length 字符"""
    if len(content) <= max_length:
        return content
    return content[:max_length] + "……（内容已截断）"


def is_low_quality_content(content: str) -> bool:
    """
    快速检查内容是否为低质量（导航菜单、列表页面、乱码、重复内容等）
    
    Returns:
        True 如果是低质量内容，应该跳过
    """
    content_stripped = content.strip()
    total_length = len(content_stripped)
    
    # 1. 内容太短（放宽到100字符，因为有些有价值的内容可能较短）
    if total_length < 100:
        return True
    
    # 2. 检查乱码（只检查明显的乱码字符和模式）
    # \ufffd 是 Unicode 替换字符，表示解码错误
    fffd_count = content.count('\ufffd')
    total_chars = len(content)
    # 只有当 \ufffd 占比超过3%时才认为是低质量
    if total_chars > 0 and fffd_count / total_chars > 0.03:
        return True
    
    # 检查是否有明显的乱码模式（如 " actor" 这种，注意：空字符串检查已移除，因为 '' in content 总是为 True）
    if ' actor' in content.lower():
        return True
    
    # 3. 检查是否主要是列表格式（列表项过多）
    # 注意：有些有价值的内容也是列表格式（如专业列表），不要太严格
    list_markers = content.count('-') + content.count('*') + content.count('•')
    lines = [l.strip() for l in content_stripped.split('\n') if l.strip()]
    if len(lines) > 0:
        list_ratio = list_markers / len(lines)
        # 只有当列表项占比非常高（>80%）且内容较短时，才认为是低质量
        if list_ratio > 0.8 and total_length < 500:
            return True
    
    # 4. 检查内容重复度（更严格的条件）
    if len(lines) > 10:  # 只有行数较多时才检查重复度
        unique_lines = len(set(lines))
        if unique_lines / len(lines) < 0.3:  # 重复度超过70%才过滤
            return True
    
    # 5. 检查导航菜单关键词
    nav_keywords = ["登录|注册", "返回主站", "MENU", "首页>", "返回列表", "下一篇", "相关推荐", "CN|EN"]
    nav_count = sum(1 for kw in nav_keywords if kw in content)
    
    # 6. 检查导航文本占比
    nav_text_patterns = [
        "登录|注册CN|EN",
        "返回主站",
        "MENU",
        "返回",
        "招生网科研处",
        "香港中文大学（深圳）©版权所有"
    ]
    
    nav_text_length = sum(len(pattern) * content.count(pattern) for pattern in nav_text_patterns)
    
    # 如果导航关键词过多或导航文本占比超过40%，跳过
    if nav_count >= 5 or (nav_text_length / total_length > 0.4 if total_length > 0 else True):
        return True
    
    # 7. 检查是否主要是导航菜单结构（包含多个导航分类）
    # 只有当导航分类很多且内容很短时，才认为是纯导航页面
    nav_categories = ['通知公告', '招生政策', '网上申请', '学费及住宿费', '常见问题', '招生活动']
    nav_category_count = sum(1 for cat in nav_categories if cat in content)
    # 放宽条件：只有包含所有6个导航分类且内容很短时才过滤
    if nav_category_count >= 6 and total_length < 400:
        return True
    
    return False


def clean_content(content: str) -> str:
    """
    清理内容：去除垃圾、重复、导航文本、乱码等
    
    Args:
        content: 原始内容
    
    Returns:
        清理后的内容
    """
    import re
    
    # 移除乱码字符
    cleaned = content.replace('\ufffd', '').replace('', '')
    
    # 移除常见的导航和垃圾文本
    nav_patterns = [
        r"登录\s*\|\s*注册\s*CN\s*\|\s*EN",
        r"返回主站",
        r"MENU",
        r"首页\s*>",
        r"返回列表",
        r"下一篇",
        r"相关推荐",
        r"招生网科研处",
        r"香港中文大学（深圳）©\s*版权所有",
        r"返回\s*$",
        r"分页\s*当前页",
        r"页面\s*\d+",
        r"上一页|下一页|末页|尾页",
    ]
    
    for pattern in nav_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    # 移除重复的换行（超过2个连续换行）
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    
    # 移除行首行尾的空白
    lines = [line.strip() for line in cleaned.split("\n")]
    lines = [line for line in lines if line]  # 移除空行
    
    # 移除重复的连续行（如果连续3行相同，只保留1行）
    deduplicated_lines = []
    prev_line = None
    prev_count = 0
    for line in lines:
        if line == prev_line:
            prev_count += 1
            if prev_count >= 2:  # 如果连续出现3次，跳过
                continue
        else:
            prev_count = 0
        deduplicated_lines.append(line)
        prev_line = line
    
    # 移除重复的段落（如果整个段落重复出现）
    final_lines = []
    seen_paragraphs = set()
    for line in deduplicated_lines:
        # 跳过太短的重复行（可能是导航）
        if len(line) < 10:
            final_lines.append(line)
            continue
        # 检查是否是完全重复的段落
        if line not in seen_paragraphs:
            seen_paragraphs.add(line)
            final_lines.append(line)
    
    cleaned = "\n".join(final_lines)
    
    # 移除多余空格
    cleaned = re.sub(r" {2,}", " ", cleaned)
    
    return cleaned.strip()


def improve_content_with_llm(
    content: str,
    client: OpenAI,
    model: str,
) -> str:
    """
    使用 LLM 改进内容质量（去除噪声、优化结构）
    
    Args:
        content: 原始内容
        client: OpenAI client
        model: 模型名称
    
    Returns:
        改进后的内容，如果失败则返回原始内容
    """
    # 先进行基础清理
    cleaned = clean_content(content)
    
    # 如果内容太短，不需要 LLM 处理
    if len(cleaned) < 50:
        return cleaned
    
    # 如果内容太长，只处理前 2000 字符
    is_truncated = len(cleaned) > 2000
    content_to_process = cleaned[:2000] if is_truncated else cleaned
    
    system_prompt = (
        "你是一名专业的内容优化专家。"
        "请优化以下文档内容，使其更加清晰、结构化。"
        "\n要求："
        "1. 去除网页噪声和无关信息（如导航菜单、版权信息等）"
        "2. 去除重复内容"
        "3. 保持核心信息完整"
        "4. 优化段落结构，使其更易读"
        "5. 如果内容已经是高质量的，可以保持原样或稍作优化"
        "\n输出格式："
        "只输出优化后的内容，不要添加任何解释或说明"
    )
    
    user_prompt = f"""请优化以下文档内容：

{content_to_process}
{"（内容已截断，仅优化前部分）" if is_truncated else ""}

优化后的内容："""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            timeout=10,  # 10秒超时
        )
        improved = response.choices[0].message.content.strip()
        
        # 如果原内容被截断，拼接剩余部分
        if is_truncated:
            return improved + "\n\n" + cleaned[2000:]
        return improved
    except Exception:
        # 10秒没输出或出错，返回基础清理后的内容
        return cleaned


# ============================================================================
# LLM 标题生成
# ============================================================================

def generate_better_title(
    content: str,
    original_title: str,
    client: OpenAI,
    model: str,
    max_retries: int = 3,
) -> Tuple[str, bool]:
    """
    使用 LLM 生成更好的 title，并判断内容质量
    
    Args:
        content: 文档内容
        original_title: 原始 title
        client: OpenAI client
        model: 模型名称
        max_retries: 最大重试次数
    
    Returns:
        (改进后的 title, 是否适合 RAG)
        如果不适合 RAG，title 为 "SKIP"，is_valid 为 False
    """
    system_prompt = (
        "你是一名专业的文档标题优化和质量评估专家。"
        "请根据文档内容，生成一个简洁、准确、能反映文档核心主题的标题。"
        "同时评估文档内容是否适合用于 RAG 系统。"
        "\n标题要求："
        "1. 简洁明了，5-30字最佳"
        "2. 准确反映文档的核心内容和主题，包含关键信息（如公司名、职位、年份、地区等）"
        "3. 对于招聘信息，应包含：公司名+职位+年份（如：中信期货2022年管理培训生招聘）"
        "4. 对于活动通知，应包含：活动名称+时间/地点（如：柔宇科技2025年11月校园行活动）"
        "5. 对于政策文件，应包含：政策名称+年份+地区（如：广东省2026年公务员考试公告）"
        "6. 标题中避免使用'首页'、'通知公告'、'招聘信息'等通用词汇"
        "7. 不要包含'香港中文大学(深圳)'名称，以便提升RAG的准确性"
        "\n质量评估："
        "如果文档内容主要是导航菜单、登录页面、列表页面（无具体内容）、或内容过短（少于50字有效内容），"
        "请输出 'SKIP' 表示不适合加入 RAG 系统。"
        "\n输出格式："
        "只输出标题文本，如果内容不适合 RAG，输出 'SKIP'"
    )
    
    truncated_content = truncate_content(content, max_length=3000)
    
    # 快速检查：如果内容主要是导航菜单或过短，直接跳过
    if is_low_quality_content(content):
        return ("SKIP", False)
    
    user_prompt = f"""任务目的：为 香港中文大学(深圳) 的 RAG 系统优化文档标题，使标题更准确地反映文档核心内容，便于检索和理解, 标题中不要包含'香港中文大学(深圳)'名称，以便提升RAG的准确性。

原始标题：{original_title}

文档内容：
{truncated_content}

请根据上述规则生成一个更好的标题。如果内容不适合 RAG（如导航菜单、列表页面、内容过短），请输出 'SKIP'。"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            timeout=10,  # 10秒超时
        )
        new_title = response.choices[0].message.content.strip()
        
        # 清理可能的引号或多余格式
        new_title = new_title.strip('"\'')
        new_title = new_title.strip()
        
        # 检查是否标记为跳过
        if new_title.upper() == "SKIP" or new_title.startswith("SKIP"):
            return ("SKIP", False)
        
        # 如果标题太长，截断
        if len(new_title) > 100:
            new_title = new_title[:100] + "..."
        
        return (new_title, True)
    except Exception:
        # 10秒没输出或出错，直接跳过
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
    improve_content: bool = True,
) -> bool:
    """
    处理单条记录，生成改进的标题和内容，并流式输出到文件
    
    Args:
        record: 记录字典（cleaned 格式）
        client: OpenAI client
        model: 模型名称
        max_retries: 最大重试次数
        output_path: 输出文件路径
        file_lock: 文件写入锁
        improve_content: 是否改进内容（默认：True）
    
    Returns:
        True 如果成功处理并输出，False 如果跳过
    """
    content = record.get("content", "")
    original_title = record.get("title", "")
    url = record.get("url", "")
    
    # 跳过内容为空或过短的记录
    if not content or len(content.strip()) < 20:
        return False
    
    if not original_title:
        return False
    
    # 改进内容（去除垃圾、重复等）
    if improve_content:
        improved_content = improve_content_with_llm(content, client, model)
        # 如果改进后内容太短，可能被过度清理，使用原始内容
        if len(improved_content.strip()) < len(content.strip()) * 0.3:
            improved_content = clean_content(content)  # 只做基础清理
    else:
        improved_content = clean_content(content)  # 只做基础清理
    
    # 如果改进后内容为空或过短，跳过
    if not improved_content or len(improved_content.strip()) < 20:
        return False
    
    # 生成更好的标题（使用改进后的内容）
    new_title, is_valid = generate_better_title(
        improved_content,
        original_title,
        client,
        model,
        max_retries,
    )
    
    # 如果不适合 RAG，跳过
    if not is_valid or new_title == "SKIP":
        return False
    
    # 计算改进后的词数
    word_count = len(improved_content.split())
    
    # 构建 RAG 格式输出
    rag_record = {
        "title": new_title,
        "content": improved_content,
        "word_count": word_count,
        "url": url,
        "depth": record.get("depth", 0),
        "crawl_time": record.get("crawl_time", ""),
        "metadata": {
            "source_title": original_title
        }
    }
    
    # 流式输出：使用锁保护文件写入
    with file_lock:
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rag_record, ensure_ascii=False) + "\n")
            f.flush()  # 确保立即写入磁盘
    
    return True


# ============================================================================
# 批量处理
# ============================================================================

def improve_rag_titles(
    records: List[Dict[str, Any]],
    output_path: Path,
    client: OpenAI,
    model: str,
    max_retries: int = 3,
    max_workers: int = 16,
    improve_content: bool = True,
) -> None:
    """
    从 cleaned 数据改进标题和内容并输出到 RAG 格式（流式处理）
    
    Args:
        records: 记录列表（cleaned 格式）
        output_path: 输出路径（RAG 格式）
        client: OpenAI client
        model: 模型名称
        max_retries: 最大重试次数
        max_workers: 并行处理线程数
        improve_content: 是否改进内容（默认：True）
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 过滤掉内容过短的记录
    valid_records = []
    for record in records:
        content = record.get("content", "")
        if content and len(content.strip()) >= 20:
            valid_records.append(record)
    
    print(f"[INFO] 有效记录：{len(valid_records)} 条，总记录：{len(records)} 条")
    
    # 使用锁保护文件写入
    file_lock = threading.Lock()
    
    # 并行处理（使用锁保护文件写入）
    processed_count = 0
    skipped_count = 0
    
    # 确保输出文件存在（即使为空）
    output_path.touch()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_record = {
            executor.submit(
                process_single_record,
                record,
                client,
                model,
                max_retries,
                output_path,
                file_lock,
                improve_content,
            ): record
            for record in valid_records
        }
        
        # 收集结果
        with tqdm(total=len(valid_records), desc="处理进度", unit="条") as pbar:
            completed_count = 0
            for future in as_completed(future_to_record):
                # 10秒没输出直接跳过
                try:
                    success = future.result(timeout=10)
                    if success:
                        processed_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    # 超时或出错，直接跳过
                    skipped_count += 1
                    if completed_count < 5:  # 前5条错误时打印
                        print(f"[DEBUG] 处理失败: {e}", file=sys.stderr)
                
                completed_count += 1
                pbar.update(1)
    
                # 每处理10条记录，输出一次进度和文件状态
                if completed_count % 10 == 0:
                    file_size = output_path.stat().st_size if output_path.exists() else 0
                    print(f"[INFO] 已完成 {completed_count}/{len(valid_records)} 条，已写入 {processed_count} 条，文件大小: {file_size} 字节", file=sys.stderr)
    
    # 最终检查
    final_size = output_path.stat().st_size if output_path.exists() else 0
    final_lines = sum(1 for _ in output_path.open('r', encoding='utf-8')) if output_path.exists() and final_size > 0 else 0
    
    print(f"[INFO] 已处理 {processed_count} 条记录，跳过 {skipped_count} 条")
    print(f"[INFO] 输出文件：{output_path}，大小：{final_size} 字节，行数：{final_lines}")
    
    if processed_count == 0:
        print(f"[WARN] 没有记录被成功处理！可能所有记录都被过滤了。", file=sys.stderr)


# ============================================================================
# 主函数
# ============================================================================

def check_vllm_health(base_url: str) -> bool:
    """检查 vLLM 服务健康状态"""
    if not requests:
        return True  # 如果 requests 未安装，跳过检查
    
    try:
        health_url = base_url.replace("/v1", "/health")
        response = requests.get(health_url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def main() -> None:
    """主函数"""
    args = parse_args()
    
    # 读取输入文件
    print(f"[INFO] 读取文件：{args.input}")
    records = read_jsonl(args.input)
    print(f"[INFO] 共读取 {len(records)} 条记录")
    
    # 采样（如果指定）
    if args.sample_size and args.sample_size < len(records):
        import random
        random.seed(42)
        records = random.sample(records, args.sample_size)
        print(f"[INFO] 采样 {args.sample_size} 条记录用于处理")
    
    # 初始化本地 vLLM API client
    base_url = args.base_url
    api_key = "empty"  # 本地 vLLM 不需要真实的 API key
    
    print(f"[INFO] 使用本地 vLLM API：{base_url}")
    print(f"[INFO] 使用模型：{args.model}")
    
    # 验证服务是否运行
    if check_vllm_health(base_url):
        print(f"[INFO] vLLM 服务运行正常")
    else:
        print(f"[WARN] vLLM 服务可能未正常运行")
        print(f"[INFO] 请确保 vLLM 服务已启动：")
        print(f"      python -m vllm.entrypoints.openai.api_server \\")
        print(f"        --model {args.model} \\")
        print(f"        --port 8000")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # 推导输出路径（从 cleaned 到 rag）
    if args.output:
        output_path = args.output
    else:
        output_path = derive_output_path(args.input)
    
    # 如果输出文件已存在，先删除（重新生成）
    if output_path.exists():
        print(f"[INFO] 输出文件已存在，将覆盖：{output_path}")
        output_path.unlink()
    
    # 改进标题和内容并流式输出
    improve_rag_titles(
        records,
        output_path,
        client,
        args.model,
        max_retries=args.max_retries,
        max_workers=args.max_workers,
        improve_content=not args.no_improve_content,
    )
    
    print("[INFO] 处理完成！")


if __name__ == "__main__":
    main()