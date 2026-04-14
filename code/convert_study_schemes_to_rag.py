#!/usr/bin/env python3
"""
将 study_schemes_processed 目录下的 JSON 文件转换为 RAG 格式
使用 DeepSeek API 整理成 title: content 对
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm


DEFAULT_SYSTEM_PROMPT = (
    "你是一名对香港中文大学（深圳）学习方案（Study Scheme）拥有全面知识的学术顾问。"
    "你的任务是将结构化的学习方案数据整理成清晰、完整的RAG格式文档。"
    "需要包含所有关键信息：专业名称、适用年度、毕业要求、课程列表、分流方向、选课建议等。"
    "内容要准确、完整，便于后续RAG系统检索和使用。"
)

USER_PROMPT_TEMPLATE = """下面是一个学习方案（Study Scheme）的结构化JSON数据。请将其整理成RAG格式的文档，包含所有关键信息。

要求：
1. 标题（title）：使用格式"专业名称_适用年度"，例如"数据科学与大数据技术_2021-22年度"
2. 内容（content）：需要包含以下所有信息：
   - 专业名称（中英文）
   - 适用年度
   - 专业类型（主修/辅修/双主修等）
   - 毕业要求（总学分、学院课程学分、必修课程学分、选修课程学分等）
   - 学院课程列表（课程代码、名称、学分等）
   - 必修课程列表（课程代码、名称、学分等）
   - 选修课程列表（课程代码、名称、学分、所属分流方向等）
   - 分流方向信息（名称、要求、课程列表等）
   - 推荐修课模式（如果有）
   - 重要备注和说明
   - 其他相关信息

3. 内容格式要求：
   - 使用清晰的分段和列表
   - 保持信息的完整性和准确性
   - 使用中文，专业术语可保留英文
   - 课程信息要详细（代码、中英文名称、学分、级别等）
   - 分流方向要明确说明要求和课程

4. 输出格式：JSON格式，包含title和content两个字段

以下是JSON数据：
{json_data}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 study_schemes_processed 目录下的 JSON 文件转换为 RAG 格式"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/study_schemes_processed"),
        help="输入目录，包含JSON文件（默认：data/study_schemes_processed）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/rag/study_schemes.jsonl"),
        help="输出JSONL文件路径（默认：data/rag/study_schemes.jsonl）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="并行处理的进程数（默认：5，对应5个API key）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-reasoner",
        help="DeepSeek模型名称（默认：deepseek-reasoner）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="采样温度（默认：0.3，降低随机性）",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="每个API调用的最大重试次数（默认：3）",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.1,
        help="API调用之间的休眠时间（秒，默认：0.1）",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://api.deepseek.com",
        help="DeepSeek API基础URL（默认：https://api.deepseek.com）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印提示，不调用API",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前N个文件，用于调试",
    )
    return parser.parse_args()


def load_json_file(json_path: Path) -> Dict[str, Any]:
    """加载JSON文件"""
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_messages(json_data: Dict[str, Any], system_prompt: str) -> List[Dict[str, str]]:
    """构建API消息"""
    json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
    user_prompt = USER_PROMPT_TEMPLATE.format(json_data=json_str)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def strip_code_fences(text: str) -> str:
    """移除代码块标记"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip("`").startswith("json"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_model_response(raw_text: str) -> Dict[str, str]:
    """解析模型返回的JSON"""
    cleaned = strip_code_fences(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"无法解析模型返回的JSON：{exc}\n原文：{raw_text}") from exc
    
    if not isinstance(data, dict):
        raise ValueError("模型返回的不是字典格式")
    
    title = data.get("title", "").strip()
    content = data.get("content", "").strip()
    
    if not title or not content:
        raise ValueError("模型返回缺少title或content字段")
    
    return {"title": title, "content": content}


def call_deepseek(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_retries: int,
) -> str:
    """调用DeepSeek API"""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # pylint: disable=broad-except
            wait = attempt * 2
            print(
                f"[WARN] 调用 DeepSeek 失败（第{attempt}次）：{exc}，{wait}s 后重试……",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError("达到最大重试次数，仍无法获取模型响应")


def process_single_json(
    json_path: Path,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_retries: int,
    sleep_seconds: float,
    system_prompt: str,
    dry_run: bool,
) -> tuple[Path, Optional[Dict[str, str]], Optional[str]]:
    """处理单个JSON文件"""
    try:
        json_data = load_json_file(json_path)
        
        if dry_run:
            print(f"[DRY-RUN] 处理文件：{json_path.name}")
            return json_path, None, None
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        messages = build_messages(json_data, system_prompt)
        
        raw_text = call_deepseek(
            client=client,
            messages=messages,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
        )
        
        rag_data = parse_model_response(raw_text)
        time.sleep(sleep_seconds)
        
        return json_path, rag_data, None
        
    except Exception as exc:  # pylint: disable=broad-except
        return json_path, None, str(exc)


def main() -> None:
    args = parse_args()
    
    # API keys
    api_keys = [
        "sk-2db26a5af44b43a7af0b3e4a55cd29e6",
        "sk-8ce77f4caa9b45bfbf741a368be18840",
        "sk-42892705f5a642b18ff533680c380162",
        "sk-ad2fffdb5d194d0b95ecab1e6b37b370",
        "sk-3930f5f79ff1450eb5c6b1ac582bc246"
    ]
    
    # 查找所有JSON文件
    json_files = sorted(args.input_dir.glob("*.json"))
    if not json_files:
        print(f"[ERROR] 在 {args.input_dir} 中未找到JSON文件", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
    
    if args.limit:
        json_files = json_files[:args.limit]
    
    print(f"[INFO] 找到 {len(json_files)} 个JSON文件", flush=True)
    print(f"[INFO] 使用 {args.max_workers} 个进程并行处理", flush=True)
    
    # 确保输出目录存在
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # 为每个文件分配API key（轮询方式）
    file_api_pairs = [
        (json_file, api_keys[i % len(api_keys)])
        for i, json_file in enumerate(json_files)
    ]
    
    results: Dict[Path, Dict[str, str]] = {}
    errors: List[tuple[Path, str]] = []
    
    # 打开输出文件，准备追加写入
    out_fp = None
    if not args.dry_run:
        out_fp = args.output.open("w", encoding="utf-8")
    
    try:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(
                    process_single_json,
                    json_file,
                    api_key,
                    args.base_url,
                    args.model,
                    args.temperature,
                    args.max_retries,
                    args.sleep_seconds,
                    DEFAULT_SYSTEM_PROMPT,
                    args.dry_run,
                ): json_file
                for json_file, api_key in file_api_pairs
            }
            
            # 使用tqdm显示总体进度
            with tqdm(total=len(json_files), desc="处理文件", unit="个") as pbar:
                for future in as_completed(future_to_file):
                    json_file = future_to_file[future]
                    try:
                        processed_file, rag_data, error = future.result()
                        if error:
                            errors.append((processed_file, error))
                            print(f"[ERROR] 处理 {processed_file.name} 时出错：{error}", file=sys.stderr)
                            sys.stderr.flush()
                        elif rag_data:
                            results[processed_file] = rag_data
                            # 立即写入输出文件
                            if out_fp:
                                out_fp.write(json.dumps(rag_data, ensure_ascii=False) + "\n")
                                out_fp.flush()
                        pbar.update(1)
                        pbar.set_postfix({
                            "成功": len(results),
                            "错误": len(errors),
                            "当前": processed_file.name[:30]
                        })
                    except Exception as exc:
                        errors.append((json_file, str(exc)))
                        print(f"[ERROR] 处理 {json_file.name} 时出错：{exc}", file=sys.stderr)
                        sys.stderr.flush()
                        pbar.update(1)
    finally:
        if out_fp:
            out_fp.close()
    
    # 输出统计信息
    print(f"\n[INFO] 处理完成！", flush=True)
    print(f"  成功处理: {len(results)} 个文件", flush=True)
    print(f"  错误数量: {len(errors)}", flush=True)
    if not args.dry_run:
        print(f"  输出文件: {args.output}", flush=True)
    
    if errors:
        print(f"\n[WARN] 以下文件处理失败：", flush=True)
        for json_file, error in errors[:10]:
            print(f"  - {json_file.name}: {error}", flush=True)
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个错误", flush=True)


if __name__ == "__main__":
    main()

