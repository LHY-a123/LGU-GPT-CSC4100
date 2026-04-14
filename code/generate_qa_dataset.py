#!/usr/bin/env python3
"""
Utility script to sample raw CUHK(SZ) corpus JSONL files and call DeepSeek's
OpenAI-compatible API to build a QA fine-tuning dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


DEFAULT_SYSTEM_PROMPT = (
    "你是一名对香港中文大学（深圳）拥有全面知识库的资深学术顾问，擅长提炼网页素材为高质量中文问答。"
    "所有回答必须严格依据提供的内容进行整理与转述，严禁臆造或扩写外部信息。"
    "若材料不足以支撑问题，请明确指出并跳过。"
    "回答需符合官方口径，逻辑分明，必要时使用列表或分段，帮助后续训练出对学校无所不知且有条理的 LLM。"
)

USER_PROMPT_TEMPLATE = """下面是一段关于香港中文大学（深圳）的网页内容。请只依据该段内容生成{qa_count}个高质量问答对，用于训练一款“对港中大（深圳）了如指掌”的本地 LLM。

要求：
1. 问题需信息密度高，优先覆盖学校概况、历史沿革、院系项目、课程特色、招生/申请条件、奖学金或费用、科研/师资亮点、校园设施、联系方式/办事流程等多元主题，可将多个要点融入同一问题。
2. 答案必须完全来自材料，可在忠于原意的前提下重写；如信息不足以回答某类问题，宁可省略；保持真实性的同时兼顾语言的流畅性、逻辑性与可读性。
3. 强调事实准确性：涉及年份、人数、指标、网址等要保留原始表述并注明单位。
4. 若内容适合条列，请用列表提升可读性；否则以短段落呈现，70-200 字最佳。
5. 每个答案需尽量整合多条关键信息（如背景、数据、项目亮点、流程步骤等），必要时可分多段或混合列表/段落，以增强信息密度。
6. 若你判定材料噪声过多、缺乏语序（例如只有栏目名称/关键词列表）或几乎无有效信息或全是网址，可直接输出：{{ "skip": true, "reason": "一句话说明原因" }}。
7. 正常情况下输出 JSON，模板如下：
{{
  "qas": [
    {{
      "question": "string, 20-60字，使用自然语言",
      "answer": "string, 120-260字，使用自然语言，可混合列表与短段落",
      "answer_type": "descriptive|procedural|comparative|eligibility|contact|other",
      "confidence_note": "一句话说明答案依据（可含原文片段）"
    }}
  ]
}}

以下是供参考的网页内容（已截断至{content_length}字符）：
---
标题：{title}
URL：{url}
正文：
{content}
---"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample raw JSONL files and build QA dataset with DeepSeek."
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        required=False,
        help="One or more JSONL files under data/raw/ or data/cleaned/. 若不提供，将自动扫描 data/cleaned/ 目录。",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of samples to draw from each file (default: 100).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="并行处理的进程数（默认：5，对应5个API key）",
    )
    parser.add_argument(
        "--top-ratio",
        type=float,
        default=2.0,
        help="从排序后的前 top_ratio * sample_size 条高信息密度记录中采样 (default: 2.0).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination JSONL path. 若不提供，将针对每个输入文件自动写入 data/qa/<文件名>.jsonl。",
    )
    parser.add_argument(
        "--qas-per-record",
        type=int,
        default=1,
        help="Number of QA pairs per sampled record (default: 1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-reasoner",
        help="DeepSeek model name (default: deepseek-chat).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for the model (default: 1.0).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per API call (default: 3).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.5,
        help="Seconds to sleep between API calls to avoid rate limits.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the API, only print sampled prompts.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        help="Override DeepSeek base URL (default env DEEPSEEK_BASE_URL or https://api.deepseek.com).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Custom system prompt text.",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def derive_output_path(input_path: Path) -> Path:
    """根据输入文件路径推导输出路径"""
    qa_root = Path("data/qa")
    return qa_root / input_path.name


def read_jsonl(file_path: Path) -> Iterable[Dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] {file_path}:{line_no} JSON解析失败：{exc}", file=sys.stderr)


def reservoir_sample(
    iterable: Iterable[Dict[str, Any]],
    k: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    sample: List[Dict[str, Any]] = []
    for idx, item in enumerate(iterable, start=1):
        if len(sample) < k:
            sample.append(item)
            continue
        j = rng.randint(1, idx)
        if j <= k:
            sample[j - 1] = item
    return sample


def compute_info_density(record: Dict[str, Any]) -> int:
    """
    计算记录的信息密度分数（用于排序）
    使用去重后的词数，更能反映真实的信息密度
    """
    content = (record.get("content") or "").strip()
    if not content:
        return 0
    
    # 计算去重后的词数
    # 对于中文，按常见分隔符（空格、标点、换行等）分割
    # 移除标点和空白字符，保留中文字符、英文字母和数字
    words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', content)
    unique_words = len(set(words))
    
    return unique_words


def smart_sample(
    records: List[Dict[str, Any]],
    k: int,
    rng: random.Random,
    top_ratio: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    智能采样：优先选择信息密度高的记录
    
    Args:
        records: 所有记录列表
        k: 需要采样的数量
        rng: 随机数生成器
        top_ratio: 从排序后的前 top_ratio * k 条记录中采样（默认2倍）
    """
    if not records:
        return []
    
    # 按信息密度降序排序
    sorted_records = sorted(
        records,
        key=compute_info_density,
        reverse=True,
    )
    
    # 从前 N 条高密度记录中采样（避免只选最长的，保持一定多样性）
    # 优先使用 k * top_ratio，但不超过总数的 30%，且至少是 k
    top_n = min(
        max(k, int(k * top_ratio)),  # 至少 k 条，优先 k * top_ratio
        max(k, int(len(sorted_records) * 0.3)),  # 不超过总数的 30%
    )
    top_candidates = sorted_records[:top_n]
    
    # 如果候选数量 <= 需要的数量，直接返回
    if len(top_candidates) <= k:
        return top_candidates
    
    # 从候选集中使用 reservoir sample 随机选择 k 条
    return reservoir_sample(top_candidates, k, rng)


def truncate_text(text: str, limit: int = 2800) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "……（内容已截断）"


def build_messages(
    record: Dict[str, Any],
    qa_count: int,
    system_prompt: str,
) -> List[Dict[str, str]]:
    title = record.get("title", "未知标题")
    content = truncate_text((record.get("content") or "").strip())
    url = record.get("url", "未知链接")
    prompt = USER_PROMPT_TEMPLATE.format(
        qa_count=qa_count,
        title=title,
        url=url,
        content=content,
        content_length=len(content),
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip("`").startswith("json"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_model_response(raw_text: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    cleaned = strip_code_fences(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"无法解析模型返回的JSON：{exc}\n原文：{raw_text}") from exc

    if isinstance(data, dict) and data.get("skip"):
        reason = data.get("reason") or data.get("message") or "模型判定样本质量不足"
        return None, reason

    qas = data.get("qas") if isinstance(data, dict) else None
    if not isinstance(qas, list):
        raise ValueError("模型返回JSON缺少 'qas' 列表")
    validated: List[Dict[str, Any]] = []
    for item in qas:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        answer = item.get("answer")
        if not question or not answer:
            continue
        validated.append(
            {
                "question": question.strip(),
                "answer": answer.strip(),
                "answer_type": item.get("answer_type", "other"),
                "confidence_note": item.get("confidence_note", "").strip(),
            }
        )
    if not validated:
        raise ValueError("模型返回中没有有效的问答对")
    return validated, None


def call_deepseek(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_retries: int,
) -> str:
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


def build_dataset_entry(
    qa: Dict[str, Any],
    record: Dict[str, Any],
    source_file: Path,
) -> Dict[str, Any]:
    return {
        "question": qa["question"],
        "answer": qa["answer"],
        "answer_type": qa.get("answer_type", "other"),
        "confidence_note": qa.get("confidence_note", ""),
        "source": {
            "title": record.get("title"),
            "url": record.get("url"),
            "word_count": record.get("word_count"),
        },
    }


def process_single_file(
    file_path: Path,
    out_fp: Optional[Any],
    args: argparse.Namespace,
    rng: random.Random,
    client: Optional[OpenAI],
) -> int:
    print(f"[INFO] 处理文件：{file_path}")
    # 先读取所有记录
    all_records = list(read_jsonl(file_path))
    if not all_records:
        print(f"[WARN] 文件 {file_path} 中没有有效记录")
        return 0
    
    # 使用智能采样：优先选择信息密度高的记录
    records = smart_sample(
        all_records,
        args.sample_size,
        rng,
        top_ratio=args.top_ratio,
    )
    print(f"[INFO] 从 {len(all_records)} 条记录中采样 {len(records)} 条（优先高信息密度）")
    written = 0
    qa_progress: Optional[tqdm] = None
    if not args.dry_run:
        expected_qas = args.sample_size * args.qas_per_record
        qa_progress = tqdm(
            total=expected_qas,
            desc=f"{file_path.name} QA",
            unit="条",
            leave=False,
        )
    for idx, record in enumerate(records, start=1):
        messages = build_messages(
            record,
            qa_count=args.qas_per_record,
            system_prompt=args.system_prompt,
        )
        if args.dry_run:
            print(f"\n[DRY-RUN] Prompt #{idx} from {file_path}:\n{messages[-1]['content']}\n")
            continue
        assert client is not None
        raw_text = call_deepseek(
            client=client,
            messages=messages,
            model=args.model,
            temperature=args.temperature,
            max_retries=args.max_retries,
        )
        try:
            qas, skip_reason = parse_model_response(raw_text)
        except ValueError as exc:
            print(f"[ERROR] 解析模型输出失败：{exc}", file=sys.stderr)
            continue
        if qas is None:
            reason_msg = skip_reason or "模型要求跳过该样本"
            print(f"[INFO] 模型跳过样本：{reason_msg}")
            continue
        for qa in qas:
            entry = build_dataset_entry(qa, record, file_path)
            if out_fp is not None:
                out_fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out_fp.flush()
            written += 1
        if qa_progress:
            qa_progress.update(len(qas))
        time.sleep(args.sleep_seconds)
    if qa_progress:
        qa_progress.close()
    return written


def process_file_with_api_key(
    file_path: Path,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_retries: int,
    sleep_seconds: float,
    sample_size: int,
    top_ratio: float,
    qas_per_record: int,
    system_prompt: str,
    dry_run: bool,
) -> Tuple[Path, int]:
    """使用指定的API key处理单个文件（用于多进程）"""
    rng = random.Random()
    client: Optional[OpenAI] = None
    if not dry_run:
        client = OpenAI(api_key=api_key, base_url=base_url)
    
    derived_output = derive_output_path(file_path)
    ensure_output_dir(derived_output)
    
    # 创建简化的 args 对象
    class SimpleArgs:
        def __init__(self):
            self.model = model
            self.temperature = temperature
            self.max_retries = max_retries
            self.sleep_seconds = sleep_seconds
            self.sample_size = sample_size
            self.top_ratio = top_ratio
            self.qas_per_record = qas_per_record
            self.system_prompt = system_prompt
            self.dry_run = dry_run
    
    args = SimpleArgs()
    written = 0
    if dry_run:
        written = process_single_file(file_path, None, args, rng, client)
    else:
        with derived_output.open("w", encoding="utf-8") as out_fp:
            written = process_single_file(file_path, out_fp, args, rng, client)
    
    return file_path, written


def find_cleaned_files() -> List[Path]:
    """自动扫描 data/cleaned 目录中的所有 jsonl 文件"""
    cleaned_dir = Path("data/cleaned")
    if not cleaned_dir.exists():
        return []
    return sorted(cleaned_dir.glob("*.jsonl"))


def main() -> None:
    args = parse_args()
    
    # 自动扫描 data/cleaned 目录或使用指定的文件
    if args.input_files:
        input_files = [Path(f) for f in args.input_files]
    else:
        input_files = find_cleaned_files()
        if not input_files:
            print("[ERROR] 未找到 data/cleaned 目录中的 jsonl 文件，请使用 --input-files 指定文件", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] 自动发现 {len(input_files)} 个文件在 data/cleaned 目录中")
    
    api_keys = [
        "sk-2db26a5af44b43a7af0b3e4a55cd29e6",
        "sk-8ce77f4caa9b45bfbf741a368be18840",
        "sk-42892705f5a642b18ff533680c380162",
        "sk-ad2fffdb5d194d0b95ecab1e6b37b370",
        "sk-3930f5f79ff1450eb5c6b1ac582bc246"
    ]
    
    # 如果指定了单一输出文件，使用旧逻辑（串行）
    if args.output is not None:
        print("[INFO] 检测到单一输出文件，使用串行处理模式")
        rng = random.Random()
        client: Optional[OpenAI] = None
        if not args.dry_run:
            client = OpenAI(api_key=api_keys[0], base_url=args.base_url)
        
        ensure_output_dir(args.output)
        total_written = 0
        with args.output.open("w", encoding="utf-8") as out_fp:
            for file_path in input_files:
                total_written += process_single_file(file_path, out_fp, args, rng, client)
        print(f"[INFO] 生成完毕，共写入 {total_written} 条问答数据 -> {args.output}")
        return
    
    # 并行处理模式：每个文件使用不同的API key
    print(f"[INFO] 使用 {args.max_workers} 个进程并行处理 {len(input_files)} 个文件")
    print(f"[INFO] 每个文件采样 {args.sample_size} 条记录，生成 {args.qas_per_record} 个QA对/记录")
    
    total_written = 0
    results: Dict[Path, int] = {}
    
    # 为每个文件分配API key（轮询方式）
    file_api_pairs = [
        (file_path, api_keys[i % len(api_keys)])
        for i, file_path in enumerate(input_files)
    ]
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(
                process_file_with_api_key,
                file_path,
                api_key,
                args.base_url,
                args.model,
                args.temperature,
                args.max_retries,
                args.sleep_seconds,
                args.sample_size,
                args.top_ratio,
                args.qas_per_record,
                args.system_prompt,
                args.dry_run,
            ): file_path
            for file_path, api_key in file_api_pairs
        }
        
        # 使用tqdm显示总体进度
        with tqdm(total=len(input_files), desc="处理文件", unit="个") as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    processed_file, written = future.result()
                    results[processed_file] = written
                    total_written += written
                    pbar.update(1)
                    pbar.set_postfix({"已写入": total_written, "当前文件": processed_file.name})
                except Exception as exc:
                    print(f"[ERROR] 处理文件 {file_path} 时出错：{exc}", file=sys.stderr)
                    pbar.update(1)
    
    if args.dry_run:
        print("[INFO] Dry-run 完成，未生成 QA 数据。")
    else:
        print(f"\n[INFO] 生成完毕，共写入 {total_written} 条问答数据")
        print(f"[INFO] 输出目录：data/qa/")
        print(f"[INFO] 各文件统计：")
        for file_path, count in sorted(results.items()):
            print(f"  - {file_path.name}: {count} 条")


if __name__ == "__main__":
    main()

