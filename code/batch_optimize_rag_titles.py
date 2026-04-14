#!/usr/bin/env python3
"""
批量优化所有 cleaned 数据，生成 RAG 数据
使用现有的 improve_rag_titles.py 脚本
"""
import subprocess
import sys
from pathlib import Path

def main():
    """批量优化所有 cleaned 文件，生成 RAG 文件"""
    cleaned_dir = Path("data/cleaned")
    rag_dir = Path("data/rag")
    
    if not cleaned_dir.exists():
        print(f"错误: {cleaned_dir} 目录不存在")
        sys.exit(1)
    
    # 获取所有 cleaned 文件
    cleaned_files = sorted(cleaned_dir.glob("*.jsonl"))
    
    if not cleaned_files:
        print("未找到 cleaned 数据文件")
        sys.exit(1)
    
    print(f"找到 {len(cleaned_files)} 个 cleaned 文件\n")
    
    # 逐个处理
    for i, cleaned_file in enumerate(cleaned_files, 1):
        print(f"[{i}/{len(cleaned_files)}] 处理: {cleaned_file.name}")
        print("-" * 60)
        
        try:
            # 调用 improve_rag_titles.py（从 cleaned 生成 rag）
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/improve_rag_titles.py",
                    "--input", str(cleaned_file),
                    "--max-workers", "16",
                    "--model", "Qwen/Qwen2.5-14B-Instruct",  # 使用当前 vLLM 运行的模型
                    "--base-url", "http://localhost:8000/v1",
                ],
                check=True,
                capture_output=False,
            )
            print(f"✓ 完成: {cleaned_file.name}\n")
        except subprocess.CalledProcessError as e:
            print(f"✗ 失败: {cleaned_file.name} (退出码: {e.returncode})\n")
        except KeyboardInterrupt:
            print("\n\n用户中断，已停止处理")
            sys.exit(1)
    
    print("=" * 60)
    print("批量优化完成！")

if __name__ == "__main__":
    main()

