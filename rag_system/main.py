"""
RAG 系统主入口脚本
"""
import argparse
import sys
import traceback
from pathlib import Path

from .config import (
    RAG_DATA_DIR,
    VECTOR_DB_DIR,
    VLLM_BASE_URL,
    VLLM_MODEL
)


def build_index_command(args):
    """构建向量索引命令（使用LightRAG）"""
    import asyncio
    from .lightrag_pipeline import build_lightrag_index
    
    if not RAG_DATA_DIR.exists():
        RAG_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    rag_files = list(RAG_DATA_DIR.glob("*.jsonl"))
    if not rag_files:
        print(f"错误: 在 {RAG_DATA_DIR} 中未找到 RAG 数据文件")
        print(f"提示: 请确保 data/rag 目录中有已处理好的 JSONL 文件")
        return
    
    working_dir = Path(args.vector_db) if args.vector_db else VECTOR_DB_DIR
    
    print(f"[INFO] 使用LightRAG构建索引...")
    print(f"[INFO] 工作目录: {working_dir}")
    print(f"[INFO] 提示: 按 Ctrl+C 可以安全中断（会清理资源）")
    
    try:
        asyncio.run(build_lightrag_index(
            rag_data_dir=RAG_DATA_DIR,
            working_dir=working_dir,
            base_url=args.vllm_base_url,
            model=args.vllm_model,
            force_recreate=args.force_recreate,
            check_vllm=not args.skip_vllm_check
        ))
        print("LightRAG索引构建完成！")
    except KeyboardInterrupt:
        print("\n[INFO] 索引构建已被用户中断")
        print("[INFO] 已保存的索引数据不会丢失，下次可以继续构建")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] 索引构建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def query_command(args):
    """查询命令（使用LightRAG）"""
    import asyncio
    from .lightrag_pipeline import create_lightrag_pipeline
    
    working_dir = Path(args.vector_db) if args.vector_db else VECTOR_DB_DIR
    
    async def _query():
        pipeline = await create_lightrag_pipeline(
            working_dir=working_dir,
            base_url=args.vllm_base_url,
            model=args.vllm_model
        )
        
        try:
            if args.query:
                # 单次查询
                result = await pipeline.query(
                    args.query,
                    mode=args.mode or "hybrid",
                    return_context=args.show_context
                )
                print(f"\n🤖 AI 回答:")
                print("=" * 50)
                print(result['answer'])
                print("=" * 50)
                print(f"\n📝 查询模式: {result['mode']}")
            else:
                # 交互式查询
                print("\n" + "="*60)
                print("LightRAG 系统已就绪！您可以开始提问了。")
                print("输入 'quit' 或 '退出' 来结束程序")
                print("="*60)
                
                while True:
                    user_input = input("\n请输入您的问题: ").strip()
                    
                    if user_input.lower() in ['quit', '退出', 'exit']:
                        print("感谢使用，再见！")
                        break
                    
                    if not user_input:
                        print("问题不能为空，请重新输入。")
                        continue
                    
                    try:
                        result = await pipeline.query(
                            user_input,
                            mode=args.mode or "hybrid",
                            return_context=args.show_context
                        )
                        
                        print(f"\n🤖 AI 回答:")
                        print("=" * 50)
                        print(result['answer'])
                        print("=" * 50)
                        print(f"\n📝 查询模式: {result['mode']}")
                    except Exception as e:
                        print(f"❌ 查询过程中出现错误: {e}")
                        print("\n" + "=" * 60)
                        print("完整错误信息 (Traceback):")
                        print("=" * 60)
                        traceback.print_exc()
                        print("=" * 60)
        finally:
            await pipeline.finalize()
    
    asyncio.run(_query())


def main():
    parser = argparse.ArgumentParser(description="CUSZ-GPT RAG 系统")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # optimize 命令：优化 RAG 数据（从 cleaned 到 rag）
    optimize_parser = subparsers.add_parser('optimize', help='优化 RAG 数据质量（从 cleaned 生成 rag）')
    optimize_parser.add_argument('--input', type=str, required=True, help='输入的 cleaned JSONL 文件路径')
    optimize_parser.add_argument('--output', type=str, default=None, help='输出的 RAG JSONL 文件路径（可选，默认：data/rag/xxx.jsonl）')
    optimize_parser.add_argument('--base-url', type=str, default=None, help='vLLM API base URL（默认：http://localhost:8000/v1）')
    optimize_parser.add_argument('--model', type=str, default=None, help='模型名称（默认：Qwen/Qwen2.5-14B-Instruct）')
    optimize_parser.add_argument('--max-workers', type=int, default=128, help='并行处理线程数')
    optimize_parser.add_argument('--max-retries', type=int, default=3, help='最大重试次数')
    optimize_parser.add_argument('--sample-size', type=int, default=None, help='采样数量（用于测试）')
    
    # build-index 命令：构建向量索引（使用LightRAG）
    build_parser = subparsers.add_parser('build-index', help='构建向量索引（使用LightRAG）')
    build_parser.add_argument('--vector-db', type=str, default=None, help='LightRAG工作目录（默认：data/vector_db/lightrag）')
    build_parser.add_argument('--force-recreate', action='store_true', help='强制重新创建索引')
    build_parser.add_argument('--vllm-base-url', type=str, default=None, help=f'vLLM API base URL（默认：{VLLM_BASE_URL}）')
    build_parser.add_argument('--vllm-model', type=str, default=None, help=f'vLLM 模型名称（默认：{VLLM_MODEL}）')
    build_parser.add_argument('--skip-vllm-check', action='store_true', help='跳过 vLLM 服务检查')
    
    # query 命令：查询（使用LightRAG）
    query_parser = subparsers.add_parser('query', help='执行查询（使用LightRAG）')
    query_parser.add_argument('--query', type=str, default=None, help='查询问题（如果为空则进入交互模式）')
    query_parser.add_argument('--vector-db', type=str, default=None, help='LightRAG工作目录（默认：data/vector_db/lightrag）')
    query_parser.add_argument('--show-context', action='store_true', help='显示检索到的上下文')
    query_parser.add_argument('--mode', type=str, default='hybrid', choices=['local', 'global', 'hybrid', 'naive', 'mix', 'bypass'], help='查询模式（默认：hybrid）')
    query_parser.add_argument('--vllm-base-url', type=str, default=None, help=f'vLLM API base URL（默认：{VLLM_BASE_URL}）')
    query_parser.add_argument('--vllm-model', type=str, default=None, help=f'vLLM 模型名称（默认：{VLLM_MODEL}）')
    
    args = parser.parse_args()
    
    if args.command == 'build-index':
        build_index_command(args)
    elif args.command == 'query':
        query_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

