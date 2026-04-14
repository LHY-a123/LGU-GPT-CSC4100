#!/usr/bin/env python3
"""
RAG 系统测试脚本
用于测试 RAG 系统的查询功能
"""
import sys
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_system.vector_store import VectorStoreManager
from rag_system.rag_pipeline import RAGPipeline
from rag_system.llm_client import LLMClient
from rag_system.config import VECTOR_DB_DIR


def test_rag_queries(
    queries: list,
    vector_db_dir: Path = None,
    use_vllm: bool = True,
    show_context: bool = True,
    log_file: Path = None
):
    """
    测试 RAG 查询
    
    Args:
        queries: 测试查询列表
        vector_db_dir: 向量数据库目录
        use_vllm: 是否使用 vLLM
        show_context: 是否显示检索到的上下文
        log_file: 日志文件路径
    """
    if vector_db_dir is None:
        vector_db_dir = VECTOR_DB_DIR
    
    # 设置日志
    if log_file is None:
        log_file = project_root / "rag_test.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("RAG 系统测试")
    logger.info(f"向量数据库目录: {vector_db_dir}")
    logger.info(f"使用 vLLM: {use_vllm}")
    logger.info(f"日志文件: {log_file}")
    
    if not vector_db_dir.exists():
        logger.error(f"向量数据库目录不存在: {vector_db_dir}")
        return
    
    # 加载向量数据库
    vector_store_manager = VectorStoreManager(persist_directory=vector_db_dir)
    vector_store_manager.load_existing()
    
    # 创建 LLM 客户端
    llm_client = LLMClient(use_vllm=use_vllm)
    
    # 创建 RAG 管道
    pipeline = RAGPipeline(vector_store_manager, llm_client)
    
    logger.info("开始测试查询")
    
    # 执行测试查询
    for i, query in enumerate(queries, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"测试查询 {i}/{len(queries)}: {query}")
        
        result = pipeline.query(query, return_context=True)
        
        logger.info(f"\n🤖 AI 回答:\n{result['answer']}")
        logger.info(f"\n📊 统计: 检索到 {result['num_retrieved']} 个文档, 耗时 {result['query_time']:.2f}秒")
        
        if show_context and result.get('context'):
            logger.info(f"\n📚 参考来源 ({len(result['context'])} 个文档):")
            for j, doc in enumerate(result['context'], 1):
                title = doc.metadata.get('source_title', '未知标题')
                url = doc.metadata.get('url', '未知')
                logger.info(f"  [{j}] {title} | {url}")
                logger.info(f"      内容预览: {doc.page_content[:150]}...")
    
    logger.info("\n" + "=" * 60)
    logger.info("测试完成！")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG 系统测试脚本")
    parser.add_argument(
        '--vector-db',
        type=str,
        default=None,
        help='向量数据库目录（默认：data/vector_db）'
    )
    parser.add_argument(
        '--use-vllm',
        action='store_true',
        default=True,
        help='使用 vLLM（默认）'
    )
    parser.add_argument(
        '--no-vllm',
        action='store_true',
        help='不使用 vLLM（使用 DeepSeek API）'
    )
    parser.add_argument(
        '--query',
        type=str,
        action='append',
        help='测试查询（可多次使用）'
    )
    parser.add_argument(
        '--no-context',
        action='store_true',
        help='不显示检索到的上下文'
    )
    parser.add_argument(
        '--test-set',
        type=str,
        choices=['default', 'admissions', 'courses', 'comprehensive'],
        default='default',
        help='使用预设的测试查询集'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='日志文件路径（默认：rag_test.log）'
    )
    
    args = parser.parse_args()
    
    # 确定是否使用 vLLM
    use_vllm = args.use_vllm and not args.no_vllm
    
    # 确定向量数据库目录
    vector_db_dir = Path(args.vector_db) if args.vector_db else VECTOR_DB_DIR
    
    # 确定日志文件
    log_file = Path(args.log_file) if args.log_file else Path("rag_test.log")
    
    # 确定测试查询
    if args.query:
        queries = args.query
    else:
        # 使用预设的测试查询集
        test_sets = {
            'default': [
                "Deep Learning 相关的有哪些教授？",
                "SDS 有哪些顶尖教授？"
            ],
            'admissions': [
                "2025年招生政策是什么？",
                "综合评价招生如何申请？",
                "学费是多少？",
                "有哪些专业可以报考？",
            ],
            'courses': [
                "有哪些课程？",
                "数据科学相关的课程有哪些？",
            ],
            'comprehensive': [
                "香港中文大学（深圳）有哪些学院？",
                "如何申请奖学金？",
                "2025年招生政策是什么？",
                "综合评价招生如何申请？",
                "学费是多少？",
                "有哪些专业可以报考？",
                "数据科学相关的课程有哪些？",
                "学校的国际化程度如何？",
            ]
        }
        queries = test_sets.get(args.test_set, test_sets['default'])
    
    # 执行测试
    test_rag_queries(
        queries=queries,
        vector_db_dir=vector_db_dir,
        use_vllm=use_vllm,
        show_context=not args.no_context,
        log_file=log_file
    )


if __name__ == "__main__":
    main()

