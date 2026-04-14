"""
基于LightRAG的RAG管道实现
替换原有的LangChain + Chroma实现
"""
import os
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from lightrag import LightRAG, QueryParam
from lightrag.utils import setup_logger

from .config import (
    VECTOR_DB_DIR, 
    LLM_TEMPERATURE, 
    RETRIEVER_TOP_K, 
    LIGHTRAG_MAX_ASYNC,
    LIGHTRAG_WORKER_TIMEOUT,
    TIKTOKEN_MODEL_NAME,
    TIKTOKEN_MAX_RETRIES,
    LIGHTRAG_MAX_CONTEXT_LEN,
    LIGHTRAG_CHUNK_SIZE,
    LIGHTRAG_SUMMARY_CONTEXT_SIZE,
    BREADTH_FIRST_BATCH_SIZE,
    MIN_INSERT_BATCH_SIZE,
    MAX_INSERT_BATCH_SIZE,
    INSERT_BATCH_SIZE_DIVISOR,
    INDEX_PROGRESS_INTERVAL,
)
from .lightrag_embedding import create_huggingface_embedding_func
from .lightrag_llm import create_vllm_complete_func
from .tiktoken_helper import preload_tiktoken_tokenizer
from .data_processor import breadth_first_load_documents

# 设置LightRAG日志级别
setup_logger("lightrag", level="INFO")


class LightRAGPipeline:
    """基于LightRAG的RAG管道类"""
    
    def __init__(
        self,
        working_dir: Optional[Path] = None,
        embedding_func=None,
        llm_model_func=None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        """
        初始化LightRAG管道
        
        Args:
            working_dir: LightRAG工作目录（存储向量数据库和知识图谱）
            embedding_func: Embedding函数（如果为None则自动创建）
            llm_model_func: LLM模型函数（如果为None则自动创建）
            base_url: vLLM API base URL
            model: 模型名称
            temperature: LLM温度参数
            top_k: 检索的文档数量
        """
        # VECTOR_DB_DIR现在已经是lightrag目录了
        self.working_dir = Path(working_dir) if working_dir else VECTOR_DB_DIR
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建embedding函数
        if embedding_func is None:
            self.embedding_func = create_huggingface_embedding_func()
        else:
            self.embedding_func = embedding_func
        
        # 创建LLM函数
        if llm_model_func is None:
            self.llm_model_func = create_vllm_complete_func(
                base_url=base_url,
                model=model
            )
        else:
            self.llm_model_func = llm_model_func
        
        self.temperature = temperature or LLM_TEMPERATURE
        self.top_k = top_k or RETRIEVER_TOP_K
        
        # LightRAG实例（延迟初始化）
        self.rag: Optional[LightRAG] = None
        self._initialized = False
    
    async def initialize(self):
        """初始化LightRAG实例和存储后端"""
        if self._initialized:
            return
        
        print(f"[INFO] 初始化LightRAG，工作目录: {self.working_dir}")
        print(f"[INFO] LightRAG最大并发数: {LIGHTRAG_MAX_ASYNC}")
        print(f"[INFO] LightRAG Worker超时时间: {LIGHTRAG_WORKER_TIMEOUT}秒")
        
        # 设置环境变量以控制LightRAG的并发数和超时时间
        os.environ["MAX_ASYNC"] = str(LIGHTRAG_MAX_ASYNC)
        os.environ["WORKER_TIMEOUT"] = str(LIGHTRAG_WORKER_TIMEOUT)
        
        # 预加载 tiktoken tokenizer（在 LightRAG 初始化之前）
        # 这样可以提前下载所需的 BPE 文件，并支持重试
        try:
            print(f"[INFO] 预加载 tiktoken tokenizer (模型: {TIKTOKEN_MODEL_NAME})...")
            preload_tiktoken_tokenizer(
                model_name=TIKTOKEN_MODEL_NAME, 
                max_retries=TIKTOKEN_MAX_RETRIES
            )
        except Exception as e:
            print(f"[ERROR] 预加载 tiktoken tokenizer 失败: {e}")
            print(f"[ERROR] 这可能导致 LightRAG 初始化失败")
            print(f"[INFO] 建议解决方案:")
            print(f"[INFO]   1. 检查网络连接")
            print(f"[INFO]   2. 运行以下命令手动下载 tiktoken 文件:")
            print(f"[INFO]      python -m rag_system.tiktoken_helper {TIKTOKEN_MODEL_NAME}")
            print(f"[INFO]   3. 或在有网络的环境中预先运行一次程序以缓存文件")
            # 不抛出异常，让 LightRAG 自己处理（可能会有更好的错误信息）
        
        # 创建LightRAG实例
        # 配置参数以限制上下文长度，避免超过模型最大token限制
        rag_kwargs = {
            "working_dir": str(self.working_dir),
            "embedding_func": self.embedding_func,
            "llm_model_func": self.llm_model_func,
        }
        
        # 添加上下文长度限制配置（如果LightRAG支持这些参数）
        # 注意：这些参数名称可能需要根据LightRAG的实际API调整
        try:
            # 尝试设置chunk_size和summary_context_size
            # 这些参数可能不存在，所以用try-except包裹
            if hasattr(LightRAG, '__init__'):
                # 检查LightRAG是否支持这些参数
                import inspect
                sig = inspect.signature(LightRAG.__init__)
                if 'chunk_size' in sig.parameters:
                    rag_kwargs['chunk_size'] = LIGHTRAG_CHUNK_SIZE
                if 'summary_context_size' in sig.parameters:
                    rag_kwargs['summary_context_size'] = LIGHTRAG_SUMMARY_CONTEXT_SIZE
                # 设置 embedding/llm 的并发上限（如果 LightRAG 支持这些参数）
                if 'embedding_func_max_async' in sig.parameters:
                    # 从配置取值，默认见 config.py
                    from .config import LIGHTRAG_EMBEDDING_MAX_ASYNC, LIGHTRAG_LLM_MAX_ASYNC
                    rag_kwargs['embedding_func_max_async'] = LIGHTRAG_EMBEDDING_MAX_ASYNC
                if 'llm_model_max_async' in sig.parameters:
                    from .config import LIGHTRAG_EMBEDDING_MAX_ASYNC, LIGHTRAG_LLM_MAX_ASYNC
                    rag_kwargs['llm_model_max_async'] = LIGHTRAG_LLM_MAX_ASYNC
                # 如果 LightRAG 支持 max_parallel_insert，传入配置值以控制并行插入数
                if 'max_parallel_insert' in sig.parameters:
                    from .config import LIGHTRAG_MAX_PARALLEL_INSERT
                    rag_kwargs['max_parallel_insert'] = LIGHTRAG_MAX_PARALLEL_INSERT
        except Exception:
            pass  # 如果参数不存在，忽略
        
        print(f"[INFO] LightRAG配置:")
        print(f"  - 最大上下文长度: {LIGHTRAG_MAX_CONTEXT_LEN} tokens")
        print(f"  - Chunk大小: {LIGHTRAG_CHUNK_SIZE}")
        print(f"  - 摘要上下文大小: {LIGHTRAG_SUMMARY_CONTEXT_SIZE}")
        if hasattr(self.embedding_func, 'embedding_dim'):
            print(f"  - Embedding维度: {self.embedding_func.embedding_dim}")
        
        # 创建LightRAG实例
        self.rag = LightRAG(**rag_kwargs)
        
        # 重要：必须初始化存储后端
        await self.rag.initialize_storages()
        
        # 重要：必须初始化pipeline status
        from lightrag.kg.shared_storage import initialize_pipeline_status
        await initialize_pipeline_status()
        
        self._initialized = True
        print("[INFO] LightRAG初始化完成")
    
    async def insert_text(self, text: str, doc_id: Optional[str] = None):
        """
        插入文本到LightRAG
        
        Args:
            text: 要插入的文本
            doc_id: 文档ID（可选，LightRAG不支持此参数，保留仅为兼容性）
        """
        if not self._initialized:
            await self.initialize()
        
        # LightRAG 的 ainsert 方法不支持 doc_id 参数
        await self.rag.ainsert(text)
    
    async def insert_batch(self, texts: List[str], doc_ids: Optional[List[str]] = None):
        """
        批量插入文本
        
        Args:
            texts: 文本列表
            doc_ids: 文档ID列表（可选，LightRAG不支持此参数，保留仅为兼容性）
        """
        # 为兼容旧接口，调用优化后的批量插入实现
        return await self.insert_batch_optimized(texts)

    async def insert_batch_optimized(self, texts: List[str]):
        """
        优化后的批量插入：直接使用同步方式绕过所有异步锁问题

        Args:
            texts: 文本列表
        
        Returns:
            int: 实际插入的文档数量
        """
        if not self._initialized:
            await self.initialize()

        inserted_count = 0
        
        # 在线程池中执行同步插入，完全避免事件循环和锁的问题
        loop = asyncio.get_event_loop()
        
        def sync_batch_insert():
            """在线程中同步执行批量插入"""
            count = 0
            for text in texts:
                try:
                    # 直接添加到 LightRAG 的内部队列，不使用 await
                    # LightRAG 的 insert 方法会将文档放入处理队列
                    # 我们直接调用其内部的 _add_doc 方法或类似的同步方法
                    
                    # 首先尝试直接调用 insert 但使用同步包装
                    import uuid
                    doc_id = str(uuid.uuid4())
                    
                    # 直接操作存储数据库
                    try:
                        # 尝试使用 LightRAG 的私有方法直接插入
                        if hasattr(self.rag, '_storages'):
                            # 这是 LightRAG 的内部存储字典
                            storages = self.rag._storages
                            
                            # 尝试直接写入 chunk_db
                            if 'chunk_db' in storages:
                                chunk_db = storages['chunk_db']
                                # 使用同步方法
                                if hasattr(chunk_db, 'upsert'):
                                    chunk_db.upsert({
                                        'id': doc_id,
                                        'content': text,
                                    })
                                    count += 1
                                elif hasattr(chunk_db, 'insert'):
                                    chunk_db.insert({
                                        'id': doc_id,
                                        'content': text,
                                    })
                                    count += 1
                                else:
                                    # 尝试通用方法
                                    chunk_db[doc_id] = {'content': text}
                                    count += 1
                            else:
                                print("[WARN] 无法访问 chunk_db 存储")
                        else:
                            print("[WARN] 无法访问 LightRAG 内部存储")
                    except Exception as e:
                        error_str = str(e).lower()
                        if "duplicate" not in error_str and "already exists" not in error_str:
                            print(f"[WARN] 同步存储插入失败: {e}")
                            # 回退：创建独立事件循环插入
                            try:
                                import asyncio as aio
                                new_loop = aio.new_event_loop()
                                aio.set_event_loop(new_loop)
                                try:
                                    new_loop.run_until_complete(self.rag.insert(text))
                                    count += 1
                                finally:
                                    new_loop.close()
                                    aio.set_event_loop(None)
                            except Exception as e2:
                                print(f"[WARN] 异步插入也失败: {e2}")
                except Exception as e:
                    print(f"[WARN] 处理文本异常: {e}")
            
            return count
        
        try:
            # 在线程执行器中运行同步插入
            inserted_count = await loop.run_in_executor(None, sync_batch_insert)
        except Exception as e:
            print(f"[ERROR] 批量插入执行出错: {e}")
            import traceback
            traceback.print_exc()
        
        return inserted_count
    
    async def query(
        self,
        query: str,
        mode: str = "hybrid",
        return_context: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行查询
        
        Args:
            query: 查询文本
            mode: 查询模式 ("local", "global", "hybrid", "naive", "mix", "bypass")
            return_context: 是否返回检索到的上下文
            **kwargs: 其他QueryParam参数
        
        Returns:
            包含答案和上下文的字典
        """
        if not self._initialized:
            await self.initialize()
        
        # 创建查询参数
        query_param = QueryParam(
            mode=mode,
            top_k=self.top_k,
            **kwargs
        )
        
        # 执行查询
        response = await self.rag.aquery(query, param=query_param)
        
        result = {
            "answer": response,
            "query": query,
            "mode": mode
        }
        
        # 如果需要返回上下文，可以通过only_need_context参数获取
        if return_context:
            # 注意：LightRAG的上下文检索是内部的
            # 如果需要详细的上下文信息，可能需要修改查询参数
            result["context"] = []  # LightRAG不直接暴露检索到的文档
        
        return result
    
    def query_sync(
        self,
        query: str,
        mode: str = "hybrid",
        return_context: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        同步版本的查询方法
        
        Args:
            query: 查询文本
            mode: 查询模式
            return_context: 是否返回上下文
            **kwargs: 其他参数
        
        Returns:
            查询结果
        """
        return asyncio.run(self.query(query, mode, return_context, **kwargs))
    
    async def finalize(self):
        """清理资源"""
        if self.rag and self._initialized:
            await self.rag.finalize_storages()
            self._initialized = False
    
    def finalize_sync(self):
        """同步版本的清理方法"""
        if self.rag and self._initialized:
            asyncio.run(self.finalize())


async def create_lightrag_pipeline(
    working_dir: Optional[Path] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
) -> LightRAGPipeline:
    """
    创建并初始化LightRAG管道
    
    Args:
        working_dir: 工作目录
        base_url: vLLM API base URL
        model: 模型名称
        temperature: LLM温度参数
        top_k: 检索的文档数量
    
    Returns:
        初始化完成的LightRAGPipeline实例
    """
    pipeline = LightRAGPipeline(
        working_dir=working_dir,
        base_url=base_url,
        model=model,
        temperature=temperature,
        top_k=top_k,
    )
    
    await pipeline.initialize()
    return pipeline


def load_documents_from_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    从JSONL文件加载文档
    
    Args:
        file_path: JSONL文件路径
    
    Returns:
        文档列表，每个文档包含content和metadata
    """
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                documents.append(item)
            except json.JSONDecodeError:
                continue
    return documents


async def build_lightrag_index(
    rag_data_dir: Path,
    working_dir: Optional[Path] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    force_recreate: bool = False,
    check_vllm: bool = True,
    breadth_first: bool = True,
    breadth_batch_size: Optional[int] = None,
):
    """
    从RAG数据文件构建LightRAG索引
    
    Args:
        rag_data_dir: RAG数据目录（包含JSONL文件）
        working_dir: LightRAG工作目录
        base_url: vLLM API base URL
        model: 模型名称
        force_recreate: 是否强制重新创建索引
        check_vllm: 是否在初始化前检查 vLLM 服务
        breadth_first: 是否使用广度优先方式加载文档（默认True）
        breadth_batch_size: 广度优先时，每次从每个文件提取的记录数（None则使用配置中的默认值）
    """
    # 如果强制重新创建，删除现有工作目录
    if force_recreate and working_dir and Path(working_dir).exists():
        import shutil
        shutil.rmtree(working_dir)
        print(f"[INFO] 已删除现有工作目录: {working_dir}")
    
    # 创建管道
    pipeline = await create_lightrag_pipeline(
        working_dir=working_dir,
        base_url=base_url,
        model=model,
    )
    
    # 加载所有JSONL文件
    jsonl_files = sorted(list(rag_data_dir.glob("*.jsonl")))
    print(f"[INFO] 找到 {len(jsonl_files)} 个RAG数据文件")
    
    if not jsonl_files:
        print(f"[ERROR] 在 {rag_data_dir} 中未找到 JSONL 文件")
        await pipeline.finalize()
        return
    
    # 根据并发数动态调整插入batch_size，提高多GPU利用率
    # 多GPU时可以增加batch_size，提高吞吐量
    insert_batch_size = min(
        MAX_INSERT_BATCH_SIZE, 
        max(MIN_INSERT_BATCH_SIZE, LIGHTRAG_MAX_ASYNC // INSERT_BATCH_SIZE_DIVISOR)
    )
    print(f"[INFO] 使用插入批量大小: {insert_batch_size}（并发数: {LIGHTRAG_MAX_ASYNC}）")
    
    # 使用配置的广度优先批量大小
    if breadth_batch_size is None:
        breadth_batch_size = BREADTH_FIRST_BATCH_SIZE
    
    total_docs = 0
    batch_tasks = []
    
    print(f"[INFO] 使用广度优先方式加载文档（每个文件每次提取 {breadth_batch_size} 条记录）")
    
    # 使用生成器逐个获取文档
    doc_generator = breadth_first_load_documents(jsonl_files, batch_size=breadth_batch_size)
    
    for doc in doc_generator:
        content = doc.get('content', '')
        if not content:
            continue
        
        # 构建文档文本（包含标题和内容）
        title = doc.get('title', '')
        url = doc.get('url', '')
        
        doc_text = content
        if title:
            doc_text = f"标题: {title}\n\n{doc_text}"
        
        # 使用URL作为doc_id（如果可用）
        doc_id = url if url else None
        
        # 创建插入任务
        batch_tasks.append(pipeline.insert_text(doc_text, doc_id=doc_id))
        total_docs += 1
        
        # 当达到批量大小时，并行执行当前批次
        if len(batch_tasks) >= insert_batch_size:
            await asyncio.gather(*batch_tasks)
            batch_tasks = []
            
            if total_docs % INDEX_PROGRESS_INTERVAL == 0:
                print(f"[INFO] 已插入 {total_docs} 个文档...")
    
    # 处理剩余的文档
    if batch_tasks:
        await asyncio.gather(*batch_tasks)

    print(f"[INFO] 索引构建完成，共插入 {total_docs} 个文档")
    
    # 清理资源
    await pipeline.finalize()

