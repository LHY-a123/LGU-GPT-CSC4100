"""
LightRAG Embedding函数包装器
支持HuggingFace离线embedding模型
支持多GPU并行（在不同GPU上加载多个模型实例）
"""
import os
import asyncio
from typing import List, Union, Optional
import numpy as np
import threading

from .config import (
    EMBEDDING_MODEL_NAME, 
    EMBEDDING_MODEL_PATH, 
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_USE_MULTI_GPU,
    EMBEDDING_GPU_IDS,
    EMBEDDING_GPU_COUNT,
)


def create_huggingface_embedding_func():
    """
    创建HuggingFace embedding函数，支持离线模型和多GPU并行
    
    Returns:
        embedding函数，接受文本列表，返回numpy数组
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "需要安装sentence-transformers: pip install sentence-transformers"
        )
    
    import torch
    
    # 确定模型路径
    model_name_or_path = EMBEDDING_MODEL_PATH or EMBEDDING_MODEL_NAME
    
    # 检查是否使用本地路径
    use_local_path = EMBEDDING_MODEL_PATH is not None and os.path.exists(EMBEDDING_MODEL_PATH)
    
    # 检测GPU设备
    use_gpu = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if use_gpu else 0
    
    # 决定是否使用多GPU
    if EMBEDDING_USE_MULTI_GPU and use_gpu and gpu_count > 1:
        # 确定要使用的GPU列表
        if EMBEDDING_GPU_IDS:
            # 使用指定的GPU列表
            gpu_ids = [int(gid.strip()) for gid in EMBEDDING_GPU_IDS.split(',') if gid.strip()]
            gpu_ids = [gid for gid in gpu_ids if 0 <= gid < gpu_count]
            if not gpu_ids:
                print(f"[WARN] 指定的GPU ID无效，使用默认配置")
                gpu_ids = list(range(min(EMBEDDING_GPU_COUNT, gpu_count)))
        else:
            # 使用前N个GPU
            gpu_ids = list(range(min(EMBEDDING_GPU_COUNT, gpu_count)))
        
        if len(gpu_ids) < 2:
            print(f"[WARN] 可用的GPU数量不足，使用单GPU模式")
            return _create_single_gpu_embedding_func(model_name_or_path, use_local_path, use_gpu)
        
        print(f"[INFO] 使用多GPU模式: {len(gpu_ids)} 个GPU")
        print(f"[INFO] GPU列表: {gpu_ids}")
        return _create_multi_gpu_embedding_func(model_name_or_path, use_local_path, gpu_ids)
    else:
        # 单GPU模式
        return _create_single_gpu_embedding_func(model_name_or_path, use_local_path, use_gpu)


def _create_single_gpu_embedding_func(model_name_or_path: str, use_local_path: bool, use_gpu: bool):
    """创建单GPU embedding函数"""
    from sentence_transformers import SentenceTransformer
    import torch
    
    device = 'cuda:0' if use_gpu else 'cpu'
    print(f"[INFO] 加载embedding模型: {model_name_or_path}")
    if use_local_path:
        print(f"[INFO] 使用本地模型路径: {EMBEDDING_MODEL_PATH}")
    else:
        print(f"[INFO] 使用HuggingFace模型: {EMBEDDING_MODEL_NAME}")
        print(f"[INFO] 注意: 如果模型已下载到缓存，将从本地加载；否则会从网络下载")
    
    print(f"[INFO] 使用设备: {device}")
    if use_gpu:
        print(f"[INFO] GPU数量: {torch.cuda.device_count()}")
        print(f"[INFO] GPU名称: {torch.cuda.get_device_name(0)}")
    
    # 加载模型
    try:
        if use_local_path:
            model = SentenceTransformer(
                EMBEDDING_MODEL_PATH,
                trust_remote_code=True,
                device=device
            )
            print(f"[INFO] 从本地路径加载模型成功")
        else:
            model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                trust_remote_code=True,
                device=device
            )
            print(f"[INFO] 从HuggingFace加载模型成功")
    except Exception as e:
        print(f"[ERROR] 加载模型失败: {e}")
        raise
    
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"[INFO] Embedding模型加载完成，向量维度: {embedding_dim}")
    
    async def embedding_func(texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        def _encode():
            return model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=EMBEDDING_BATCH_SIZE
            )
        
        embeddings = await asyncio.to_thread(_encode)
        return embeddings
    
    # 设置embedding_dim属性（LightRAG需要这个属性）
    embedding_func.embedding_dim = embedding_dim
    
    # 添加标识，确保LightRAG知道这是自定义的embedding函数
    embedding_func._is_custom_embedding = True
    
    return embedding_func


def _create_multi_gpu_embedding_func(model_name_or_path: str, use_local_path: bool, gpu_ids: List[int]):
    """创建多GPU embedding函数（在不同GPU上加载多个模型实例）"""
    from sentence_transformers import SentenceTransformer
    import torch
    
    print(f"[INFO] 加载embedding模型: {model_name_or_path}")
    if use_local_path:
        print(f"[INFO] 使用本地模型路径: {EMBEDDING_MODEL_PATH}")
    else:
        print(f"[INFO] 使用HuggingFace模型: {EMBEDDING_MODEL_NAME}")
    
    # 在每个GPU上加载模型实例
    models = []
    for gpu_id in gpu_ids:
        device = f'cuda:{gpu_id}'
        print(f"[INFO] 在 {device} 上加载模型实例...")
        try:
            if use_local_path:
                model = SentenceTransformer(
                    EMBEDDING_MODEL_PATH,
                    trust_remote_code=True,
                    device=device
                )
            else:
                model = SentenceTransformer(
                    EMBEDDING_MODEL_NAME,
                    trust_remote_code=True,
                    device=device
                )
            models.append((model, gpu_id))
            print(f"[INFO] GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)}) 模型加载成功")
        except Exception as e:
            print(f"[ERROR] 在GPU {gpu_id}上加载模型失败: {e}")
            raise
    
    embedding_dim = models[0][0].get_sentence_embedding_dimension()
    print(f"[INFO] Embedding模型加载完成，向量维度: {embedding_dim}")
    print(f"[INFO] 已加载 {len(models)} 个模型实例，使用轮询负载均衡")
    print(f"[INFO] 并行机制: LightRAG的asyncio.gather + 多GPU模型实例实现真正的并行处理")
    
    # 负载均衡：跟踪每个GPU的活跃任务数
    _load_lock = threading.Lock()
    _active_tasks = {gpu_id: 0 for gpu_id in gpu_ids}  # 每个GPU的活跃任务数
    
    async def embedding_func(texts: Union[str, List[str]]) -> np.ndarray:
        """
        多GPU embedding函数，使用基于负载的动态选择
        选择活跃任务数最少的GPU
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # 选择负载最低的GPU
        with _load_lock:
            # 找到活跃任务数最少的GPU
            min_load = min(_active_tasks.values())
            candidates = [i for i, gpu_id in enumerate(gpu_ids) if _active_tasks[gpu_id] == min_load]
            # 如果有多个GPU负载相同，选择第一个（可以进一步优化为随机选择）
            selected_idx = candidates[0]
            selected_gpu_id = gpu_ids[selected_idx]
            _active_tasks[selected_gpu_id] += 1
        
        model, gpu_id = models[selected_idx]
        
        try:
            def _encode():
                return model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=EMBEDDING_BATCH_SIZE
                )
            
            embeddings = await asyncio.to_thread(_encode)
            return embeddings
        finally:
            # 任务完成后减少活跃任务数
            with _load_lock:
                _active_tasks[selected_gpu_id] -= 1
    
    # 设置embedding_dim属性（LightRAG需要这个属性）
    embedding_func.embedding_dim = embedding_dim
    
    # 添加标识，确保LightRAG知道这是自定义的embedding函数
    embedding_func._is_custom_embedding = True
    
    return embedding_func

