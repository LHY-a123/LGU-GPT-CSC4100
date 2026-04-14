#!/usr/bin/env python3
"""
启动 Embedding API 服务（OpenAI-compatible）
使用 HuggingFace 模型提供 embedding 服务，供 LightRAG Server 使用

使用方法：
    python3 scripts/start_embedding_api.py --port 8081
"""
import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import asyncio

# 导入项目配置和 embedding 函数
from rag_system.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_USE_MULTI_GPU,
    EMBEDDING_GPU_IDS,
)
from rag_system.lightrag_embedding import create_huggingface_embedding_func

app = FastAPI(
    title="Embedding API Service",
    description="OpenAI-compatible Embedding API using HuggingFace models",
    version="1.0.0"
)

# 允许 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 embedding 函数
embedding_func = None
embedding_dim = None


class EmbeddingRequest(BaseModel):
    input: str | List[str]
    model: Optional[str] = None
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


@app.on_event("startup")
async def startup_event():
    """启动时加载 embedding 模型"""
    global embedding_func, embedding_dim
    
    print("[INFO] 正在加载 embedding 模型...")
    print(f"[INFO] 模型名称: {EMBEDDING_MODEL_NAME}")
    if EMBEDDING_MODEL_PATH:
        print(f"[INFO] 模型路径: {EMBEDDING_MODEL_PATH}")
    
    try:
        # 创建 embedding 函数
        embedding_func = create_huggingface_embedding_func()
        
        # 获取 embedding 维度（通过测试调用）
        test_embedding = await embedding_func("test")
        embedding_dim = test_embedding.shape[-1] if isinstance(test_embedding, np.ndarray) else len(test_embedding)
        
        print(f"[INFO] Embedding 模型加载成功，向量维度: {embedding_dim}")
        print(f"[INFO] 支持多GPU: {EMBEDDING_USE_MULTI_GPU}")
        if EMBEDDING_USE_MULTI_GPU:
            print(f"[INFO] 使用的GPU: {EMBEDDING_GPU_IDS}")
    except Exception as e:
        print(f"[ERROR] 加载 embedding 模型失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "model": EMBEDDING_MODEL_NAME,
        "embedding_dim": embedding_dim
    }


@app.get("/v1/models")
async def list_models():
    """列出可用的模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": EMBEDDING_MODEL_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "local"
            }
        ]
    }


@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    """创建 embedding"""
    global embedding_func, embedding_dim
    
    if embedding_func is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    # 处理输入
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input
    
    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")
    
    try:
        # 调用 embedding 函数
        embeddings = await embedding_func(texts)
        
        # 转换为列表格式
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        
        # 构建响应
        embedding_data = []
        for i, embedding in enumerate(embeddings):
            embedding_data.append({
                "object": "embedding",
                "embedding": embedding,
                "index": i
            })
        
        # 计算 token 使用量（近似值，使用字符数估算）
        total_tokens = sum(len(text) // 4 for text in texts)  # 简单估算
        
        response = EmbeddingResponse(
            data=embedding_data,
            model=request.model or EMBEDDING_MODEL_NAME,
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        )
        
        return response.dict()
    
    except Exception as e:
        print(f"[ERROR] 生成 embedding 失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description="启动 Embedding API 服务")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务监听地址（默认: 0.0.0.0）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="服务监听端口（默认: 8081）"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="工作进程数（默认: 1，embedding 模型已支持多GPU并行）"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Embedding API 服务")
    print("=" * 60)
    print(f"[INFO] 监听地址: {args.host}:{args.port}")
    print(f"[INFO] 模型: {EMBEDDING_MODEL_NAME}")
    if EMBEDDING_MODEL_PATH:
        print(f"[INFO] 模型路径: {EMBEDDING_MODEL_PATH}")
    print(f"[INFO] 多GPU支持: {EMBEDDING_USE_MULTI_GPU}")
    print(f"[INFO] API 文档: http://{args.host}:{args.port}/docs")
    print(f"[INFO] 健康检查: http://{args.host}:{args.port}/health")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )


if __name__ == "__main__":
    main()






