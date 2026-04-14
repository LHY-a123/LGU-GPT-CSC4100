#!/bin/bash
# 下载嵌入模型脚本（使用镜像源）

# 获取脚本所在目录的父目录（项目根目录)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"

# 创建模型目录
mkdir -p "$MODELS_DIR"

# 配置镜像源（可以通过环境变量覆盖）
USE_MIRROR=${USE_MIRROR:-true}
if [ "$USE_MIRROR" = "true" ]; then
    export HF_ENDPOINT=https://hf-mirror.com
    echo "✓ 已配置 HuggingFace 镜像源: $HF_ENDPOINT"
else
    unset HF_ENDPOINT
    echo "✓ 使用原始 HuggingFace 源"
fi

echo "开始下载模型: BAAI/bge-large-zh-v1.5"
echo "模型将保存到: $MODELS_DIR"
echo "这可能需要一些时间，请耐心等待..."
echo ""

# 使用 Python 下载模型
python3 << PYTHON_SCRIPT
from sentence_transformers import SentenceTransformer
import os
import sys
import time

# 模型保存路径
models_dir = "$MODELS_DIR"
model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
use_mirror = "$USE_MIRROR" == "true"

# 设置镜像源
if use_mirror:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"使用镜像源: {os.environ.get('HF_ENDPOINT')}")
else:
    if "HF_ENDPOINT" in os.environ:
        del os.environ["HF_ENDPOINT"]
    print("使用原始 HuggingFace 源")

# 增加超时时间（5分钟）
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

try:
    print("正在下载模型...")
    if use_mirror:
        print("提示: 如果镜像源连接超时，可以设置 USE_MIRROR=false 使用原始源")
    print("")
    
    # 下载模型到指定目录
    model = SentenceTransformer(model_name, cache_folder=models_dir)
    # 保存模型到指定目录
    model_path = os.path.join(models_dir, model_name.replace("/", "--"))
    model.save(model_path)
    print("✓ 模型下载成功！")
    print(f"模型已保存到: {model_path}")
    print(f"\n提示: 系统会自动检测 models/ 目录中的模型")
except Exception as e:
    error_msg = str(e)
    print(f"\n❌ 下载失败: {error_msg}")
    print("\n可能的解决方案:")
    print("1. 如果镜像源超时，尝试使用原始源:")
    print("   USE_MIRROR=false bash scripts/download_embedding_model.sh")
    print("2. 检查网络连接")
    print("3. 尝试使用更小的模型:")
    print("   EMBEDDING_MODEL_NAME=BAAI/bge-m3 bash scripts/download_embedding_model.sh")
    print("4. 手动设置 HuggingFace 镜像源:")
    print("   export HF_ENDPOINT=https://hf-mirror.com")
    sys.exit(1)
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 下载完成！"
    echo ""
    echo "模型路径: $MODELS_DIR"
    echo ""
    echo "提示: 系统会自动检测 models/ 目录中的模型，或设置环境变量："
    echo "  export EMBEDDING_MODEL_PATH=$MODELS_DIR/BAAI--bge-large-zh-v1.5"
    echo ""
    echo "现在可以构建向量索引了："
    echo "  python3 -m rag_system.main build-index --use-vllm"
else
    echo ""
    echo "❌ 下载失败"
    echo ""
    echo "如果镜像源连接超时，可以尝试："
    echo "  USE_MIRROR=false bash scripts/download_embedding_model.sh"
    echo ""
    echo "或者使用更小的模型："
    echo "  EMBEDDING_MODEL_NAME=BAAI/bge-m3 bash scripts/download_embedding_model.sh"
    exit 1
fi
