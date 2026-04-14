#!/bin/bash
# 启动LightRAG Server (WebUI)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "启动 LightRAG Server (WebUI)"
echo "=========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3"
    exit 1
fi

# 检查LightRAG是否安装
if ! python3 -c "import lightrag" 2>/dev/null; then
    echo "错误: LightRAG未安装"
    echo "请先安装: pip install 'lightrag-hku[api]'"
    exit 1
fi

# 运行启动脚本
python3 "$SCRIPT_DIR/start_lightrag_server.py"

