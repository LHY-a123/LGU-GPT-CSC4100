#!/bin/bash
"""
启动 Embedding API 服务的 Shell 脚本
"""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# 设置默认端口
PORT=${EMBEDDING_API_PORT:-8081}
HOST=${EMBEDDING_API_HOST:-0.0.0.0}

echo "=========================================="
echo "启动 Embedding API 服务"
echo "=========================================="
echo "[INFO] 工作目录: $PROJECT_ROOT"
echo "[INFO] 监听地址: $HOST:$PORT"
echo "[INFO] 日志文件: embedding_api.log"
echo "=========================================="

# 使用 nohup 在后台运行，输出重定向到日志文件
nohup python3 "$SCRIPT_DIR/start_embedding_api.py" \
    --host "$HOST" \
    --port "$PORT" \
    > embedding_api.log 2>&1 &

PID=$!
echo "[INFO] Embedding API 服务已启动，PID: $PID"
echo "[INFO] 查看日志: tail -f embedding_api.log"
echo "[INFO] 停止服务: kill $PID"
echo ""
echo "服务访问地址:"
echo "  - API: http://$HOST:$PORT/v1/embeddings"
echo "  - 文档: http://$HOST:$PORT/docs"
echo "  - 健康检查: http://$HOST:$PORT/health"






