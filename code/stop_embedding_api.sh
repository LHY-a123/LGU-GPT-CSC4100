#!/bin/bash
"""
停止 Embedding API 服务的 Shell 脚本
"""
echo "=========================================="
echo "停止 Embedding API 服务"
echo "=========================================="

# 查找进程
PIDS=$(ps aux | grep "start_embedding_api.py" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "[INFO] Embedding API 服务未运行"
    exit 0
fi

echo "[INFO] 找到进程: $PIDS"

# 停止进程
for PID in $PIDS; do
    echo "[INFO] 正在停止进程 $PID..."
    kill $PID
    
    # 等待进程结束（最多等待5秒）
    for i in {1..5}; do
        if ! kill -0 $PID 2>/dev/null; then
            echo "[INFO] 进程 $PID 已停止"
            break
        fi
        sleep 1
    done
    
    # 如果进程还在运行，强制杀死
    if kill -0 $PID 2>/dev/null; then
        echo "[WARN] 进程 $PID 未响应，强制停止..."
        kill -9 $PID
        echo "[INFO] 进程 $PID 已强制停止"
    fi
done

echo "[INFO] Embedding API 服务已停止"






