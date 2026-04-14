#!/bin/bash
# 停止 LightRAG Server 服务的 Shell 脚本

echo "=========================================="
echo "停止 LightRAG Server 服务"
echo "=========================================="

# 查找所有相关进程
# 1. lightrag-server 主进程
# 2. start_lightrag_server.py 启动脚本进程
# 3. start_lightrag_server.sh shell 脚本进程

PIDS=$(ps aux | grep -E "lightrag-server|start_lightrag_server" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "[INFO] LightRAG Server 服务未运行"
    exit 0
fi

echo "[INFO] 找到以下进程:"
ps aux | grep -E "lightrag-server|start_lightrag_server" | grep -v grep | awk '{print "  PID: "$2" - "$11" "$12" "$13" "$14}'
echo ""

# 首先停止主进程（lightrag-server），然后停止启动脚本
# 按进程名称排序，先停止子进程

# 停止所有进程
for PID in $PIDS; do
    # 获取进程名称
    PROCESS_NAME=$(ps -p $PID -o comm= 2>/dev/null || echo "")
    
    if [ -z "$PROCESS_NAME" ]; then
        continue  # 进程可能已经不存在了
    fi
    
    echo "[INFO] 正在停止进程 $PID ($PROCESS_NAME)..."
    kill $PID 2>/dev/null || true
    
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
        kill -9 $PID 2>/dev/null || true
        sleep 0.5
        if ! kill -0 $PID 2>/dev/null; then
            echo "[INFO] 进程 $PID 已强制停止"
        fi
    fi
done

# 再次检查是否还有残留进程
REMAINING=$(ps aux | grep -E "lightrag-server|start_lightrag_server" | grep -v grep | awk '{print $2}')

if [ -z "$REMAINING" ]; then
    echo ""
    echo "[INFO] ✓ LightRAG Server 服务已完全停止"
else
    echo ""
    echo "[WARN] 以下进程仍在运行:"
    ps aux | grep -E "lightrag-server|start_lightrag_server" | grep -v grep
    echo "[WARN] 请手动检查并停止这些进程"
fi
