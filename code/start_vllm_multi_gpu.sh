#!/bin/bash
# 启动 vLLM 服务，使用数据并行模式（单个服务，多个GPU）

# 默认配置
LOCAL_MODEL_PATH="/kongweiwen/models/Qwen2.5-14B"
API_PORT=8000

# 检测可用GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "[INFO] 检测到 $GPU_COUNT 个GPU"

# 使用GPU进行数据并行（默认使用2张GPU：0,1）
USE_GPUS=${USE_GPUS:-"0,1"}
GPU_ARRAY=($(echo "$USE_GPUS" | tr ',' ' '))
ACTUAL_GPU_COUNT=${#GPU_ARRAY[@]}

echo "[INFO] 将使用以下GPU启动单个vLLM服务: ${GPU_ARRAY[@]} (共 $ACTUAL_GPU_COUNT 张)"
echo "[INFO] 运行模式: 数据并行（所有GPU共享一个模型实例）"

# 数据并行配置
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.86}  # 数据并行模式，更高的内存使用率

# 检查模型路径是否存在
if [ ! -d "$LOCAL_MODEL_PATH" ]; then
    echo "[ERROR] Qwen模型路径不存在: $LOCAL_MODEL_PATH"
    echo "请检查模型路径是否正确"
    exit 1
fi

echo "========================================="
echo "启动 vLLM 服务（数据并行模式）"
echo "单个服务，多个GPU共享"
echo "========================================="
echo "Qwen模型路径: $LOCAL_MODEL_PATH"
echo "使用GPU: ${GPU_ARRAY[@]}"
echo "运行模式: 数据并行（PP=1, DP=${ACTUAL_GPU_COUNT}）"
echo "GPU 内存使用率: $GPU_MEMORY_UTIL"
echo "API 端口: $API_PORT"
echo "========================================="
echo ""

# 显示GPU信息
echo "[INFO] GPU 信息:"
for gpu_id in ${GPU_ARRAY[@]}; do
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader --id=$gpu_id
done
echo ""

# 最大上下文长度配置
MAX_MODEL_LEN=${MAX_MODEL_LEN:-15000}

# 验证模型文件是否完整
echo "[INFO] 验证Qwen模型文件..."
if [ ! -f "$LOCAL_MODEL_PATH/config.json" ]; then
    echo "[ERROR] 模型配置文件不存在: $LOCAL_MODEL_PATH/config.json"
    exit 1
fi

if [ ! -f "$LOCAL_MODEL_PATH/model.safetensors.index.json" ]; then
    echo "[ERROR] 模型索引文件不存在: $LOCAL_MODEL_PATH/model.safetensors.index.json"
    exit 1
fi

SAFETENSORS_COUNT=$(ls -1 "$LOCAL_MODEL_PATH"/*.safetensors 2>/dev/null | wc -l)
if [ "$SAFETENSORS_COUNT" -eq 0 ]; then
    echo "[ERROR] 未找到模型权重文件 (*.safetensors)"
    exit 1
fi
echo "[INFO] 找到 $SAFETENSORS_COUNT 个Qwen模型权重文件"

# 强制使用本地模型 - 设置环境变量禁用网络下载
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 设置 PyTorch 并行线程数
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
echo "[INFO] 设置并行线程数: OMP_NUM_THREADS=$OMP_NUM_THREADS"

# 确保使用绝对路径
LOCAL_MODEL_PATH=$(realpath "$LOCAL_MODEL_PATH")
echo "[INFO] Qwen模型绝对路径: $LOCAL_MODEL_PATH"
echo "[INFO] 强制本地模式：已禁用 Hugging Face 网络下载"
echo ""

# 获取脚本目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================="
echo "开始启动 vLLM 服务..."
echo "========================================="
echo ""

# 设置CUDA_VISIBLE_DEVICES为指定的GPU列表
CUDA_DEVICES=$(IFS=','; echo "${GPU_ARRAY[*]}")
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# 启动单个vLLM服务，使用数据并行
echo "[INFO] 启动vLLM服务（端口: $API_PORT）..."
echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_DEVICES"

nohup python -m vllm.entrypoints.openai.api_server \
  --model "$LOCAL_MODEL_PATH" \
  --port "$API_PORT" \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size "$ACTUAL_GPU_COUNT" \
  --distributed-executor-backend ray \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --host 0.0.0.0 \
  --trust-remote-code \
  > "vllm.log" 2>&1 &

VLLM_PID=$!
echo "$VLLM_PID" > "$PROJECT_ROOT/vllm.pid"

echo "[INFO] vLLM 服务已启动，PID: $VLLM_PID"
echo "[INFO] 日志文件: vllm.log"
echo ""

# 等待服务启动
echo "[INFO] 等待服务启动..."
sleep 5

# 检查服务是否成功启动
if ps -p $VLLM_PID > /dev/null; then
    echo "========================================="
    echo "vLLM 服务启动成功！"
    echo "========================================="
    echo ""
    echo "API信息:"
    echo "  - 地址: http://0.0.0.0:$API_PORT/v1"
    echo "  - 使用GPU: ${GPU_ARRAY[@]} (共 $ACTUAL_GPU_COUNT 张)"
    echo "  - 模式: 数据并行"
    echo ""
    echo "常用命令:"
    echo "  - 查看日志: tail -f vllm.log"
    echo "  - 停止服务: kill $VLLM_PID"
    echo "  - 查看进程: ps aux | grep vllm"
else
    echo "[ERROR] vLLM 服务启动失败"
    echo "请查看日志文件: vllm.log"
    exit 1
fi
