"""
RAG 系统配置文件
支持从 .env 文件加载配置
"""
import os
from pathlib import Path

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    # 从项目根目录加载 .env 文件
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[INFO] 已加载 .env 文件: {env_path}")
except ImportError:
    # 如果没有安装 python-dotenv，跳过
    pass

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
RAG_DATA_DIR = DATA_DIR / "rag"
QA_DATA_DIR = DATA_DIR / "qa"

# 向量数据库目录（LightRAG工作目录）
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db" / "lightrag"

# vLLM 配置
# 单个vLLM服务地址（使用数据并行模式）
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

# vLLM 模型配置
# 优先顺序：
# 1. 环境变量 VLLM_MODEL_PATH（本地路径）
# 2. 环境变量 VLLM_MODEL（HuggingFace 模型名称或本地路径）
# 3. 本地模型路径 /kongweiwen/models/Qwen2.5-14B（如果存在）
# 4. HuggingFace 模型名称 Qwen/Qwen2.5-14B-Instruct（默认）
_default_vllm_local_path = Path("/kongweiwen/models/Qwen2.5-14B")

if os.getenv("VLLM_MODEL_PATH"):
    # 优先使用环境变量指定的本地路径
    VLLM_MODEL = os.getenv("VLLM_MODEL_PATH")
elif os.getenv("VLLM_MODEL"):
    # 使用环境变量指定的模型（可以是路径或模型名）
    VLLM_MODEL = os.getenv("VLLM_MODEL")
elif _default_vllm_local_path.exists():
    # 如果默认本地路径存在，使用本地路径
    VLLM_MODEL = str(_default_vllm_local_path)
else:
    # 默认使用 HuggingFace 模型名称
    VLLM_MODEL = "Qwen/Qwen2.5-14B-Instruct"

# DeepSeek API 配置（备用）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# RAG 分块配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "160"))

# 嵌入模型配置
# 支持本地模型路径或 HuggingFace 模型名称
# 如果指定本地路径，使用绝对路径，例如: "/path/to/local/model"
# 可用模型: 
#   - Qwen3-Embedding-8B (Qwen3 8B embedding模型)
#   - BAAI/bge-large-zh-v1.5 (中文，效果好，较大)
#   - BAAI/bge-m3 (多语言，支持中英文)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-zh-v1.5")

# 本地模型路径
# 优先顺序：
# 1. 环境变量 EMBEDDING_MODEL_PATH
# 2. 项目 models/ 目录中的模型（BAAI--bge-large-zh-v1.5）
# 3. HuggingFace 缓存目录中的模型
MODELS_DIR = PROJECT_ROOT / "models"
_default_local_model = MODELS_DIR / "BAAI--bge-large-zh-v1.5"
_default_cache_snapshot = os.path.expanduser(
    "~/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5"
)

if os.getenv("EMBEDDING_MODEL_PATH"):
    EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
elif _default_local_model.exists():
    EMBEDDING_MODEL_PATH = str(_default_local_model)
elif os.path.exists(_default_cache_snapshot):
    EMBEDDING_MODEL_PATH = _default_cache_snapshot
else:
    EMBEDDING_MODEL_PATH = None

# 检索配置（减少到 5 以避免上下文过长）
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))

# LLM 配置
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
# LLM模型名称配置
# 优先级：LLM_MODEL 环境变量 > VLLM_MODEL > 默认值
# 注意：如果使用 vLLM，模型名称应该是 vLLM API 返回的模型 ID（可能是路径形式）
_llm_model_env = os.getenv("LLM_MODEL", "")
if _llm_model_env:
    LLM_MODEL = _llm_model_env
elif VLLM_MODEL:
    # 使用 VLLM_MODEL（可能是路径或模型名称）
    LLM_MODEL = VLLM_MODEL
else:
    # 默认值
    LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct"

# LightRAG Worker 配置
# 控制并行处理的worker数量，可以加快索引构建速度
# 注意：增加worker数量会增加内存和CPU使用
# 通过环境变量 MAX_ASYNC 控制最大并发数（LightRAG内部使用）
LIGHTRAG_MAX_ASYNC = int(os.getenv("MAX_ASYNC", os.getenv("LIGHTRAG_MAX_ASYNC", "128")))  # 最大并发数（默认64）
# Worker超时配置（秒），控制单个任务的最大执行时间
# 如果任务超过此时间未完成，会抛出 WorkerTimeoutError
# 默认360秒，可以根据实际情况调整（如设置为120秒或180秒）
LIGHTRAG_WORKER_TIMEOUT = int(os.getenv("WORKER_TIMEOUT", os.getenv("LIGHTRAG_WORKER_TIMEOUT", "60")))  # Worker超时时间（默认120秒）

# LightRAG 上下文长度配置
# 用于限制输入上下文长度，避免超过模型最大token限制
# Qwen2.5-14B 最大上下文长度为 32768 tokens
# 设置一个安全的上限，预留空间给输出和系统提示词
LIGHTRAG_MAX_CONTEXT_LEN = int(os.getenv("LIGHTRAG_MAX_CONTEXT_LEN", "30000"))  # 最大上下文长度（默认30000，预留2768给输出）
LIGHTRAG_CHUNK_SIZE = int(os.getenv("LIGHTRAG_CHUNK_SIZE", "1000"))  # LightRAG内部chunk大小（默认1000）
LIGHTRAG_SUMMARY_CONTEXT_SIZE = int(os.getenv("LIGHTRAG_SUMMARY_CONTEXT_SIZE", "10000"))  # 摘要上下文大小（默认10000）

# LightRAG 每类函数的并发上限（可通过环境变量覆盖）
# embedding 对应 LightRAG 构造参数 `embedding_func_max_async`，默认 8
LIGHTRAG_EMBEDDING_MAX_ASYNC = int(os.getenv("LIGHTRAG_EMBEDDING_MAX_ASYNC", "128"))
# llm 对应 LightRAG 构造参数 `llm_model_max_async`，默认 128
LIGHTRAG_LLM_MAX_ASYNC = int(os.getenv("LIGHTRAG_LLM_MAX_ASYNC", "128"))
# LightRAG 并行插入限制（对应 LightRAG 构造参数 `max_parallel_insert`）
LIGHTRAG_MAX_PARALLEL_INSERT = int(os.getenv("LIGHTRAG_MAX_PARALLEL_INSERT", "64"))

# LightRAG 索引构建配置
# 广度优先加载配置
BREADTH_FIRST_BATCH_SIZE = int(os.getenv("BREADTH_FIRST_BATCH_SIZE", "1"))  # 广度优先时，每次从每个文件提取的记录数（默认1条）

# 索引插入批量配置
# 根据并发数动态调整插入batch_size，提高多GPU利用率
# insert_batch_size = min(MAX_INSERT_BATCH_SIZE, max(MIN_INSERT_BATCH_SIZE, max_async // 2))
MIN_INSERT_BATCH_SIZE = int(os.getenv("MIN_INSERT_BATCH_SIZE", "16"))  # 最小插入批量大小（默认20）
MAX_INSERT_BATCH_SIZE = int(os.getenv("MAX_INSERT_BATCH_SIZE", "64"))  # 最大插入批量大小（默认50）
INSERT_BATCH_SIZE_DIVISOR = int(os.getenv("INSERT_BATCH_SIZE_DIVISOR", "1"))  # 批量大小除数（默认1，即 max_async // 1）

# 索引构建进度打印配置
INDEX_PROGRESS_INTERVAL = int(os.getenv("INDEX_PROGRESS_INTERVAL", "50"))  # 每处理多少个文档打印一次进度（默认50）

# Embedding 批量配置
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))  # Embedding批量大小，可根据GPU显存调整（默认32）

# Embedding 多GPU配置
# 是否使用多GPU进行embedding（在不同GPU上加载多个模型实例）
EMBEDDING_USE_MULTI_GPU = os.getenv("EMBEDDING_USE_MULTI_GPU", "true").lower() == "true"  # 默认启用多GPU
# 用于embedding的GPU列表（例如："0,1,2,3" 表示使用GPU 0,1,2,3）
# 如果为空，则自动使用前N个GPU（N由 EMBEDDING_GPU_COUNT 决定）
EMBEDDING_GPU_IDS = os.getenv("EMBEDDING_GPU_IDS", "0,1,2,3")  # 默认使用GPU 0,1,2,3（四张卡并行）
# 用于embedding的GPU数量（仅在 EMBEDDING_GPU_IDS 为空时使用）
EMBEDDING_GPU_COUNT = int(os.getenv("EMBEDDING_GPU_COUNT", "4"))  # 默认使用4个GPU

# RAG 查询上下文格式化配置
RAG_MAX_CHARS_PER_DOC = int(os.getenv("RAG_MAX_CHARS_PER_DOC", "2000"))  # 每个文档的最大字符数（默认2000）
RAG_MAX_TOTAL_CHARS = int(os.getenv("RAG_MAX_TOTAL_CHARS", "20000"))  # 所有文档的总最大字符数（默认20000）

# Tiktoken Tokenizer 配置
# 用于 LightRAG 的 tokenizer 模型名称
# 可选值: gpt-4o-mini, gpt-4, gpt-3.5-turbo 等
TIKTOKEN_MODEL_NAME = os.getenv("TIKTOKEN_MODEL_NAME", "gpt-4o-mini")
TIKTOKEN_MAX_RETRIES = int(os.getenv("TIKTOKEN_MAX_RETRIES", "1"))  # 最大重试次数

# 内容过滤配置
TITLE_EXCLUSION_KEYWORDS = [
    "内部文件",
    "学生信息汇总",
    "404 Not Found"
]

CONTENT_EXCLUSION_KEYWORDS = [
    "此页面不存在",
    "页面未找到",
    "404错误",
    "under construction",
    "- 任意 -",
    "-任意-",
    "正在建设中",
    "页面维护中",
    "暂无内容",
    "内容更新中"
]

NOISE_PATTERNS = [
    r"查看更多",
    r"更多新闻",
    r"更多公示",
    r"您的浏览器不支持video播放，请升级浏览器！",
    r"主题：.*?$",
    r"报告人：.*?\n",
    r"日期：.*?\n",
    r"[\s\r\t]+",
]

MIN_CONTENT_LENGTH = int(os.getenv("MIN_CONTENT_LENGTH", "50"))

