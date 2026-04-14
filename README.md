# CUSZ-GPT RAG 系统

香港中文大学（深圳）知识问答系统，基于 RAG（检索增强生成）技术。

## 目录

- [从头开始配置](#从头开始配置)
- [快速开始](#快速开始)
- [数据目录结构](#数据目录结构)
- [完整流程](#完整流程)
- [脚本命令说明](#脚本命令说明)
- [RAG 系统命令](#rag-系统命令)
- [配置说明](#配置说明)
- [爬虫根域列表](#爬虫根域列表)

## 从头开始配置

本指南将帮助您从零开始配置整个 CUSZ-GPT RAG 系统。

### 前置要求

#### 1. 系统要求

- **操作系统**：Linux（推荐 Ubuntu 20.04+）
- **Python**：3.8 或更高版本
- **GPU**：NVIDIA GPU（推荐显存 ≥ 16GB，用于运行 vLLM）
- **内存**：≥ 32GB RAM
- **磁盘空间**：≥ 100GB（用于存储模型和数据）

#### 2. 检查系统环境

```bash
# 检查 Python 版本
python3 --version  # 需要 >= 3.8

# 检查 GPU
nvidia-smi  # 确认 GPU 可用

# 检查 CUDA（vLLM 需要 CUDA 11.8+）
nvcc --version  # 或检查 /usr/local/cuda/version
```

### 步骤 1：克隆项目并创建虚拟环境

```bash
# 1.1 进入工作目录
cd /path/to/your/workspace

# 1.2 克隆项目（如果从 Git 仓库）
# git clone <repository-url> CUSZ-GPT
# cd CUSZ-GPT

# 1.3 创建 Python 虚拟环境
python3 -m venv venv
source venv/bin/activate

# 1.4 升级 pip
pip install --upgrade pip setuptools wheel
```

### 步骤 2：安装依赖

```bash
# 2.1 安装基础依赖
pip install -r requirements.txt

# 2.2 安装 vLLM（需要 CUDA 支持）
# 注意：vLLM 需要 CUDA 11.8 或 12.1
pip install vllm

# 如果安装失败，可以尝试指定 CUDA 版本：
# pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
```

> **提示**：如果遇到 vLLM 安装问题，请参考 [vLLM 官方文档](https://docs.vllm.ai/en/latest/getting_started/installation.html)

### 步骤 3：配置环境变量

创建或编辑 `~/.bashrc` 或 `~/.zshrc`，添加以下环境变量：

```bash
# HuggingFace 镜像源（加速模型下载）
export HF_ENDPOINT=https://hf-mirror.com

# vLLM 配置（可选，使用默认值时可省略）
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=Qwen/Qwen2.5-14B-Instruct

# 嵌入模型配置（可选）
export EMBEDDING_MODEL_NAME=BAAI/bge-large-zh-v1.5

# DeepSeek API 配置（用于生成 QA 数据集，可选）
export DEEPSEEK_API_KEY=your_api_key_here
export DEEPSEEK_API_KEYS=sk1,sk2,sk3  # 多个 API key，用逗号分隔
export DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# LightRAG 并发配置（可选）
export MAX_ASYNC=32  # 或 LIGHTRAG_MAX_ASYNC=32
```

使环境变量生效：

```bash
source ~/.bashrc  # 或 source ~/.zshrc
```

### 步骤 4：下载模型

#### 4.1 下载嵌入模型（BGE）

嵌入模型用于将文本转换为向量，支持语义检索。

```bash
# 使用脚本自动下载（推荐）
# 模型将下载到 CUSZ-GPT/models/ 目录
bash scripts/download_embedding_model.sh

# 或手动指定模型
export EMBEDDING_MODEL_NAME=BAAI/bge-m3  # 多语言模型，体积更小
bash scripts/download_embedding_model.sh
```

**可用模型选项**：
- `BAAI/bge-large-zh-v1.5`（默认）：中文效果好，体积较大（~1.3GB）
- `BAAI/bge-m3`：多语言支持，体积适中（~600MB）

**模型存储位置**：
- 默认下载到：`CUSZ-GPT/models/` 目录
- 系统会自动检测并使用该目录中的模型
- 如需手动指定路径，可设置环境变量：`export EMBEDDING_MODEL_PATH=/path/to/model`

#### 4.2 下载 Qwen 模型（用于 vLLM）

Qwen 模型用于生成和优化文本内容。

```bash
# 配置镜像源（如果未在环境变量中设置）
export HF_ENDPOINT=https://hf-mirror.com

# 方式1：vLLM 会自动下载，但可以手动预下载
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-14B-Instruct')"

# 方式2：使用 huggingface-cli（如果已安装）
huggingface-cli download Qwen/Qwen2.5-14B-Instruct

# 方式3：直接使用 vLLM 启动时会自动下载（但首次启动较慢）
```

> **提示**：
> - 嵌入模型会下载到 `CUSZ-GPT/models/` 目录
> - Qwen2.5-14B-Instruct 模型约 28GB，会下载到 HuggingFace 缓存目录（`~/.cache/huggingface/hub/`）
> - 如果网络较慢，建议使用镜像源或手动下载

### 步骤 5：启动 vLLM 服务

vLLM 提供高性能的 LLM 推理服务，用于文本生成和优化。

```bash
# 5.1 启动 vLLM 服务（后台运行）
nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --port 8000 > vllm.log 2>&1 &

# 5.2 检查服务状态
# 等待 1-2 分钟让服务启动，然后检查：
curl http://localhost:8000/v1/models

# 或查看日志
tail -f vllm.log

# 5.3 如果服务启动失败，检查：
# - GPU 是否可用：nvidia-smi
# - CUDA 版本是否兼容：nvcc --version
# - 显存是否足够（至少需要 12GB）
```

**vLLM 启动参数说明**：
- `--model`：模型名称，必须与下载的模型一致
- `--tensor-parallel-size`：张量并行数，单 GPU 设为 1，多 GPU 可增加
- `--gpu-memory-utilization`：GPU 显存使用率（0.0-1.0），建议 0.9
- `--port`：服务端口，默认 8000

**多 GPU 配置示例**：
```bash
# 使用 2 个 GPU
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

### 步骤 6：准备数据目录

```bash
# 创建必要的数据目录
mkdir -p data/{raw,cleaned,rag,qa,vector_db}
```

### 步骤 7：数据采集与处理

#### 7.1 爬取网站数据

```bash
# 爬取单个网站
python3 scripts/crawl_website.py \
  --urls "['https://www.cuhk.edu.cn/zh-hans']" \
  --depth 4 \
  --max-pages 2000

# 爬取多个网站
python3 scripts/crawl_website.py \
  --urls "['https://www.cuhk.edu.cn/zh-hans','https://sse.cuhk.edu.cn']" \
  --depth 4 \
  --max-pages 2000

# 输出：data/raw/*.jsonl
```

#### 7.2 清洗原始数据

```bash
# 清洗单个文件
python3 scripts/clean_raw_dataset.py \
  --inputs data/raw/admissions_*.jsonl

# 清洗多个文件
python3 scripts/clean_raw_dataset.py \
  --inputs data/raw/*.jsonl

# 输出：data/cleaned/*.jsonl
```

#### 7.3 生成 RAG 数据（优化标题和内容）

**重要**：此步骤需要 vLLM 服务运行。

```bash
# 确保 vLLM 服务正在运行
curl http://localhost:8000/v1/models

# 优化单个文件
python3 scripts/improve_rag_titles.py \
  --input data/cleaned/admissions_cuhk_edu_cn.jsonl \
  --model Qwen/Qwen2.5-14B-Instruct \
  --max-workers 16

# 批量处理所有 cleaned 文件（推荐）
python3 scripts/batch_optimize_rag_titles.py

# 输出：data/rag/*.jsonl
```

### 步骤 8：构建向量索引

向量索引用于快速检索相关文档。

```bash
# 8.1 构建向量索引（使用 vLLM）
python3 -m rag_system.main build-index --use-vllm

# 8.2 检查索引是否构建成功
ls -lh data/vector_db/lightrag/

# 输出：data/vector_db/lightrag/ 目录下会生成多个文件
```

**构建索引说明**：
- 首次构建可能需要较长时间（取决于数据量）
- 索引构建完成后会自动保存，无需重复构建
- 如果数据更新，需要重新构建索引

### 步骤 9：测试 RAG 系统

```bash
# 9.1 使用测试脚本（推荐）
python3 scripts/test_rag.py --test-set comprehensive

# 9.2 单次查询测试
python3 -m rag_system.main query \
  --query "香港中文大学（深圳）有哪些学院？" \
  --show-context

# 9.3 交互式查询
python3 -m rag_system.main query --show-context
```

### 步骤 10：（可选）生成 QA 数据集

如果需要生成问答数据集用于评估或训练：

```bash
# 10.1 配置 DeepSeek API Key（如果未设置）
export DEEPSEEK_API_KEYS=sk1,sk2,sk3

# 10.2 生成 QA 数据集
python3 scripts/generate_qa_dataset.py \
  --input-files data/cleaned/admissions_cuhk_edu_cn.jsonl \
  --sample-size 200 \
  --qas-per-record 2

# 输出：data/qa/*.jsonl
```

### 验证安装

运行以下命令验证所有组件是否正常工作：

```bash
# 1. 检查 Python 依赖
python3 -c "import lightrag, sentence_transformers, vllm; print('✓ 依赖安装正常')"

# 2. 检查嵌入模型
python3 -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('BAAI/bge-large-zh-v1.5'); print('✓ 嵌入模型可用')"

# 3. 检查 vLLM 服务
curl http://localhost:8000/v1/models && echo "✓ vLLM 服务正常"

# 4. 检查向量索引
[ -d "data/vector_db/lightrag" ] && echo "✓ 向量索引已构建" || echo "⚠ 向量索引未构建"

# 5. 测试查询
python3 -m rag_system.main query --query "测试" --show-context
```

### 常见问题排查

#### 问题 1：vLLM 启动失败

**症状**：服务无法启动或报错

**解决方案**：
```bash
# 检查 GPU 和 CUDA
nvidia-smi
nvcc --version

# 检查 vLLM 版本
pip show vllm

# 尝试降低显存使用率
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --gpu-memory-utilization 0.7 \
  --port 8000
```

#### 问题 2：模型下载失败

**症状**：下载超时或网络错误

**解决方案**：
```bash
# 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载后指定路径
# 如果模型已下载到 CUSZ-GPT/models/ 目录，系统会自动检测
# 也可以手动指定：
export EMBEDDING_MODEL_PATH=/path/to/local/model
```

#### 问题 3：内存不足

**症状**：构建索引时 OOM（Out of Memory）

**解决方案**：
```bash
# 减少并发数
export MAX_ASYNC=16  # 默认是 32

# 或分批处理数据
```

#### 问题 4：查询结果不准确

**症状**：检索到的文档不相关

**解决方案**：
- 检查数据质量：确保 `data/rag/` 目录下的数据已优化
- 调整检索数量：修改 `rag_system/config.py` 中的 `RETRIEVER_TOP_K`
- 重新构建索引：删除 `data/vector_db/` 后重新构建

### 下一步

配置完成后，您可以：

1. **定期更新数据**：使用爬虫脚本定期爬取网站更新
2. **优化检索效果**：调整 `rag_system/config.py` 中的参数
3. **扩展功能**：参考 [脚本命令说明](#脚本命令说明) 使用更多工具
4. **部署服务**：参考 `scripts/start_lightrag_server.py` 启动 Web UI

---

## 快速开始

```bash
# 1. 下载嵌入模型
bash scripts/download_embedding_model.sh

# 2. 启动 vLLM 服务
nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --port 8000 > vllm.log 2>&1 &

# 3. 构建向量索引（如果已有 RAG 数据，使用LightRAG）
python3 -m rag_system.main build-index

# 4. 测试查询
python3 scripts/test_rag.py --test-set default
```

## 数据目录结构

```
data/
├── raw/              # 原始爬取数据
├── cleaned/          # 清洗后的数据
├── rag/              # RAG 格式数据（已优化标题和内容）
├── qa/               # QA 数据集
└── vector_db/        # 向量数据库（构建索引后生成）
```

## 完整流程

### 1. 模型下载

在开始使用 RAG 系统之前，需要先下载必要的模型：

**下载嵌入模型（BGE）**
```bash
# 使用镜像源加速下载
# 模型将下载到 CUSZ-GPT/models/ 目录
bash scripts/download_embedding_model.sh

# 或手动指定模型（可选）
export EMBEDDING_MODEL_NAME=BAAI/bge-m3  # 多语言模型
bash scripts/download_embedding_model.sh
```

**下载 Qwen 模型（用于 vLLM）**
```bash
# 配置镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 下载 Qwen 模型（vLLM 会自动下载，但也可以手动预下载）
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-14B-Instruct')"

# 或者使用 huggingface-cli
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct
```

> **提示**：如果网络较慢，建议配置 HuggingFace 镜像源 `export HF_ENDPOINT=https://hf-mirror.com`

### 2. 数据采集与清洗

```bash
# 2.1 爬取网站数据
python3 scripts/crawl_website.py \
  --urls "['https://www.cuhk.edu.cn/zh-hans','https://sse.cuhk.edu.cn']" \
  --depth 4 \
  --max-pages 2000

# 输出：data/raw/*.jsonl

# 2.2 清洗原始数据
python3 scripts/clean_raw_dataset.py \
  --inputs data/raw/admissions_*.jsonl

# 输出：data/cleaned/*.jsonl
```

### 3. 生成 RAG 数据

```bash
# 3.1 启动 vLLM 服务（如果未启动）
nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --port 8000 > vllm.log 2>&1 &

# 检查服务状态
curl http://localhost:8000/v1/models

# 3.2 优化标题和内容（从 cleaned 生成 rag）
python3 scripts/improve_rag_titles.py \
  --input data/cleaned/admissions_cuhk_edu_cn.jsonl \
  --model Qwen/Qwen2.5-14B-Instruct \
  --max-workers 16

# 输出：data/rag/admissions_cuhk_edu_cn.jsonl

# 3.3 批量处理所有 cleaned 文件
python3 scripts/batch_optimize_rag_titles.py
```

### 4. 构建向量索引

```bash
# 4.1 构建向量索引
python3 -m rag_system.main build-index --use-vllm

# 输出：data/vector_db/
```

### 5. 测试 RAG 系统

```bash
# 5.1 使用测试脚本
python3 scripts/test_rag.py --test-set comprehensive

# 5.2 使用命令行查询
python3 -m rag_system.main query \
  --query "香港中文大学（深圳）有哪些学院？" \
  --show-context

# 5.3 交互式查询
python3 -m rag_system.main query --show-context
```

## 脚本命令说明

| 脚本 | 作用 | 常用命令示例 |
| --- | --- | --- |
| `crawl_website.py` | 递归爬取根域，输出原始 HTML/JSONL | `python3 scripts/crawl_website.py --urls "['https://www.cuhk.edu.cn/zh-hans']" --depth 4 --max-pages 2000` |
| `clean_raw_dataset.py` | 对 `data/raw/*.jsonl` 做 rule-base 清洗，产生 `data/cleaned/*.jsonl` | `python3 scripts/clean_raw_dataset.py --inputs data/raw/admissions_*.jsonl` |
| `improve_rag_titles.py` | 从 cleaned 数据优化标题和内容，生成 RAG 数据 | `python3 scripts/improve_rag_titles.py --input data/cleaned/admissions_cuhk_edu_cn.jsonl --model Qwen/Qwen2.5-14B-Instruct --max-workers 16` |
| `batch_optimize_rag_titles.py` | 批量处理所有 cleaned 文件，生成 RAG 数据 | `python3 scripts/batch_optimize_rag_titles.py` |
| `download_embedding_model.sh` | 下载 BGE 嵌入模型（用于向量检索） | `bash scripts/download_embedding_model.sh` |
| `test_rag.py` | 测试 RAG 系统查询功能 | `python3 scripts/test_rag.py --test-set comprehensive --log-file rag_test.log` |
| `convert_courses_to_rag_qa.py` | 课程 JSON → 翻译（本地 vLLM）→ cleaned/RAG | `python3 scripts/convert_courses_to_rag_qa.py --input data/cuhksz_courses.json` |
| `generate_courses_qa.py` | cleaned 课程 → DeepSeek 并行 QA | `python3 scripts/generate_courses_qa.py --input data/cleaned/cuhksz_courses.jsonl` |
| `generate_qa_dataset.py` | 针对任意 cleaned JSONL 调用 DeepSeek 生成 QA（支持多文件/采样） | `python3 scripts/generate_qa_dataset.py --input-files data/cleaned/admissions_cuhk_edu_cn.jsonl --sample-size 200 --qas-per-record 2` |
| `process_study_schemes.py` | study_schemes PDF → 结构化 chunk JSONL | `python3 scripts/process_study_schemes.py --input-dir data/study_schemes --output data/study_schemes_processed/chunks.jsonl` |
| `parse_study_schemes_enhanced.py` | study_schemes PDF → 结构化课程信息（并行处理，每个PDF一个JSON文件） | `python3 scripts/parse_study_schemes_enhanced.py --input-dir data/study_schemes --output-dir data/study_schemes_processed --max-workers 8` |

> **提示**：
> - 默认 DeepSeek 并行脚本会读取 `DEEPSEEK_API_KEYS=sk1,sk2,...`；若未设置将使用内置 key
> - 若需切换数据源，只需调整命令中的 `--input` 路径，cleaned/RAG/QA 会落在匹配的子目录

## RAG 系统命令

| 命令 | 作用 | 示例 |
| --- | --- | --- |
| `build-index` | 构建向量索引 | `python3 -m rag_system.main build-index --use-vllm` |
| `query` | 执行查询（单次或交互式） | `python3 -m rag_system.main query --query "问题" --show-context` |
| `process` | 处理数据（清洗和分块） | `python3 -m rag_system.main process --input data/raw/file.jsonl` |
| `optimize` | 优化 RAG 数据 | `python3 -m rag_system.main optimize --input data/cleaned/file.jsonl` |

## 配置说明

- **嵌入模型**：默认使用 `BAAI/bge-large-zh-v1.5`，可通过 `EMBEDDING_MODEL_NAME` 环境变量修改
- **模型路径**：默认从 `CUSZ-GPT/models/` 目录加载，可通过 `EMBEDDING_MODEL_PATH` 环境变量修改
- **检索数量**：默认检索 5 个文档（`RETRIEVER_TOP_K=5`）
- **上下文限制**：每个文档最多 2000 字符，总长度最多 20000 字符
- **vLLM 服务**：默认地址 `http://localhost:8000/v1`
- **本地模式**：RAG 系统强制使用本地模型，请确保已下载嵌入模型到 `models/` 目录
- **向量索引**：构建后会自动保存，无需重复构建

## 爬虫根域列表

以下为需要定期爬取的门户/学院主站（按字母排序）。除明确标注外，默认使用 `https://` 协议。

- `https://www.cuhk.edu.cn/zh-hans`
- `https://sai.cuhk.edu.cn`
- `https://sds.cuhk.edu.cn`
- `https://sme.cuhk.edu.cn`
- `https://spp.cuhk.edu.cn`
- `https://sse.cuhk.edu.cn`
- `https://admissions.cuhk.edu.cn`
- `https://alumni.cuhk.edu.cn`
- `https://career.cuhk.edu.cn`
- `https://diligentia.cuhk.edu.cn`
- `https://gs.cuhk.edu.cn`
- `https://harmonia.cuhk.edu.cn`
- `https://hss.cuhk.edu.cn`
- `https://ling.cuhk.edu.cn`
- `https://lhs.cuhk.edu.cn`
- `https://med.cuhk.edu.cn`
- `https://minerva.cuhk.edu.cn`
- `https://muse.cuhk.edu.cn`
- `https://music.cuhk.edu.cn`
- `https://registry.cuhk.edu.cn`
- `https://shaw.cuhk.edu.cn`
- `https://library.cuhk.edu.cn/zh-hans`
- `https://tencentlab.cuhk.edu.cn`
- `https://foundation.cuhk.edu.cn`
- `https://osa.cuhk.edu.cn`
- `http://uac.cuhk.edu.cn`
- `https://shcc.cuhk.edu.cn`
- `https://oal.cuhk.edu.cn`
- `https://ccco.cuhk.edu.cn`
- `https://i.cuhk.edu.cn`
- `https://itso.cuhk.edu.cn`
- `http://itsm.cuhk.edu.cn`
- `https://ge.cuhk.edu.cn`
- `https://peu.cuhk.edu.cn`
- `https://www.cuhk.edu.hk/chinese`