#!/usr/bin/env python3
"""
手动下载嵌入模型脚本（使用镜像源）
"""
import os
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

def download_model(model_name: str = "BAAI/bge-large-zh-v1.5", use_mirror: bool = True):
    """下载嵌入模型到项目 models/ 目录"""
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    # 配置镜像源
    if use_mirror:
        # 使用 HuggingFace 镜像源
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("✓ 已配置 HuggingFace 镜像源: https://hf-mirror.com")
    else:
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
        print("✓ 使用原始 HuggingFace 源")
    
    # 增加超时时间
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5分钟超时
    
    print(f"开始下载模型: {model_name}")
    print(f"模型将保存到: {models_dir}")
    print("这可能需要一些时间，请耐心等待...")
    print("")
    
    try:
        print("正在下载模型...")
        if use_mirror:
            print("提示: 如果镜像源连接超时，可以设置 use_mirror=False")
        print("")
        
        # 下载模型到指定目录
        model = SentenceTransformer(model_name, cache_folder=str(models_dir))
        # 保存模型到指定目录
        model_path = models_dir / model_name.replace("/", "--")
        model.save(str(model_path))
        print(f"✓ 模型下载成功！")
        print(f"模型已保存到: {model_path}")
        print(f"\n提示: 系统会自动检测 models/ 目录中的模型")
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ 下载失败: {error_msg}")
        print("\n可能的解决方案:")
        print("1. 如果镜像源超时，尝试使用原始源:")
        print("   python3 scripts/download_embedding_model.py --no-mirror")
        print("2. 检查网络连接")
        print("3. 尝试使用更小的模型:")
        print("   python3 scripts/download_embedding_model.py BAAI/bge-m3")
        print("4. 手动设置 HuggingFace 镜像源:")
        print("   export HF_ENDPOINT=https://hf-mirror.com")
        return False

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-large-zh-v1.5"
    use_mirror = "--no-mirror" not in sys.argv
    success = download_model(model_name, use_mirror)
    sys.exit(0 if success else 1)
