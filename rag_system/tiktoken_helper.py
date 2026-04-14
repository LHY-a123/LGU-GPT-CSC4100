"""
Tiktoken Tokenizer 辅助模块
用于预加载 tiktoken tokenizer，提前下载所需的 BPE 文件
"""
import sys
import time
from typing import Optional


from .config import TIKTOKEN_MODEL_NAME, TIKTOKEN_MAX_RETRIES


def preload_tiktoken_tokenizer(
    model_name: str = None,
    max_retries: int = None
) -> None:
    """
    预加载 tiktoken tokenizer，提前下载所需的 BPE 文件
    
    Args:
        model_name: tokenizer 模型名称（None则使用配置中的默认值）
        max_retries: 最大重试次数（None则使用配置中的默认值）
    
    Raises:
        Exception: 如果加载失败且超过最大重试次数
    """
    if model_name is None:
        model_name = TIKTOKEN_MODEL_NAME
    if max_retries is None:
        max_retries = TIKTOKEN_MAX_RETRIES
    
    try:
        import tiktoken
    except ImportError:
        raise ImportError(
            "tiktoken 模块未安装，请运行: pip install tiktoken"
        )
    
    print(f"[INFO] 开始预加载 tiktoken tokenizer: {model_name}")
    
    for attempt in range(1, max_retries):
        try:
            # 尝试加载 tokenizer
            # 这会自动下载所需的 BPE 文件（如果尚未缓存）
            encoding = tiktoken.encoding_for_model(model_name)
            
            # 测试 tokenizer 是否正常工作
            test_text = "测试文本"
            tokens = encoding.encode(test_text)
            decoded = encoding.decode(tokens)
            
            if decoded == test_text:
                print(f"[INFO] tiktoken tokenizer 预加载成功 (模型: {model_name})")
                print(f"[INFO] 测试编码/解码正常，token 数量: {len(tokens)}")
                return
            else:
                raise ValueError(f"Tokenizer 解码测试失败: 原始文本 '{test_text}' != 解码文本 '{decoded}'")
                
        except Exception as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt  # 指数退避
                print(f"[WARN] 尝试 {attempt}/{max_retries} 失败: {e}")
                print(f"[INFO] {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"[ERROR] 预加载 tiktoken tokenizer 失败 (尝试 {max_retries} 次)")
                raise Exception(
                    f"无法加载 tiktoken tokenizer '{model_name}': {e}\n"
                    f"请确保:\n"
                    f"  1. 已安装 tiktoken: pip install tiktoken\n"
                    f"  2. 网络连接正常（首次使用需要下载 BPE 文件）\n"
                    f"  3. 模型名称正确: {model_name}"
                )


def main():
    """
    命令行入口：可以作为模块直接运行
    用法: python -m rag_system.tiktoken_helper <model_name>
    """
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "gpt-4o-mini"
    
    try:
        preload_tiktoken_tokenizer(model_name=model_name, max_retries=5)
        print("[INFO] 完成！")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] 失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

