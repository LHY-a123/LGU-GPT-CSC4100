"""
LightRAG LLM函数包装器
支持vLLM OpenAI兼容接口
"""
import asyncio
import json
from typing import List, Dict, Any, Optional
from urllib.request import urlopen
from urllib.error import URLError
from openai import OpenAI

from .config import VLLM_BASE_URL, VLLM_MODEL, LIGHTRAG_MAX_CONTEXT_LEN, LLM_TEMPERATURE


def create_vllm_complete_func(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: str = "empty"
):
    """
    创建vLLM complete函数，兼容LightRAG的LLM接口
    
    Args:
        base_url: vLLM API base URL（默认使用配置中的地址）
        model: 模型名称（如果为None，会尝试从vLLM服务自动检测）
        api_key: API key（vLLM通常不需要真实的key）
    
    Returns:
        async complete函数
    """
    # 使用提供的base_url或配置中的地址
    if not base_url:
        base_url = VLLM_BASE_URL
    
    print(f"[INFO] vLLM 服务地址: {base_url}")
    
    # 创建OpenAI客户端
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # 如果没有指定模型，尝试从vLLM服务自动检测
    if model is None:
        model = VLLM_MODEL
        # 尝试从vLLM服务获取实际模型名称
        try:
            models_url = base_url.replace("/v1", "") + "/v1/models"
            with urlopen(models_url, timeout=5) as response:
                models_data = json.loads(response.read().decode())
                if models_data.get("data") and len(models_data["data"]) > 0:
                    detected_model = models_data["data"][0].get("id")
                    if detected_model:
                        print(f"[INFO] 从vLLM服务检测到模型: {detected_model}")
                        model = detected_model
        except (URLError, json.JSONDecodeError, KeyError, Exception) as e:
            print(f"[WARN] 无法从vLLM服务检测模型名称，使用配置的模型名称: {model}")
    
    async def complete_func(
        messages,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        异步complete函数，调用vLLM生成回复
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
                    如果传入字符串，会转换为消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数（会过滤掉vLLM不支持的参数）
        
        Returns:
            LLM生成的文本
        """
        try:
            # 验证和转换messages参数
            if isinstance(messages, str):
                # 如果messages是字符串，转换为消息列表
                messages = [{"role": "user", "content": messages}]
            elif not isinstance(messages, list):
                raise ValueError(f"messages必须是列表或字符串，当前类型: {type(messages)}")
            
            # 确保messages中的每个元素都是字典
            validated_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    validated_messages.append(msg)
                elif isinstance(msg, str):
                    validated_messages.append({"role": "user", "content": msg})
                else:
                    raise ValueError(f"消息格式错误: {type(msg)}")
            
            # 过滤掉vLLM不支持的参数
            unsupported_params = {
                'hashing_kv', 
                'system_prompt', 
                'history_messages',
                'extra_body', 
                'extra_headers',
                'response_format',
            }
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in unsupported_params}
            
            # 处理system_prompt：如果存在，将其添加到messages的开头
            if 'system_prompt' in kwargs and kwargs['system_prompt']:
                system_msg = {"role": "system", "content": kwargs['system_prompt']}
                validated_messages = [system_msg] + validated_messages
            
            # 处理history_messages：如果存在，将其合并到messages中
            if 'history_messages' in kwargs and kwargs['history_messages']:
                history = kwargs['history_messages']
                if isinstance(history, list):
                    if validated_messages and validated_messages[0].get('role') == 'system':
                        validated_messages = [validated_messages[0]] + history + validated_messages[1:]
                    else:
                        validated_messages = history + validated_messages
            
            # 使用配置的temperature（如果未提供）
            if temperature is None:
                temperature = LLM_TEMPERATURE
            
            # 检查输入长度
            total_chars = sum(len(str(msg.get('content', ''))) for msg in validated_messages)
            estimated_tokens = total_chars // 2
            
            if estimated_tokens > LIGHTRAG_MAX_CONTEXT_LEN:
                print(f"[WARN] 输入长度可能超过限制（估算: {estimated_tokens} tokens，限制: {LIGHTRAG_MAX_CONTEXT_LEN} tokens）")
                print(f"[WARN] 尝试截断最长的消息...")
                
                max_len = LIGHTRAG_MAX_CONTEXT_LEN * 2 // len(validated_messages) * 2
                for msg in validated_messages:
                    content = msg.get('content', '')
                    if len(content) > max_len:
                        msg['content'] = content[:max_len] + "\n\n[内容已截断...]"
                        print(f"[WARN] 已截断消息，原长度: {len(content)}, 新长度: {len(msg['content'])}")
            
            # 使用asyncio.to_thread在异步函数中调用同步API
            def _call_api():
                response = client.chat.completions.create(
                    model=model,
                    messages=validated_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **filtered_kwargs
                )
                return response.choices[0].message.content.strip()
            
            result = await asyncio.to_thread(_call_api)
            return result
            
        except Exception as e:
            error_str = str(e)
            if "maximum context length" in error_str or "context length" in error_str.lower():
                print(f"[ERROR] 上下文长度超限: {error_str}")
                print(f"[INFO] 建议:")
                print(f"[INFO]   1. 减小文档块大小")
                print(f"[INFO]   2. 增加 vLLM 的 max_model_len")
                print(f"[INFO]   3. 设置环境变量 LIGHTRAG_MAX_CONTEXT_LEN 为更小的值")
            raise RuntimeError(f"vLLM调用失败: {e}")
    
    return complete_func


def create_vllm_complete_func_sync(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: str = "empty"
):
    """
    创建同步版本的vLLM complete函数（用于非异步场景）
    
    Args:
        base_url: vLLM API base URL（默认使用配置中的地址）
        model: 模型名称（如果为None，会尝试从vLLM服务自动检测）
        api_key: API key
    
    Returns:
        sync complete函数
    """
    # 使用提供的base_url或配置中的地址
    if not base_url:
        base_url = VLLM_BASE_URL
    
    print(f"[INFO] vLLM 服务地址: {base_url}")
    
    # 创建OpenAI客户端
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # 如果没有指定模型，尝试从vLLM服务自动检测
    if model is None:
        model = VLLM_MODEL
        # 尝试从vLLM服务获取实际模型名称
        try:
            models_url = base_url.replace("/v1", "") + "/v1/models"
            with urlopen(models_url, timeout=5) as response:
                models_data = json.loads(response.read().decode())
                if models_data.get("data") and len(models_data["data"]) > 0:
                    detected_model = models_data["data"][0].get("id")
                    if detected_model:
                        print(f"[INFO] 从vLLM服务检测到模型: {detected_model}")
                        model = detected_model
        except (URLError, json.JSONDecodeError, KeyError, Exception) as e:
            print(f"[WARN] 无法从vLLM服务检测模型名称，使用配置的模型名称: {model}")
    
    def complete_func(
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        同步complete函数，调用vLLM生成回复
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数（会过滤掉vLLM不支持的参数）
        
        Returns:
            LLM生成的文本
        """
        try:
            # 使用配置的temperature（如果未提供）
            if temperature is None:
                temperature = LLM_TEMPERATURE
            
            # 过滤掉vLLM不支持的参数
            unsupported_params = {
                'hashing_kv', 
                'system_prompt', 
                'history_messages',
                'extra_body', 
                'extra_headers',
                'response_format',
            }
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in unsupported_params}
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **filtered_kwargs
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_str = str(e)
            if "maximum context length" in error_str or "context length" in error_str.lower():
                print(f"[ERROR] 上下文长度超限: {error_str}")
                print(f"[INFO] 建议:")
                print(f"[INFO]   1. 减小文档块大小")
                print(f"[INFO]   2. 增加 vLLM 的 max_model_len")
                print(f"[INFO]   3. 设置环境变量 LIGHTRAG_MAX_CONTEXT_LEN 为更小的值")
            raise RuntimeError(f"vLLM调用失败: {e}")
    
    return complete_func

