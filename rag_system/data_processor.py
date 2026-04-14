"""
RAG 数据加载模块
从 data/rag 目录中加载已处理好的 RAG 数据
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Generator

from .config import BREADTH_FIRST_BATCH_SIZE


class Document:
    """简单的 Document 类替代 LangChain 的 Document"""
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


def breadth_first_load_documents_with_progress(
    file_paths: List[Path],
    batch_size: int = None,
    file_line_offsets: Dict[str, int] = None,
    progress_callback = None
) -> Generator[Dict[str, Any], None, None]:
    """
    带进度跟踪的广度优先文档加载器
    
    Args:
        file_paths: JSONL 文件路径列表
        batch_size: 每次从每个文件提取的记录数
        file_line_offsets: 每个文件的起始行号（用于断点续传）
        progress_callback: 进度回调函数，每次读取一行后调用 callback(file_key, line_number)
    
    Yields:
        文档字典，格式: {"title": "...", "content": "...", "url": "..."}
    """
    if not file_paths:
        return
    
    if batch_size is None:
        batch_size = BREADTH_FIRST_BATCH_SIZE
    
    if file_line_offsets is None:
        file_line_offsets = {}
    
    file_handles = []
    file_iterators = []
    file_current_lines = {}
    
    try:
        for file_path in file_paths:
            try:
                f = open(file_path, 'r', encoding='utf-8')
                file_handles.append(f)
                
                file_key = str(file_path.resolve())
                line_offset = file_line_offsets.get(file_key, 0)
                file_current_lines[file_key] = line_offset
                
                # 跳过已处理的行
                for _ in range(line_offset):
                    try:
                        next(f)
                    except StopIteration:
                        break
                
                file_iterators.append(iter(f))
            except FileNotFoundError:
                print(f"警告: 文件未找到 {file_path}，已跳过")
                file_handles.append(None)
                file_iterators.append(None)
        
        total_yielded = 0
        file_yielded_count = [0] * len(file_paths)
        
        while True:
            has_more_data = False
            
            for file_idx, (file_path, file_iter) in enumerate(zip(file_paths, file_iterators)):
                if file_iter is None:
                    continue
                
                file_key = str(file_path.resolve())
                extracted_count = 0
                
                while extracted_count < batch_size:
                    try:
                        line = next(file_iter)
                        file_current_lines[file_key] += 1
                        
                        # 调用进度回调
                        if progress_callback:
                            progress_callback(file_key, file_current_lines[file_key])
                        
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            item = json.loads(line)
                            content = item.get("content", "")
                            if not content:
                                continue
                            
                            doc = {
                                "title": item.get("title", "未知标题"),
                                "content": content,
                                "url": item.get("url", "")
                            }
                            
                            yield doc
                            total_yielded += 1
                            file_yielded_count[file_idx] += 1
                            extracted_count += 1
                            has_more_data = True
                            
                        except json.JSONDecodeError:
                            continue
                            
                    except StopIteration:
                        if file_handles[file_idx]:
                            file_handles[file_idx].close()
                            file_handles[file_idx] = None
                            file_iterators[file_idx] = None
                        break
                    except Exception as e:
                        print(f"警告: 处理文件 {file_path} 时发生错误: {e}")
                        continue
            
            if not has_more_data:
                break
        
        print(f"[INFO] 广度优先加载完成，共加载 {total_yielded} 个文档")
        for file_path, count in zip(file_paths, file_yielded_count):
            if count > 0:
                print(f"  - {file_path.name}: {count} 个文档")
    
    finally:
        for f in file_handles:
            if f:
                try:
                    f.close()
                except:
                    pass


def breadth_first_load_documents(
    file_paths: List[Path], 
    batch_size: int = None,
    file_line_offsets: Dict[str, int] = None
) -> Generator[Dict[str, Any], None, None]:
    """
    广度优先方式从多个 JSONL 文件中轮流加载文档。
    从每个文件轮流提取 batch_size 条记录，循环直到所有文件都处理完。
    这样可以避免连续处理同一主题的内容，提高索引构建效率。
    
    Args:
        file_paths: JSONL 文件路径列表
        batch_size: 每次从每个文件提取的记录数（None则使用配置中的默认值）
        file_line_offsets: 每个文件的起始行号（用于断点续传），格式: {文件路径字符串: 行号}
    
    Yields:
        文档字典，格式: {"title": "...", "content": "...", "url": "..."}
    """
    if not file_paths:
        return
    
    # 使用配置的默认值
    if batch_size is None:
        batch_size = BREADTH_FIRST_BATCH_SIZE
    
    if file_line_offsets is None:
        file_line_offsets = {}
    
    # 为每个文件创建文件句柄和行迭代器
    file_handles = []
    file_iterators = []
    
    try:
        # 打开所有文件，并跳过已处理的行
        for file_path in file_paths:
            try:
                f = open(file_path, 'r', encoding='utf-8')
                file_handles.append(f)
                
                # 获取文件的绝对路径作为key
                file_key = str(file_path.resolve())
                line_offset = file_line_offsets.get(file_key, 0)
                
                # 跳过已处理的行
                for _ in range(line_offset):
                    try:
                        next(f)
                    except StopIteration:
                        break
                
                file_iterators.append(iter(f))
            except FileNotFoundError:
                print(f"警告: 文件未找到 {file_path}，已跳过")
                file_handles.append(None)
                file_iterators.append(None)
        
        # 统计信息
        total_yielded = 0
        file_yielded_count = [0] * len(file_paths)
        
        # 广度优先循环：从每个文件轮流提取 batch_size 条记录
        while True:
            has_more_data = False
            
            # 遍历每个文件
            for file_idx, (file_path, file_iter) in enumerate(zip(file_paths, file_iterators)):
                if file_iter is None:
                    continue
                
                file_key = str(file_path.resolve())
                
                # 从当前文件提取 batch_size 条记录
                extracted_count = 0
                while extracted_count < batch_size:
                    try:
                        line = next(file_iter)
                        line = line.strip()
                        if not line:
                            # 空行，继续读取下一行
                            continue
                        
                        # 解析 JSON
                        try:
                            item = json.loads(line)
                            
                            # 提取内容
                            content = item.get("content", "")
                            if not content:
                                # 内容为空，继续读取下一行
                                continue
                            
                            # 构建标准格式的文档字典：{"title": "...", "content": "...", "url": "..."}
                            doc = {
                                "title": item.get("title", "未知标题"),
                                "content": content,
                                "url": item.get("url", "")
                            }
                            
                            yield doc
                            total_yielded += 1
                            file_yielded_count[file_idx] += 1
                            extracted_count += 1
                            has_more_data = True
                            
                        except json.JSONDecodeError:
                            # JSON解析错误，继续读取下一行
                            continue
                            
                    except StopIteration:
                        # 当前文件已读完，关闭文件句柄
                        if file_handles[file_idx]:
                            file_handles[file_idx].close()
                            file_handles[file_idx] = None
                            file_iterators[file_idx] = None
                        break
                    except Exception as e:
                        print(f"警告: 处理文件 {file_path} 时发生错误: {e}")
                        # 发生错误，继续读取下一行
                        continue
            
            # 如果所有文件都没有更多数据，退出循环
            if not has_more_data:
                break
        
        # 打印统计信息
        print(f"[INFO] 广度优先加载完成，共加载 {total_yielded} 个文档")
        for file_path, count in zip(file_paths, file_yielded_count):
            if count > 0:
                print(f"  - {file_path.name}: {count} 个文档")
    
    finally:
        # 确保所有文件句柄都被关闭
        for f in file_handles:
            if f:
                try:
                    f.close()
                except:
                    pass

