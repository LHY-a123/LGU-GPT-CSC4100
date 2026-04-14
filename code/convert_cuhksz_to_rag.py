#!/usr/bin/env python3
"""
将 data/cleaned/cuhksz_courses.jsonl 转换为 RAG 格式
- title: code + title (合并)
- content: 其他所有字段组合
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm


def convert_to_rag_format(course: Dict[str, Any]) -> Dict[str, Any]:
    """
    将课程数据转换为 RAG 格式
    
    Args:
        course: 原始课程数据字典
        
    Returns:
        RAG 格式的文档字典，包含 title 和 content
    """
    # 合并 code 和 title 作为 title
    code = course.get("code", "").strip()
    title = course.get("title", "").strip()
    
    if code and title:
        final_title = f"{code} {title}"
    elif code:
        final_title = code
    elif title:
        final_title = title
    else:
        final_title = "未知课程"
    
    # 收集其他所有字段作为 content
    content_parts = []
    
    # 按顺序添加字段
    if course.get("level"):
        content_parts.append(f"学历层次：{course['level']}")
    
    if course.get("school"):
        content_parts.append(f"所属学院：{course['school']}")
    
    if course.get("credits"):
        content_parts.append(f"学分：{course['credits']}")
    
    if course.get("grading"):
        content_parts.append(f"评分方式：{course['grading']}")
    
    if course.get("teaching_mode"):
        content_parts.append(f"教学方式：{course['teaching_mode']}")
    
    if course.get("term"):
        content_parts.append(f"开课学期：{course['term']}")
    
    if course.get("category"):
        content_parts.append(f"课程类别：{course['category']}")
    
    if course.get("url"):
        content_parts.append(f"课程链接：{course['url']}")
    
    if course.get("description"):
        content_parts.append(f"\n课程描述：\n{course['description']}")
    
    # 如果原数据中有 content 字段，也添加进去
    if course.get("content"):
        content_parts.append(f"\n{course['content']}")
    
    # 组合所有内容
    final_content = "\n".join(content_parts)
    
    return {
        "title": final_title,
        "content": final_content
    }


def convert_file(input_path: Path, output_path: Path) -> None:
    """
    转换整个 JSONL 文件
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    """
    print(f"正在读取文件: {input_path}")
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_count = 0
    converted_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        # 先统计总行数（用于进度条）
        lines = f_in.readlines()
        total_count = len(lines)
        
        # 重新打开文件处理
        f_in.seek(0)
        
        for line in tqdm(lines, desc="转换课程", unit="门"):
            line = line.strip()
            if not line:
                continue
            
            try:
                course = json.loads(line)
                rag_doc = convert_to_rag_format(course)
                
                # 只保存有内容的文档
                if rag_doc["content"].strip():
                    f_out.write(json.dumps(rag_doc, ensure_ascii=False) + "\n")
                    converted_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"\n警告: JSON 解析错误: {e}")
                continue
            except Exception as e:
                print(f"\n警告: 处理错误: {e}")
                continue
    
    print(f"\n转换完成！")
    print(f"总记录数: {total_count}")
    print(f"成功转换: {converted_count}")
    print(f"输出文件: {output_path}")


def main():
    """主函数"""
    # 设置路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    input_file = project_root / "data" / "cleaned" / "cuhksz_courses.jsonl"
    output_file = project_root / "data" / "rag" / "cuhksz_courses.jsonl"
    
    # 检查输入文件是否存在
    if not input_file.exists():
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    # 执行转换
    convert_file(input_file, output_file)
    print(f"\n✅ RAG 格式文件已保存到: {output_file}")


if __name__ == "__main__":
    main()

