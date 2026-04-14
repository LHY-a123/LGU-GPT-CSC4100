#!/usr/bin/env python3
"""
增强版 study scheme PDF 解析器
提取结构化信息：课程代码、学分、必修/选修、分流、毕业要求等
支持多种PDF格式，使用CPU并行处理
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pdfplumber
import zhconv


def to_simplified(text: str) -> str:
    """转换为简体中文"""
    if not text:
        return text
    return zhconv.convert(text, 'zh-cn')


TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "intersection_y_tolerance": 5,
    "intersection_x_tolerance": 5,
    "snap_tolerance": 3,
}


@dataclass
class Course:
    """课程信息"""
    code: str  # 课程代码，如 CSC1001
    name_en: Optional[str] = None
    name_zh: Optional[str] = None
    units: Optional[int] = None
    course_type: Optional[str] = None  # "required", "elective", "school_package"
    stream: Optional[str] = None  # 分流方向
    level: Optional[int] = None  # 1000, 2000, 3000, 4000
    term: Optional[str] = None  # 推荐学期
    notes: Optional[str] = None


@dataclass
class Stream:
    """分流方向"""
    name_en: Optional[str] = None
    name_zh: Optional[str] = None
    courses: List[Course] = None
    min_units: Optional[int] = None
    max_units: Optional[int] = None
    
    def __post_init__(self):
        if self.courses is None:
            self.courses = []


@dataclass
class StudyScheme:
    """学习方案"""
    programme_name_en: Optional[str] = None
    programme_name_zh: Optional[str] = None
    applicable_years: Optional[str] = None  # 适用年度
    programme_type: Optional[str] = None  # "major", "minor", "double_major"
    
    # 毕业要求
    total_units_required: Optional[int] = None
    school_package_units: Optional[int] = None
    required_courses_units: Optional[int] = None
    elective_courses_units: Optional[int] = None
    
    # 课程列表
    school_package_courses: List[Course] = None
    required_courses: List[Course] = None
    elective_courses: List[Course] = None
    
    # 分流
    streams: List[Stream] = None
    
    # 推荐课程模式
    recommended_pattern: Optional[str] = None
    
    # 其他信息
    notes: List[str] = None
    
    # 原始内容（供LLM后续处理）
    raw_text: Optional[str] = None  # 完整的PDF文本内容
    raw_tables_text: Optional[str] = None  # 表格文本内容
    
    def __post_init__(self):
        if self.school_package_courses is None:
            self.school_package_courses = []
        if self.required_courses is None:
            self.required_courses = []
        if self.elective_courses is None:
            self.elective_courses = []
        if self.streams is None:
            self.streams = []
        if self.notes is None:
            self.notes = []


def extract_course_codes(text: str) -> List[str]:
    """从文本中提取课程代码（如 CSC1001, MAT1002）"""
    # 匹配课程代码模式：2-4个大写字母 + 4位数字 + 可选字母后缀
    pattern = r'\b([A-Z]{2,4}\d{4}[A-Z]?)\b'
    codes = re.findall(pattern, text)
    return list(set(codes))  # 去重


def parse_course_list(text: str) -> List[Course]:
    """解析课程列表文本，提取课程代码"""
    courses = []
    codes = extract_course_codes(text)
    for code in codes:
        # 正确提取课程级别：CSC3100 -> 3000, CSC1001 -> 1000
        level_match = re.match(r'[A-Z]+(\d)(\d{3})', code)
        if level_match:
            level = int(level_match.group(1)) * 1000
        else:
            level = None
        
        course = Course(
            code=code,
            level=level
        )
        courses.append(course)
    return courses


def extract_units(text: str) -> Optional[int]:
    """从文本中提取学分数字"""
    # 查找 "Units" 或 "學分" 后面的数字
    patterns = [
        r'(?:Units|學分)[:\s]*(\d+)',
        r'(\d+)\s*(?:units|學分)',
        r'共[:\s]*(\d+)',
        r'Total[:\s]*(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def parse_table_for_courses(table: List[List], page_num: int) -> List[Course]:
    """从表格中解析课程信息 - 支持多种表格格式"""
    courses = []
    if not table or len(table) < 2:
        return courses
    
    # 尝试识别表头
    header = table[0]
    code_col = None
    name_en_col = None
    name_zh_col = None
    units_col = None
    
    for idx, col in enumerate(header):
        col_lower = str(col).lower() if col else ""
        if "code" in col_lower or "課程代碼" in col_lower or "代碼" in col_lower:
            code_col = idx
        elif "title" in col_lower and "english" in col_lower:
            name_en_col = idx
        elif ("title" in col_lower and "chinese" in col_lower) or "中文" in col_lower or "課程名稱" in col_lower:
            name_zh_col = idx
        elif "unit" in col_lower or "學分" in col_lower or "学分" in col_lower:
            units_col = idx
    
    # 如果没有找到标准表头，尝试其他格式
    if code_col is None:
        # 格式1: 第一列是课程代码
        if len(header) > 0 and header[0]:
            first_col = str(header[0]).strip()
            if re.match(r'^[A-Z]{2,4}\d{4}', first_col):
                code_col = 0
    
    # 解析数据行
    for row_idx, row in enumerate(table[1:], start=1):
        if not row:
            continue
        
        course_code = None
        name_en = None
        name_zh = None
        units = None
        
        # 尝试从第一列提取课程代码
        if code_col is not None and code_col < len(row):
            course_code = str(row[code_col]).strip() if row[code_col] else None
        elif len(row) > 0:
            # 尝试从第一列直接提取
            first_cell = str(row[0]).strip() if row[0] else ""
            code_match = re.search(r'\b([A-Z]{2,4}\d{4}[A-Z]?)\b', first_cell)
            if code_match:
                course_code = code_match.group(1)
        
        if course_code:
            # 清理课程代码（移除换行符等）
            course_code = re.sub(r'\s+', '', course_code)
            # 验证是否是有效的课程代码
            if not re.match(r'^[A-Z]{2,4}\d{4}[A-Z]?$', course_code):
                continue
        
        if name_en_col is not None and name_en_col < len(row):
            name_en = str(row[name_en_col]).strip() if row[name_en_col] else None
        
        if name_zh_col is not None and name_zh_col < len(row):
            name_zh = str(row[name_zh_col]).strip() if row[name_zh_col] else None
            if name_zh:
                name_zh = to_simplified(name_zh)  # 转换为简体
        
        if units_col is not None and units_col < len(row):
            units_str = str(row[units_col]).strip() if row[units_col] else None
            if units_str:
                units_match = re.search(r'(\d+)', units_str)
                if units_match:
                    units = int(units_match.group(1))
        
        if course_code:
            # 正确提取课程级别：CSC3100 -> 3000, CSC1001 -> 1000
            level_match = re.match(r'[A-Z]+(\d)(\d{3})', course_code)
            if level_match:
                level = int(level_match.group(1)) * 1000
            else:
                level = None
            
            course = Course(
                code=course_code,
                name_en=name_en,
                name_zh=name_zh,
                units=units,
                level=level
            )
            courses.append(course)
    
    return courses


def parse_text_for_requirements(text: str) -> Dict:
    """从文本中解析毕业要求 - 支持多种格式（文本应为简体）"""
    requirements = {}
    
    # 提取总学分 - 多种模式（简体关键词）
    total_patterns = [
        r'(?:Total|共|总计)[:\s]*(\d+)',
        r'(\d+)\s*(?:units|学分)\s*(?:total|总计|共)',
    ]
    for pattern in total_patterns:
        total_match = re.search(pattern, text, re.IGNORECASE)
        if total_match:
            requirements['total_units'] = int(total_match.group(1))
            break
    
    # 提取各部分学分 - 更灵活的匹配（简体关键词）
    patterns = [
        (r'(?:School Package|学院课程)[:\s]*(\d+)', 'school_package_units'),
        (r'(?:Required Courses|必修科目|必修课程)[:\s]*(\d+)', 'required_units'),
        (r'(?:Elective Courses|选修科目|选修课程)[:\s]*(\d+)', 'elective_units'),
        # 支持中文格式（简体）
        (r'(\d+)\s*(?:units|学分).*?(?:School Package|学院课程)', 'school_package_units'),
        (r'(\d+)\s*(?:units|学分).*?(?:Required|必修)', 'required_units'),
        (r'(\d+)\s*(?:units|学分).*?(?:Elective|选修)', 'elective_units'),
    ]
    
    for pattern, key in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            requirements[key] = int(match.group(1))
    
    return requirements


def _extract_streams_from_text(text: str, allow_multiline: bool = False) -> List[Stream]:
    """从文本中提取分流（内部辅助函数，支持单行或多行模式）"""
    streams = []
    flags = re.MULTILINE | (re.DOTALL if allow_multiline else 0)
    
    # 模式1: i) Stream Name Stream
    pattern1 = r'(?:^|\n)\s*(i{1,5}\)|v\))\s+([A-Z][A-Za-z\s]{3,40}?Stream)'
    for match in re.finditer(pattern1, text, flags | re.IGNORECASE):
        stream_name = match.group(2).strip()
        stream_name = re.sub(r'\s+', ' ', stream_name)
        words = stream_name.replace('Stream', '').strip().split()
        if 2 <= len(words) <= 6:
            if not any(s.name_en == stream_name for s in streams):
                streams.append(Stream(name_en=stream_name))
    
    # 模式2: i) 方向名称 方向
    pattern2 = r'(?:^|\n)\s*(i{1,5}\)|v\))\s+([^：:\n()]{2,15}?)(?:方向|专修范围)'
    for match in re.finditer(pattern2, text, flags):
        stream_name = to_simplified(match.group(2).strip())
        stream_name = re.sub(r'\s+', ' ', stream_name)
        if any('\u4e00' <= c <= '\u9fff' for c in stream_name):
            if 2 <= len(stream_name) <= 15:
                if not any(s.name_zh == stream_name for s in streams):
                    streams.append(Stream(name_zh=stream_name))
    
    # 模式3: (1) Stream Name; and (2) Stream Name
    pattern3 = r'\((\d+)\)\s+([A-Z][A-Za-z\s]{3,40}?)(?:;|and|$)'
    for match in re.finditer(pattern3, text, flags):
        stream_name = match.group(2).strip()
        stream_name = re.sub(r'\s+', ' ', stream_name)
        stream_name = re.sub(r'[;\s]+(and|及)$', '', stream_name, flags=re.IGNORECASE).strip()
        words = stream_name.split()
        if 2 <= len(words) <= 6:
            if not any(s.name_en == stream_name for s in streams):
                streams.append(Stream(name_en=stream_name))
    
    # 模式4: （1）方向名称；及 （2）方向名称
    pattern4 = r'（(\d+)）\s+([^：:\n()]{2,20}?)(?:；|;|及|and|$)'
    for match in re.finditer(pattern4, text, flags):
        stream_name = to_simplified(match.group(2).strip())
        stream_name = re.sub(r'\s+', ' ', stream_name)
        stream_name = re.sub(r'[；;\s]+(及|and)$', '', stream_name).strip()
        if any('\u4e00' <= c <= '\u9fff' for c in stream_name):
            if 2 <= len(stream_name) <= 15:
                if not any(s.name_zh == stream_name for s in streams):
                    streams.append(Stream(name_zh=stream_name))
    
    return streams


def extract_streams(text: str) -> List[Stream]:
    """提取分流信息 - 支持多种格式模式，更通用的方法"""
    # 策略1: 查找明确的分流说明部分（简体关键词）
    stream_intro_patterns = [
        r'(?:are divided into|分设).*?(?:streams|专修范围|stream).*?[：:]\s*([^A-Z会计学]+?)(?=\n[A-Z]|Accounting|会计|Students|学生|Total|共|Elective|选修)',
        r'(?:Studies in|本课程).*?(?:divided into|分设).*?(?:streams|专修范围).*?[：:]\s*([^A-Z会计学]+?)(?=\n[A-Z]|Accounting|会计|Students|学生)',
        r'(?:specialize in one of the|选修其中一项).*?[：:]\s*([^A-Z会计学]+?)(?=\n[A-Z]|Accounting|会计|Students|学生)',
    ]
    
    stream_intro_text = ""
    for pattern in stream_intro_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            stream_intro_text = match.group(1)
            if len(stream_intro_text) < 500:
                break
    
    # 策略2: 查找选修课程部分，分流通常在这里
    elective_section_match = re.search(
        r'(?:Elective Courses|选修科目|选修课程)[:\s]*([^N]+?)(?=Notes|注|Total|共|Explanatory)',
        text,
        re.IGNORECASE | re.DOTALL
    )
    
    # 策略3: 查找包含分流关键词的段落
    stream_keyword_section = ""
    if not stream_intro_text and not elective_section_match:
        stream_para_match = re.search(
            r'(?:^|\n).*?(?:stream|方向|专修范围).*?[：:]\s*([^\n]{50,500}?)(?=\n\n|\n[A-Z]{3,})',
            text,
            re.IGNORECASE | re.DOTALL | re.MULTILINE
        )
        if stream_para_match:
            stream_keyword_section = stream_para_match.group(1)
    
    # 优先使用分流说明部分，否则使用选修部分，再使用关键词部分，最后使用整个文本
    if stream_intro_text:
        search_text = stream_intro_text
    elif elective_section_match:
        search_text = elective_section_match.group(1)
    elif stream_keyword_section:
        search_text = stream_keyword_section
    else:
        search_text = text
    
    # 使用统一的提取函数
    streams = _extract_streams_from_text(search_text, allow_multiline=False)
    
    # 统一过滤掉非分流项（如年级、要求等）
    filtered_streams = []
    for stream in streams:
        name_en = (stream.name_en or '').lower()
        name_zh = (stream.name_zh or '').lower()
        name = name_en + ' ' + name_zh
        
        # 跳过明显的非分流项
        if any(keyword in name for keyword in [
            'year of attendance', 'attendance', 'core requirements',
            'major requirements', 'university core', 'medical core',
            'first year', 'second year', 'third year', 'fourth year',
            'fifth year', 'sixth year', '年级', '年度', '要求', '核心',
            '必修要求', '核心课程', 'university requirements'
        ]):
            continue
        filtered_streams.append(stream)
    
    # 去重
    seen = set()
    unique_streams = []
    for stream in filtered_streams:
        key = (stream.name_en, stream.name_zh)
        if key not in seen and (stream.name_en or stream.name_zh):
            seen.add(key)
            unique_streams.append(stream)
    
    return unique_streams


def extract_programme_info(text: str) -> Dict:
    """提取课程基本信息 - 支持多种格式（文本应为简体）"""
    info = {}
    
    # 提取课程名称 - 更精确的匹配，支持多种格式（简体关键词）
    title_patterns = [
        r'(?:Programme Title|Minor Programme Title|Major Programme Title)[:\s]*([^\n]+)',
        r'(?:课程名称|副修课程名称|主修课程名称)[:\s]*([^\n]+)',
    ]
    
    for pattern in title_patterns:
        title_match = re.search(pattern, text, re.IGNORECASE)
        if title_match:
            name = title_match.group(1).strip()
            name = re.sub(r'\s+', ' ', name)
            # 判断是中英文
            if any('\u4e00' <= c <= '\u9fff' for c in name):
                if not info.get('programme_name_zh'):
                    info['programme_name_zh'] = name
            else:
                if not info.get('programme_name_en'):
                    info['programme_name_en'] = name
    
    # 提取适用年度 - 更精确的匹配（简体关键词）
    year_patterns = [
        r'Applicable to students admitted in ([^\n]+)',
        r'适用于([^\n]+?)(?:入学|年度|学生)',
    ]
    
    for pattern in year_patterns:
        year_match = re.search(pattern, text, re.IGNORECASE)
        if year_match:
            years = year_match.group(1).strip()
            years = re.sub(r'\s+', ' ', years)
            info['applicable_years'] = years
            break
    
    # 识别课程类型（简体关键词）
    if re.search(r'Minor Programme|副修', text, re.IGNORECASE):
        info['programme_type'] = 'minor'
    elif re.search(r'Double Major|双主修', text, re.IGNORECASE):
        info['programme_type'] = 'double_major'
    else:
        info['programme_type'] = 'major'
    
    return info


def extract_chunks_from_pdf(pdf_path: Path) -> StudyScheme:
    """从 PDF 提取结构化信息 - 支持多种PDF格式"""
    scheme = StudyScheme()
    
    faculty = pdf_path.parent.name
    scheme_name = pdf_path.stem
    
    all_text = ""
    all_tables = []
    tables_text = ""
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # 提取文本并立即转换为简体
                text = page.extract_text(layout=True) or ""
                text = to_simplified(text)  # 第一步：转换为简体
                all_text += text + "\n"
                
                # 提取表格 - 尝试多种策略
                tables = page.extract_tables(table_settings=TABLE_SETTINGS) or []
                all_tables.extend(tables)
                
                # 将表格转换为文本（供LLM处理），并转换为简体
                for table in tables:
                    if table:
                        for row in table:
                            if row:
                                row_text = " ".join([str(cell) if cell else "" for cell in row])
                                row_text = to_simplified(row_text)  # 转换为简体
                                tables_text += row_text + "\n"
    except Exception as e:
        logging.warning(f"读取PDF {pdf_path} 时出错: {e}")
        return scheme
    
    # 保存原始内容（已经是简体）
    scheme.raw_text = all_text
    scheme.raw_tables_text = tables_text
    
    # 提取基本信息（文本已经是简体）
    programme_info = extract_programme_info(all_text)
    scheme.programme_name_en = programme_info.get('programme_name_en')
    scheme.programme_name_zh = programme_info.get('programme_name_zh') or ''
    scheme.applicable_years = programme_info.get('applicable_years') or ''
    scheme.programme_type = programme_info.get('programme_type', 'major')
    
    # 提取毕业要求
    requirements = parse_text_for_requirements(all_text)
    scheme.total_units_required = requirements.get('total_units')
    scheme.school_package_units = requirements.get('school_package_units')
    scheme.required_courses_units = requirements.get('required_units')
    scheme.elective_courses_units = requirements.get('elective_units')
    
    # 从文本中提取课程代码 - 改进的匹配模式（简体关键词）
    # 查找 School Package 部分 - 支持多种格式
    school_patterns = [
        r'(?:School Package|学院课程)[:\s]*([^2-3]+?)(?=\d+\.\s*(?:Required|必修|Elective|选修))',
        r'1\.\s*(?:School Package|学院课程)[:\s]*([^2]+?)(?=2\.)',
    ]
    
    for pattern in school_patterns:
        school_package_match = re.search(pattern, all_text, re.IGNORECASE | re.DOTALL)
        if school_package_match:
            school_text = school_package_match.group(1)
            scheme.school_package_courses = parse_course_list(school_text)
            for course in scheme.school_package_courses:
                course.course_type = "school_package"
            break
    
    # 查找 Required Courses 部分（简体关键词）
    required_patterns = [
        r'(?:Required Courses|必修科目|必修课程)[:\s]*([^3]+?)(?=\d+\.\s*(?:Elective|选修)|Total|共|Notes|注)',
        r'2\.\s*(?:Required Courses|必修科目|必修课程)[:\s]*([^3]+?)(?=3\.)',
    ]
    
    for pattern in required_patterns:
        required_match = re.search(pattern, all_text, re.IGNORECASE | re.DOTALL)
        if required_match:
            required_text = required_match.group(1)
            scheme.required_courses = parse_course_list(required_text)
            for course in scheme.required_courses:
                course.course_type = "required"
            break
    
    # 查找 Elective Courses 部分（简体关键词）
    elective_patterns = [
        r'(?:Elective Courses|选修科目|选修课程)[:\s]*([^N]+?)(?=Notes|注|Total|共|Explanatory)',
        r'3\.\s*(?:Elective Courses|选修科目|选修课程)[:\s]*([^N]+?)(?=Notes|注|Total|共)',
    ]
    
    for pattern in elective_patterns:
        elective_match = re.search(pattern, all_text, re.IGNORECASE | re.DOTALL)
        if elective_match:
            elective_text = elective_match.group(1)
            scheme.elective_courses = parse_course_list(elective_text)
            for course in scheme.elective_courses:
                course.course_type = "elective"
            break
    
    # 从表格中提取详细课程信息
    for table in all_tables:
        if not table:
            continue
        
        # 检查是否是课程列表表格（简体关键词）
        header_str = " ".join([str(cell) for cell in table[0] if cell])
        header_str = to_simplified(header_str)  # 转换为简体
        if any(keyword in header_str for keyword in ["Course Code", "课程代码", "代码", "Code", "Course"]):
            table_courses = parse_table_for_courses(table, 0)
            # 合并到现有课程列表
            for table_course in table_courses:
                # 查找是否已存在
                existing = None
                for course_list in [scheme.school_package_courses, scheme.required_courses, scheme.elective_courses]:
                    for c in course_list:
                        if c.code == table_course.code:
                            existing = c
                            break
                    if existing:
                        break
                
                if existing:
                    # 更新现有课程信息
                    if table_course.name_en and not existing.name_en:
                        existing.name_en = table_course.name_en
                    if table_course.name_zh and not existing.name_zh:
                        existing.name_zh = table_course.name_zh
                    if table_course.units and not existing.units:
                        existing.units = table_course.units
                else:
                    # 添加到选修课程（如果没有指定类型）
                    if not table_course.course_type:
                        table_course.course_type = "elective"
                    scheme.elective_courses.append(table_course)
    
    # 提取分流信息 - 从文本和表格中提取
    # 策略1: 从文本中提取
    scheme.streams = extract_streams(all_text)
    
    # 策略2: 如果文本中没有找到，尝试从表格中提取
    if not scheme.streams:
        table_text = ""
        for table in all_tables:
            if table:
                # 将表格转换为文本（保持结构）
                for row in table:
                    if row:
                        row_text = " ".join([str(cell) if cell else "" for cell in row])
                        table_text += row_text + "\n"
        
        if table_text:
            table_text = to_simplified(table_text)  # 转换为简体
            table_streams = extract_streams(table_text)
            if table_streams:
                scheme.streams = table_streams
    
    # 策略3: 如果还是没找到，合并所有文本（包括表格文本）再提取
    if not scheme.streams:
        combined_text = all_text
        for table in all_tables:
            if table:
                for row in table:
                    if row:
                        row_text = " ".join([str(cell) if cell else "" for cell in row])
                        row_text = to_simplified(row_text)  # 转换为简体
                        combined_text += "\n" + row_text
        
        scheme.streams = extract_streams(combined_text)
    
    # 策略4: 如果仍然没找到，尝试更激进的提取方法（包括表格单元格）
    if not scheme.streams:
        # 合并所有文本和表格文本
        combined_text = all_text + "\n" + tables_text
        aggressive_streams = _extract_streams_from_text(combined_text, allow_multiline=True)
        if aggressive_streams:
            scheme.streams = aggressive_streams
        
        # 如果还是没找到，直接从表格单元格中提取（支持多行）
        if not scheme.streams:
            for table in all_tables:
                if not table:
                    continue
                for row in table:
                    if not row:
                        continue
                    for cell in row:
                        if not cell:
                            continue
                        cell_text = to_simplified(str(cell))
                        cell_streams = _extract_streams_from_text(cell_text, allow_multiline=True)
                        for s in cell_streams:
                            if not any(existing.name_en == s.name_en and existing.name_zh == s.name_zh 
                                     for existing in scheme.streams):
                                scheme.streams.append(s)
    
    # 提取备注 - 更精确的匹配（简体关键词）
    notes_section_patterns = [
        r'(?:Notes|注)[:\s]*\n(.*?)(?=\n\n|\n[A-Z]|$)',
        r'Explanatory Notes[:\s]*\n(.*?)(?=\n\n|$)',
    ]
    
    for pattern in notes_section_patterns:
        notes_section_match = re.search(pattern, all_text, re.IGNORECASE | re.DOTALL)
        if notes_section_match:
            notes_text = notes_section_match.group(1)
            notes_match = re.findall(r'\[([a-z])\]\s*([^\n]+)', notes_text, re.IGNORECASE)
            for note_match in notes_match:
                note_content = note_match[1].strip()
                # 过滤掉明显不是备注的文本
                if len(note_content) > 5 and len(note_content) < 500:
                    scheme.notes.append(f"[{note_match[0]}] {note_content}")
            break
    
    return scheme


def scheme_to_dict(scheme: StudyScheme) -> Dict:
    """将 StudyScheme 转换为字典"""
    def course_to_dict(course: Course) -> Dict:
        result = {k: v for k, v in asdict(course).items() if v is not None}
        # 确保课程名称是简体
        if result.get('name_zh'):
            result['name_zh'] = to_simplified(result['name_zh'])
        return result
    
    def stream_to_dict(stream: Stream) -> Dict:
        return {
            'name_en': stream.name_en,
            'name_zh': stream.name_zh,
            'min_units': stream.min_units,
            'max_units': stream.max_units,
            'courses': [course_to_dict(c) for c in stream.courses],
        }
    
    result = {
        'programme_name_en': scheme.programme_name_en,
        'programme_name_zh': scheme.programme_name_zh or '',  # 已经是简体
        'applicable_years': scheme.applicable_years or '',  # 已经是简体
        'programme_type': scheme.programme_type,
        'total_units_required': scheme.total_units_required,
        'school_package_units': scheme.school_package_units,
        'required_courses_units': scheme.required_courses_units,
        'elective_courses_units': scheme.elective_courses_units,
        'school_package_courses': [course_to_dict(c) for c in scheme.school_package_courses],
        'required_courses': [course_to_dict(c) for c in scheme.required_courses],
        'elective_courses': [course_to_dict(c) for c in scheme.elective_courses],
        'streams': [stream_to_dict(s) for s in scheme.streams],
        'recommended_pattern': scheme.recommended_pattern,
        'notes': scheme.notes or [],  # 已经是简体
        # 保留原始简体PDF全文数据
        'raw_text': scheme.raw_text or '',  # 完整的PDF文本内容（简体）
        'raw_tables_text': scheme.raw_tables_text or '',  # 表格文本内容（简体）
    }
    
    # 确保所有中文字段都是简体（双重保险）
    if result['programme_name_zh']:
        result['programme_name_zh'] = to_simplified(result['programme_name_zh'])
    if result['applicable_years']:
        result['applicable_years'] = to_simplified(result['applicable_years'])
    if result['notes']:
        result['notes'] = [to_simplified(note) for note in result['notes']]
    
    # 确保分流名称也是简体
    for stream_dict in result['streams']:
        if stream_dict.get('name_zh'):
            stream_dict['name_zh'] = to_simplified(stream_dict['name_zh'])
    
    # 如果分流未识别，添加额外的提示信息（原始全文已经在raw_text中）
    if not scheme.streams and scheme.raw_text:
        # 提取可能包含分流的部分（简体关键词），作为额外提示
        elective_section = ""
        if scheme.elective_courses:
            # 尝试提取选修部分
            elective_match = re.search(
                r'(?:Elective Courses|选修科目|选修课程)[:\s]*([^N]+?)(?=Notes|注|Total|共|Explanatory)',
                scheme.raw_text,
                re.IGNORECASE | re.DOTALL
            )
            if elective_match:
                elective_section = elective_match.group(1)[:2000]  # 限制长度
        
        if elective_section:
            result['raw_elective_section'] = elective_section  # 额外的提示信息
    
    return result


def process_single_pdf(pdf_path: Path) -> Tuple[Path, Dict, Optional[str]]:
    """处理单个PDF文件（用于并行处理）"""
    try:
        scheme = extract_chunks_from_pdf(pdf_path)
        scheme_dict = scheme_to_dict(scheme)
        
        # 从文件名提取专业信息
        faculty = pdf_path.parent.name
        filename = pdf_path.stem
        
        scheme_dict['pdf_path'] = str(pdf_path)
        scheme_dict['faculty'] = faculty
        scheme_dict['filename'] = filename
        
        # 统计信息
        course_count = (
            len(scheme.school_package_courses) +
            len(scheme.required_courses) +
            len(scheme.elective_courses)
        )
        
        return pdf_path, scheme_dict, None
    except Exception as e:
        return pdf_path, {}, str(e)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="增强版 study scheme PDF 解析器（支持并行处理）"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/study_schemes"),
        help="存放 study scheme PDF 的根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/study_schemes_processed"),
        help="输出目录（每个PDF生成一个JSON文件）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选：输出单个JSONL文件（所有结果合并）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 个 PDF，方便调试",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=f"并行处理的进程数（默认：CPU核心数）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印解析统计，不写文件",
    )
    return parser.parse_args()


def main() -> None:
    # 设置标准输出编码为UTF-8（Windows）
    if sys.platform == 'win32':
        import io
        import codecs
        # 重新包装stdout和stderr以支持UTF-8
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # 获取所有 PDF 文件
    pdf_files = sorted(args.input_dir.rglob("*.pdf"))
    if args.limit:
        pdf_files = pdf_files[:args.limit]
    
    if not pdf_files:
        logging.error("未在 %s 找到 PDF 文件", args.input_dir)
        return
    
    logging.info("找到 %d 个 PDF 文件", len(pdf_files))
    
    # 确定并行工作进程数
    max_workers = args.max_workers or cpu_count()
    max_workers = min(max_workers, len(pdf_files))
    
    logging.info("使用 %d 个进程并行处理", max_workers)
    
    # 创建输出目录
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_fp = None
        if args.output:
            output_fp = args.output.open("w", encoding="utf-8")
    
    total_processed = 0
    total_courses = 0
    errors = []
    results = []
    
    try:
        # 并行处理
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_pdf = {
                executor.submit(process_single_pdf, pdf_path): pdf_path
                for pdf_path in pdf_files
            }
            
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    processed_path, scheme_dict, error = future.result()
                    
                    if error:
                        errors.append((str(pdf_path), error))
                        logging.error("处理 %s 失败: %s", pdf_path.name, error)
                        continue
                    
                    # 统计课程数量
                    course_count = (
                        len(scheme_dict.get('school_package_courses', [])) +
                        len(scheme_dict.get('required_courses', [])) +
                        len(scheme_dict.get('elective_courses', []))
                    )
                    total_courses += course_count
                    
                    # 生成输出文件名（基于PDF文件名）
                    output_filename = pdf_path.stem + ".json"
                    output_path = args.output_dir / output_filename
                    
                    logging.info(
                        "%s -> %d 门课程, %d 个分流",
                        pdf_path.name,
                        course_count,
                        len(scheme_dict.get('streams', []))
                    )
                    
                    if not args.dry_run:
                        # 写入单独的文件
                        with output_path.open("w", encoding="utf-8") as f:
                            json.dump(scheme_dict, f, ensure_ascii=False, indent=2)
                        
                        # 如果指定了合并输出，也写入
                        if output_fp:
                            json.dump(scheme_dict, output_fp, ensure_ascii=False)
                            output_fp.write("\n")
                    
                    results.append((pdf_path, course_count))
                    total_processed += 1
                    
                except Exception as e:
                    errors.append((str(pdf_path), str(e)))
                    logging.error("处理 %s 时出错: %s", pdf_path.name, e, exc_info=True)
    
    finally:
        if output_fp:
            output_fp.close()
    
    # 输出统计信息
    logging.info("=" * 60)
    logging.info("处理完成!")
    logging.info("  成功处理: %d 个 PDF", total_processed)
    logging.info("  提取课程: %d 门", total_courses)
    logging.info("  错误数量: %d", len(errors))
    if not args.dry_run:
        logging.info("  输出目录: %s", args.output_dir)
        if args.output:
            logging.info("  合并输出: %s", args.output)
    
    if errors:
        logging.warning("以下文件处理失败:")
        for pdf_path, error in errors[:10]:  # 只显示前10个错误
            logging.warning("  - %s: %s", Path(pdf_path).name, error)
        if len(errors) > 10:
            logging.warning("  ... 还有 %d 个错误", len(errors) - 10)


if __name__ == "__main__":
    main()
