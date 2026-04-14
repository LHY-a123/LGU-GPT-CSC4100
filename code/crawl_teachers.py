#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUHKSZ教师列表爬虫
功能：爬取教师搜索页面的所有分页，并提取每个教师的详细信息
"""

import os
import json
import time
import requests
import urllib3
import ssl
import warnings
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from pathlib import Path
import re
from typing import Dict, List, Set
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 尝试导入cloudscraper（可选，用于更好的反爬虫处理）
import cloudscraper

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.simplefilter('ignore', urllib3.exceptions.InsecureRequestWarning)


class TLS12HttpAdapter(HTTPAdapter):
    """强制使用 TLS1.2 的适配器，提高与旧/严格站点的握手兼容性"""
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.set_ciphers("ECDHE+AESGCM:HIGH:!aNULL:!MD5:!RC4")
        pool_kwargs["ssl_context"] = context
        return super().init_poolmanager(connections, maxsize, block, **pool_kwargs)


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TeacherCrawler:
    def __init__(
        self,
        base_url: str = "https://www.cuhk.edu.cn/zh-hans/aggregate-search-teacher",
        output_slug: str = "teachers",
        delay: float = 1.0,
        resume: bool = True,
    ):
        """
        初始化教师爬虫
        
        Args:
            base_url: 教师搜索页面URL
            output_slug: 输出文件名标识
            delay: 请求延迟（秒）
            resume: 是否启用断点续传
        """
        self.base_url = base_url.rstrip('/')
        self.output_dir = Path("data/raw")
        self.output_slug = output_slug
        self.delay = delay
        self.resume = resume
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = Path("data/states")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储已访问的URL
        self.visited_teacher_urls: Set[str] = set()
        self.visited_page_urls: Set[str] = set()
        
        # 统计信息
        self.stats = {
            'total_pages': 0,
            'total_teachers': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
        
        # 初始化JSONL文件
        self.jsonl_file = self.output_dir / f"{self.output_slug}.jsonl"
        self.state_file = self.state_dir / f"{self.output_slug}_state.json"
        
        # 初始化session
        self.session = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False},
            delay=1
        )
        
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # 安全重试策略
        retry_strategy = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        # 挂载 TLS1.2 适配器
        tls_adapter = TLS12HttpAdapter(max_retries=retry_strategy)
        self.session.mount("http://", tls_adapter)
        self.session.mount("https://", tls_adapter)
        logger.info("已挂载 TLS1.2 适配器")

    def _load_state(self):
        """加载断点续传的状态"""
        if self.resume and self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                visited_teachers = state.get('visited_teachers', [])
                visited_pages = state.get('visited_pages', [])
                self.stats = state.get('stats', self.stats)
                
                for url in visited_teachers:
                    self.visited_teacher_urls.add(url)
                for url in visited_pages:
                    self.visited_page_urls.add(url)
                    
                logger.info(f"恢复爬取状态: 已访问 {len(self.visited_teacher_urls)} 个教师页面, {len(self.visited_page_urls)} 个列表页面")
            except Exception as exc:
                logger.warning(f"加载状态文件失败，将重新开始: {exc}")

    def _save_state(self):
        """保存当前的爬取状态"""
        state_payload = {
            'base_url': self.base_url,
            'visited_teachers': list(self.visited_teacher_urls),
            'visited_pages': list(self.visited_page_urls),
            'stats': self.stats,
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state_payload, f, ensure_ascii=False, indent=2)

    def safe_get(self, url, timeout=30):
        """带重试的 GET 请求"""
        response = self.session.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        return response

    def get_max_page(self, soup: BeautifulSoup) -> int:
        """从页面获取最大页码"""
        max_page = 1
        
        # 方法1: 查找分页元素，常见的选择器
        pagination_selectors = [
            '.pagination',
            '.pager',
            '.page-nav',
            '[class*="pagination"]',
            '[class*="pager"]',
            '[class*="page"]',
            'nav[aria-label*="页"]',
            'nav[aria-label*="page"]'
        ]
        
        for selector in pagination_selectors:
            pagination = soup.select_one(selector)
            if pagination:
                # 查找所有页码链接
                page_links = pagination.find_all('a', href=True)
                for link in page_links:
                    href = link.get('href', '')
                    # 查找page=数字的模式
                    match = re.search(r'page=(\d+)', href)
                    if match:
                        page_num = int(match.group(1))
                        max_page = max(max_page, page_num)
                
                # 也检查文本内容中的页码
                text = pagination.get_text()
                # 查找类似 "1 2 3 4 5 ... 10" 的模式
                page_matches = re.findall(r'\b(\d+)\b', text)
                for match in page_matches:
                    try:
                        page_num = int(match)
                        if 1 <= page_num <= 1000:  # 合理的页码范围
                            max_page = max(max_page, page_num)
                    except:
                        pass
        
        # 方法2: 在整个页面中搜索包含page=的链接
        all_links = soup.find_all('a', href=True)
        for link in all_links:
            href = link.get('href', '')
            match = re.search(r'page=(\d+)', href)
            if match:
                page_num = int(match.group(1))
                max_page = max(max_page, page_num)
        
        logger.info(f"检测到的最大页码: {max_page}")
        return max_page if max_page > 1 else 1

    def extract_teacher_links(self, soup: BeautifulSoup, page_url: str) -> List[str]:
        """从列表页面提取教师链接"""
        teacher_links = []
        seen_urls = set()
        
        # 尝试多种选择器来找到教师列表
        # 通常教师列表会在表格、列表或特定的容器中
        selectors = [
            'table tbody tr td a',
            'table tbody tr a',
            '.teacher-list a',
            '.teacher-item a',
            '[class*="teacher"] a',
            '[class*="faculty"] a',
            'ul li a',
            '.list-item a',
            '.search-result a',
            'article a',
            'main a',
            '.content a'
        ]
        
        for selector in selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href', '')
                if href:
                    full_url = urljoin(page_url, href)
                    normalized_url = full_url.split('?')[0].split('#')[0]  # 移除查询参数和锚点
                    # 过滤掉明显的非教师链接（如分页链接、导航链接等）
                    if normalized_url not in seen_urls and self._is_teacher_link(full_url, link):
                        teacher_links.append(full_url)
                        seen_urls.add(normalized_url)
        
        # 如果上面的选择器都没找到，尝试查找主要内容区域的所有链接并过滤
        if not teacher_links:
            # 尝试找到主要内容区域
            main_content = soup.select_one('main, .content, .main-content, #content, #main')
            if main_content:
                all_links = main_content.find_all('a', href=True)
            else:
                all_links = soup.find_all('a', href=True)
            
            for link in all_links:
                href = link.get('href', '')
                if href:
                    full_url = urljoin(page_url, href)
                    normalized_url = full_url.split('?')[0].split('#')[0]
                    if normalized_url not in seen_urls and self._is_teacher_link(full_url, link):
                        teacher_links.append(full_url)
                        seen_urls.add(normalized_url)
        
        logger.info(f"从页面提取到 {len(teacher_links)} 个教师链接")
        return teacher_links

    def _is_teacher_link(self, url: str, link_element) -> bool:
        """判断是否是教师详情页链接"""
        # 排除外部域名（如 weibo.com）
        parsed = urlparse(url)
        excluded_domains = ['weibo.com', 'twitter.com', 'facebook.com', 'linkedin.com', 
                          'instagram.com', 'youtube.com', 'i.cuhk.edu.cn']
        if parsed.netloc and any(domain in parsed.netloc.lower() for domain in excluded_domains):
            return False
        
        # 排除分页链接
        if 'page=' in url or 'aggregate-search-teacher' in url:
            return False
        
        # 排除明显的导航链接
        excluded_keywords = ['javascript:', 'mailto:', 'tel:', '#', 'javascript']
        if any(keyword in url.lower() for keyword in excluded_keywords):
            return False
        
        # 排除文件链接
        excluded_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.jpg', '.png']
        if any(url.lower().endswith(ext) for ext in excluded_extensions):
            return False
        
        # 检查URL路径，排除常见的非教师页面
        path = parsed.path.lower()
        
        # 排除常见的网站导航页面
        excluded_paths = [
            '/students', '/faculty-staff', '/visitors', '/parents', '/about-us',
            '/governing-board', '/university-officers', '/news-events', '/article-search',
            '/event-search', '/media', '/academics', '/research', '/research-news',
            '/photo', '/clear', '/taxonomy', '/node', '/page',
            '/college', '/event', '/news', '/article', '/media'
        ]
        if any(path.startswith(excluded) or path == excluded.rstrip('/') for excluded in excluded_paths):
            return False
        
        # 排除学院主页（通常是根路径或很短的路径）
        if path == '/' or path == '' or (path.count('/') == 1 and path.endswith('/')):
            return False
        
        # 排除以常见非教师关键词开头的路径
        non_teacher_prefixes = ['/zh-hans/students', '/zh-hans/faculty', '/zh-hans/visitors',
                                '/zh-hans/parents', '/zh-hans/about', '/zh-hans/news',
                                '/zh-hans/research', '/zh-hans/academics', '/zh-hans/media',
                                '/zh-hans/event', '/zh-hans/article', '/zh-hans/taxonomy',
                                '/zh-hans/node', '/zh-hans/page', '/zh-hans/college']
        if any(path.startswith(prefix) for prefix in non_teacher_prefixes):
            return False
        
        # 检查链接文本，教师链接通常包含姓名
        link_text = link_element.get_text(strip=True)
        
        # 如果链接文本为空或太长，不是教师链接
        if not link_text or len(link_text) > 50:
            return False
        
        # 排除明显的非人名文本
        excluded_texts = ['更多', 'more', '查看', 'view', '详情', 'detail', '链接', 'link',
                          '首页', 'home', '返回', 'back', '上一页', '下一页', 'next', 'prev',
                          '搜索', 'search', '登录', 'login', '注册', 'register']
        if link_text.lower() in [t.lower() for t in excluded_texts]:
            return False
        
        # 检查链接文本是否看起来像人名
        # 中文姓名通常是2-4个字符
        # 英文姓名通常包含空格或点，或者是一个合理的名字长度
        is_chinese_name = False
        is_english_name = False
        
        # 检查是否是中文姓名（2-4个中文字符）
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', link_text)
        if len(chinese_chars) >= 2 and len(chinese_chars) <= 4 and len(link_text) <= 6:
            is_chinese_name = True
        
        # 检查是否是英文姓名（包含空格或点，或合理的名字格式）
        if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)+$', link_text):  # 如 "John Smith"
            is_english_name = True
        elif re.match(r'^[A-Z][a-z]+\.[A-Z][a-z]+$', link_text):  # 如 "J.Smith"
            is_english_name = True
        elif re.match(r'^[A-Z][a-z]{2,20}$', link_text):  # 单个英文名
            is_english_name = True
        
        # 如果链接文本看起来像人名，且URL路径合理，可能是教师链接
        if is_chinese_name or is_english_name:
            # URL路径应该有一定的深度（至少2层），但不应该是列表页
            if path.count('/') >= 2 and 'search' not in path and 'list' not in path:
                return True
        
        # 如果URL包含教师相关的路径关键词
        teacher_keywords = ['teacher', 'faculty', 'professor', 'people', 'staff']
        if any(keyword in path for keyword in teacher_keywords):
            # 但排除列表页
            if 'search' not in path and 'list' not in path:
                return True
        
        return False

    def _is_minimal_page(self, teacher_data: Dict) -> bool:
        """判断页面内容是否过少（只有基本信息的行政人员页面）"""
        content = teacher_data.get('content', '')
        word_count = teacher_data.get('word_count', 0)
        teacher_info = teacher_data.get('teacher_info', {})
        
        # 检查内容行数，如果只有很少的非空行（少于3行），可能是基本信息页面
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) < 3:
            return True
        
        # 如果字数太少（少于15个词），可能是只有基本信息的页面
        if word_count < 15:
            return True
        
        # 检查是否只包含基本信息（姓名、职位、邮箱）
        # 行政人员页面通常只有：姓名、职位、电子邮件
        content_lower = content.lower()
        has_name = bool(teacher_info.get('name', ''))
        has_email = bool(teacher_info.get('email', ''))
        
        # 检查是否包含描述性关键词（教师页面通常会有这些内容）
        descriptive_keywords = [
            '研究', 'research', '教育', 'education', '背景', 'background', 
            '经历', 'experience', '发表', 'publication', '论文', 'paper',
            '项目', 'project', '课程', 'course', '教学', 'teaching',
            '简介', 'introduction', 'biography', 'bio', '个人', 'personal',
            '博士', 'phd', '教授', 'professor', '副教授', 'associate',
            '学位', 'degree', '学术', 'academic', '领域', 'field',
            '研究领域', 'research area', '研究兴趣', 'research interest',
            '教育背景', 'educational background', '文学', '文学学士',
            '文学硕士', '艺术硕士', '学部分类', '创意', '剧本',
            '社交媒体', '语料库', '第二语言', '语言教学', '语言教育'
        ]
        has_descriptive = any(keyword in content_lower for keyword in descriptive_keywords)
        
        # 如果有描述性关键词，即使字数较少也认为是有效页面
        if has_descriptive:
            return False
        
        # 如果字数少于30且没有描述性内容，可能是基本信息页面
        if word_count < 30 and not has_descriptive:
            return True
        
        # 如果只有姓名和邮箱，且字数少于20，且没有描述性内容，肯定是基本信息页面
        if has_name and has_email and word_count < 20 and not has_descriptive:
            return True
        
        return False

    def extract_teacher_info(self, soup: BeautifulSoup, url: str) -> Dict:
        """提取教师详细信息"""
        # 提取标题
        title = ''
        if soup.title:
            title = soup.title.get_text(strip=True)
        elif soup.find('h1'):
            title = soup.find('h1').get_text(strip=True)
        
        # 提取主内容
        main_content = ''
        content_selectors = [
            'main', 'article', '.content', '.main-content',
            '#content', '#main', '.post-content', '.entry-content',
            '.teacher-detail', '.profile', '.bio'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                text = content_elem.get_text(separator='\n', strip=True)
                if len(text) > 50:
                    main_content = text
                    break
        
        # 如果没找到，使用body
        if not main_content or len(main_content) < 50:
            soup_copy = BeautifulSoup(str(soup), 'html.parser')
            for script in soup_copy(["script", "style", "noscript"]):
                script.decompose()
            main_content = soup_copy.get_text(separator='\n', strip=True)
            
            # 清理空白行
            lines = []
            for line in main_content.split('\n'):
                cleaned_line = line.strip()
                if cleaned_line:
                    lines.append(cleaned_line)
            main_content = '\n'.join(lines)
        
        # 提取结构化信息（尝试提取姓名、职位、邮箱等）
        teacher_info = {
            'name': '',
            'title': '',
            'email': '',
            'department': '',
            'research_areas': '',
            'bio': ''
        }
        
        # 尝试从页面中提取结构化信息
        # 查找包含姓名、职位等的元素
        all_text = soup.get_text()
        
        # 提取邮箱
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, all_text)
        if emails:
            teacher_info['email'] = emails[0]
        
        # 提取标题（通常是h1或h2）
        if soup.find('h1'):
            teacher_info['name'] = soup.find('h1').get_text(strip=True)
        elif soup.find('h2'):
            teacher_info['name'] = soup.find('h2').get_text(strip=True)
        
        # 查找个人网站链接（港中大（深圳）个人网站）
        personal_website_url = None
        personal_website_keywords = ['港中大（深圳）个人网站', '个人网站', 'personal website', 
                                    'myweb.cuhk.edu.cn', '港中大个人网站']
        
        # 查找包含这些关键词的链接
        for link in soup.find_all('a', href=True):
            link_text = link.get_text(strip=True)
            href = link.get('href', '')
            
            # 检查链接文本或URL是否包含个人网站关键词
            if any(keyword in link_text for keyword in personal_website_keywords) or \
               'myweb.cuhk.edu.cn' in href.lower():
                full_url = urljoin(url, href)
                # 清理URL（移除锚点等）
                personal_website_url = full_url.split('#')[0].split('?')[0]
                logger.info(f"发现个人网站链接: {personal_website_url}")
                break
        
        return {
            'url': url,
            'title': title,
            'content': main_content,
            'teacher_info': teacher_info,
            'word_count': len(main_content.split()),
            'personal_website_url': personal_website_url,  # 添加个人网站URL
            'crawl_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }

    def crawl_personal_website_all_pages(self, base_url: str, teacher_info: Dict) -> Dict:
        """爬取个人网站的所有标签页（Home、Research、Publications、Teaching等）"""
        all_content_parts = []
        visited_pages = set()
        
        # 常见的个人网站标签页
        common_tabs = ['home', 'research', 'publications', 'teaching', 'about', 'biography', 
                       'cv', 'contact', 'news', 'projects', 'students']
        
        # 首先爬取基础URL（通常是Home页面）
        try:
            logger.info(f"爬取个人网站首页: {base_url}")
            response = self.safe_get(base_url, timeout=30)
            if response.status_code == 200:
                html_content = response.content.decode('utf-8', errors='replace')
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # 提取首页内容
                for script in soup(["script", "style", "noscript"]):
                    script.decompose()
                home_content = soup.get_text(separator='\n', strip=True)
                if home_content:
                    all_content_parts.append(f"=== Home ===\n{home_content}")
                    visited_pages.add(base_url)
                
                # 查找导航链接，提取所有标签页URL
                tab_urls = set()
                
                # 查找导航菜单
                nav_selectors = ['nav', '.nav', '.navigation', '.menu', '.navbar', 
                                '[class*="nav"]', '[class*="menu"]', 'ul.nav', 'ul.menu']
                for selector in nav_selectors:
                    nav_elem = soup.select_one(selector)
                    if nav_elem:
                        for link in nav_elem.find_all('a', href=True):
                            href = link.get('href', '')
                            link_text = link.get_text(strip=True).lower()
                            
                            # 检查是否是标签页链接
                            if any(tab in link_text for tab in common_tabs) or \
                               any(tab in href.lower() for tab in common_tabs):
                                full_url = urljoin(base_url, href)
                                # 清理URL
                                clean_url = full_url.split('#')[0].split('?')[0]
                                if clean_url not in visited_pages:
                                    tab_urls.add((clean_url, link_text))
                
                # 也查找所有链接，寻找可能的标签页
                base_parsed = urlparse(base_url)
                base_domain = base_parsed.netloc
                
                for link in soup.find_all('a', href=True):
                    href = link.get('href', '')
                    link_text = link.get_text(strip=True).lower()
                    
                    # 如果链接文本是常见的标签页名称
                    if link_text in common_tabs or any(tab in link_text for tab in common_tabs):
                        full_url = urljoin(base_url, href)
                        clean_url = full_url.split('#')[0].split('?')[0]
                        
                        # 检查是否是同一域名
                        parsed_url = urlparse(clean_url)
                        if parsed_url.netloc == base_domain and clean_url not in visited_pages:
                            tab_urls.add((clean_url, link_text))
                
                # 爬取所有找到的标签页
                for tab_url, tab_name in tab_urls:
                    if tab_url in visited_pages:
                        continue
                    
                    try:
                        logger.info(f"爬取个人网站标签页: {tab_name} - {tab_url}")
                        tab_response = self.safe_get(tab_url, timeout=30)
                        if tab_response.status_code == 200:
                            tab_html = tab_response.content.decode('utf-8', errors='replace')
                            tab_soup = BeautifulSoup(tab_html, 'html.parser')
                            
                            # 提取标签页内容
                            for script in tab_soup(["script", "style", "noscript"]):
                                script.decompose()
                            tab_content = tab_soup.get_text(separator='\n', strip=True)
                            
                            if tab_content:
                                # 使用标签名称作为标题
                                tab_title = tab_name.capitalize() if tab_name else 'Page'
                                all_content_parts.append(f"=== {tab_title} ===\n{tab_content}")
                                visited_pages.add(tab_url)
                                
                                if self.delay:
                                    time.sleep(self.delay)
                    except Exception as e:
                        logger.warning(f"爬取标签页失败 {tab_url}: {e}")
                        continue
                
        except Exception as e:
            logger.warning(f"爬取个人网站失败 {base_url}: {e}")
        
        # 合并所有内容
        combined_content = '\n\n'.join(all_content_parts)
        
        return {
            'content': combined_content,
            'word_count': len(combined_content.split()),
            'pages_crawled': len(visited_pages)
        }

    def crawl_teacher_page(self, teacher_url: str) -> bool:
        """爬取单个教师页面"""
        if teacher_url in self.visited_teacher_urls:
            logger.debug(f"跳过已访问的教师页面: {teacher_url}")
            return True
        
        try:
            logger.info(f"爬取教师页面: {teacher_url}")
            
            response = self.safe_get(teacher_url, timeout=30)
            
            if response.status_code >= 400:
                logger.warning(f"HTTP状态码 {response.status_code}: {teacher_url}")
                if response.status_code == 404:
                    return False
            
            # 检查内容类型
            content_type = response.headers.get('Content-Type', '')
            if content_type and 'text/html' not in content_type:
                logger.debug(f"跳过（非HTML）: {teacher_url}")
                return False
            
            # 获取响应文本
            html_content = response.content.decode('utf-8', errors='replace')
            
            # 解析HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 提取内容
            teacher_data = self.extract_teacher_info(soup, teacher_url)
            
            # 检查是否有个人网站链接，如果有则爬取所有标签页并合并到教师页面数据中
            personal_website_url = teacher_data.get('personal_website_url')
            if personal_website_url and personal_website_url not in self.visited_teacher_urls:
                try:
                    logger.info(f"发现个人网站，开始爬取所有标签页: {personal_website_url}")
                    
                    # 爬取个人网站的所有标签页
                    personal_result = self.crawl_personal_website_all_pages(
                        personal_website_url, 
                        teacher_data.get('teacher_info', {})
                    )
                    
                    if personal_result['content']:
                        # 将个人网站内容合并到教师页面数据中
                        teacher_data['content'] = teacher_data.get('content', '') + '\n\n=== 个人网站内容 ===\n' + personal_result['content']
                        teacher_data['word_count'] = len(teacher_data['content'].split())
                        teacher_data['personal_website_crawled'] = True
                        teacher_data['personal_website_pages_crawled'] = personal_result['pages_crawled']
                        logger.info(f"成功爬取个人网站所有标签页内容，共 {personal_result['pages_crawled']} 页，合并后字数: {teacher_data['word_count']}")
                    
                    # 标记个人网站已访问（避免重复爬取）
                    self.visited_teacher_urls.add(personal_website_url)
                except Exception as e:
                    logger.warning(f"爬取个人网站失败 {personal_website_url}: {e}，继续处理原页面")
            
            # 检查页面内容是否足够丰富（过滤掉只有基本信息的行政人员页面）
            if self._is_minimal_page(teacher_data):
                logger.info(f"跳过内容过少的页面（可能是行政人员）: {teacher_url}")
                self.visited_teacher_urls.add(teacher_url)
                self.stats['skipped'] += 1
                return True
            
            # 保存合并后的教师页面数据（包含个人网站内容）
            self.visited_teacher_urls.add(teacher_url)
            self.stats['success'] += 1
            self.stats['total_teachers'] += 1
            
            # 追加到JSONL文件（一条记录包含教师页面和个人网站内容）
            self.append_to_jsonl(teacher_data)
            
            return True
            
        except Exception as e:
            logger.error(f"处理教师页面失败 {teacher_url}: {e}")
            self.stats['failed'] += 1
            return False

    def crawl_list_page(self, page_num: int) -> List[str]:
        """爬取列表页面并返回教师链接"""
        if page_num == 1:
            page_url = self.base_url
        else:
            page_url = f"{self.base_url}?keywords=&page={page_num}"
        
        if page_url in self.visited_page_urls:
            logger.debug(f"跳过已访问的列表页面: {page_url}")
            return []
        
        try:
            logger.info(f"爬取列表页面 {page_num}: {page_url}")
            
            response = self.safe_get(page_url, timeout=30)
            
            if response.status_code >= 400:
                logger.warning(f"HTTP状态码 {response.status_code}: {page_url}")
                return []
            
            # 获取响应文本
            html_content = response.content.decode('utf-8', errors='replace')
            
            # 解析HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 提取教师链接
            teacher_links = self.extract_teacher_links(soup, page_url)
            
            # 标记页面已访问
            self.visited_page_urls.add(page_url)
            self.stats['total_pages'] += 1
            
            return teacher_links
            
        except Exception as e:
            logger.error(f"处理列表页面失败 {page_url}: {e}")
            return []

    def append_to_jsonl(self, teacher_data: Dict):
        """追加单条数据到JSONL文件"""
        cleaned_data = self._clean_data(teacher_data)
        with open(self.jsonl_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(cleaned_data, ensure_ascii=False) + '\n')
            f.flush()

    def _clean_data(self, data: Dict) -> Dict:
        """清理数据中的控制字符"""
        cleaned = {}
        for key, value in data.items():
            if isinstance(value, str):
                cleaned_value = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', value)
                cleaned[key] = cleaned_value
            elif isinstance(value, dict):
                cleaned[key] = self._clean_data(value)
            else:
                cleaned[key] = value
        return cleaned

    def run(self):
        """运行爬虫"""
        logger.info(f"开始爬取教师列表: {self.base_url}")
        logger.info(f"输出文件: {self.jsonl_file}")
        
        start_time = time.time()
        
        # 加载状态
        self._load_state()
        
        # 首先爬取第一页，获取最大页码
        logger.info("正在获取分页信息...")
        first_page_url = self.base_url
        if first_page_url not in self.visited_page_urls:
            response = self.safe_get(first_page_url)
            html_content = response.content.decode('utf-8', errors='replace')
            soup = BeautifulSoup(html_content, 'html.parser')
            max_page = self.get_max_page(soup)
            
            # 提取第一页的教师链接
            first_page_links = self.extract_teacher_links(soup, first_page_url)
            self.visited_page_urls.add(first_page_url)
            self.stats['total_pages'] += 1
            self._save_state()
        else:
            # 如果第一页已访问，重新获取以确定最大页码
            response = self.safe_get(first_page_url)
            html_content = response.content.decode('utf-8', errors='replace')
            soup = BeautifulSoup(html_content, 'html.parser')
            max_page = self.get_max_page(soup)
            first_page_links = []
        
        logger.info(f"检测到最大页码: {max_page}")
        
        # 处理第一页的教师链接
        for teacher_url in first_page_links:
            if teacher_url not in self.visited_teacher_urls:
                self.crawl_teacher_page(teacher_url)
                self._save_state()
                if self.delay:
                    time.sleep(self.delay)
        
        # 遍历所有分页
        consecutive_empty_pages = 0
        for page_num in range(2, max_page + 1):
            teacher_links = self.crawl_list_page(page_num)
            self._save_state()
            
            if not teacher_links:
                consecutive_empty_pages += 1
                logger.warning(f"第 {page_num} 页没有找到教师链接")
                # 如果连续3页都没有链接，提前退出
                if consecutive_empty_pages >= 3:
                    logger.info("连续多页没有找到教师链接，提前结束爬取")
                    break
            else:
                consecutive_empty_pages = 0
            
            # 处理每个教师链接
            for teacher_url in teacher_links:
                if teacher_url not in self.visited_teacher_urls:
                    self.crawl_teacher_page(teacher_url)
                    self._save_state()
                    if self.delay:
                        time.sleep(self.delay)
            
            if self.delay:
                time.sleep(self.delay)
        
        elapsed_time = time.time() - start_time
        
        # 打印统计信息
        logger.info("\n" + "="*50)
        logger.info("爬取完成！")
        logger.info(f"总耗时: {elapsed_time:.2f}秒")
        logger.info(f"访问列表页面: {self.stats['total_pages']}")
        logger.info(f"成功爬取教师: {self.stats['success']}")
        logger.info(f"失败: {self.stats['failed']}")
        logger.info(f"总计教师: {self.stats['total_teachers']}")
        logger.info("="*50)
        
        logger.info(f"\n所有数据已保存到: {self.jsonl_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='爬取CUHKSZ教师列表')
    parser.add_argument(
        '--url',
        type=str,
        default='https://www.cuhk.edu.cn/zh-hans/aggregate-search-teacher',
        help='教师搜索页面URL'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='teachers',
        help='输出文件名标识'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0,
        help='请求延迟（秒）'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='禁用断点续传'
    )
    
    args = parser.parse_args()
    
    crawler = TeacherCrawler(
        base_url=args.url,
        output_slug=args.output,
        delay=args.delay,
        resume=not args.no_resume
    )
    
    crawler.run()


if __name__ == '__main__':
    main()

