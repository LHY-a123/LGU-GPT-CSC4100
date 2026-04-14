#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUHKSZ官网爬虫
功能：递归爬取网站所有页面，保持良好结构
"""

import os
import json
import time
import requests
import urllib3
import ssl
import warnings
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
from bs4 import BeautifulSoup
from pathlib import Path
import re
import ast
from typing import Set, Dict, List
import logging
from multiprocessing import Pool, cpu_count
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import SSLError

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

DEFAULT_SEED_URLS = [
    "https://hss.cuhk.edu.cn"
]


class WebsiteCrawler:
    def __init__(
        self,
        base_url: str,
        output_slug: str,
        max_depth: int = 5,
        delay: float = 1.0,
        allowed_domains: List[str] = None,
        resume: bool = True,
        max_pages: int = 2000,
    ):
        """
        初始化爬虫
        
        Args:
            base_url: 起始URL
            output_dir: 输出目录
            max_depth: 最大爬取深度
            delay: 请求延迟（秒）
            allowed_domains: 允许的域名列表
            max_pages: 最大爬取网页数量
        """
        self.base_url = base_url.rstrip('/')
        self.output_dir = Path("data/raw")
        self.output_slug = output_slug
        self.max_depth = max_depth
        self.delay = delay
        self.max_pages = max_pages
        self.allowed_domains = allowed_domains or [urlparse(base_url).netloc]
        if 'myweb.cuhk.edu.cn' not in self.allowed_domains:
            self.allowed_domains.append('myweb.cuhk.edu.cn')
        base_path = urlparse(self.base_url).path.strip('/')
        self.base_path_segments = [seg for seg in base_path.split('/') if seg]
        
        # 创建输出目录（原始内容与状态文件分开）
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = Path("data/states")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储已访问的URL
        self.visited_urls: Set[str] = set()
        self.url_data: Dict[str, Dict] = {}
        
        # 统计信息（必须在_load_existing_data之前初始化）
        self.stats = {
            'total_pages': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
        
        # 初始化JSONL文件
        self.jsonl_file = self.output_dir / f"{self.output_slug}.jsonl"
        self.state_file = self.state_dir / f"{self.output_slug}_state.json"
        
        # 是否启用断点续传
        self.resume = resume
        self.to_visit = deque()
        self.pending_urls: Set[str] = set()
        

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
        
        # 挂载 TLS1.2 适配器；若失败则退回普通适配器
        tls_adapter = TLS12HttpAdapter(max_retries=retry_strategy)
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", tls_adapter)
        self.session.mount("https://", tls_adapter)
        logger.info("已挂载 TLS1.2 适配器")

    def _initialize_queue(self):
        """初始化待爬队列"""
        normalized_base = self.normalize_url(self.base_url)
        if normalized_base not in self.visited_urls and normalized_base not in self.pending_urls:
            self.to_visit.append(normalized_base)
            self.pending_urls.add(normalized_base)
        self._save_state()

    def _load_state(self):
        """加载断点续传的队列状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                queued_urls = state.get('queue', [])
                visited = state.get('visited', [])
                for url in visited:
                    self.visited_urls.add(url)
                for url in queued_urls:
                    normalized = self.normalize_url(url)
                    if normalized not in self.visited_urls and normalized not in self.pending_urls:
                        self.to_visit.append(normalized)
                        self.pending_urls.add(normalized)
                logger.info(f"恢复爬取状态: 待爬 {len(self.to_visit)} 个页面")
            except Exception as exc:
                logger.warning(f"加载状态文件失败，将重新初始化队列: {exc}")
        if not self.to_visit:
            self._initialize_queue()

    def _save_state(self):
        """持久化当前的爬取状态，便于中断续传"""
        state_payload = {
            'base_url': self.base_url,
            'visited': list(self.visited_urls),
            'queue': list(self.to_visit),
            'stats': self.stats,
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state_payload, f, ensure_ascii=False, indent=2)

    def _enqueue_url(self, url: str):
        """将新的URL加入待爬队列"""
        normalized = self.normalize_url(url)
        if (normalized not in self.visited_urls 
                and normalized not in self.pending_urls
                and normalized.startswith(self.base_url)
                and self.is_valid_url(normalized)):
            self.to_visit.append(normalized)
            self.pending_urls.add(normalized)

    def is_valid_url(self, url: str) -> bool:
        """检查URL是否有效"""
        parsed = urlparse(url)
        
        # 检查域名
        if parsed.netloc not in self.allowed_domains:
            return False
        
        # 排除文件类型
        excluded_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', 
                              '.zip', '.rar', '.jpg', '.jpeg', '.png', 
                              '.gif', '.mp4', '.mp3', '.exe'}
        if any(url.lower().endswith(ext) for ext in excluded_extensions):
            return False
        
        # 排除英文页面（/en 开头）
        path_lower = parsed.path.lower()
        if path_lower == '/en' or path_lower.startswith('/en/') or path_lower.startswith('/event/') or path_lower == '/event':
            return False
        
        # 排除常见的不需要爬取的路径
        excluded_paths = ['/javascript:', '/mailto:', '/tel:', '#']
        if any(url.startswith(path) for path in excluded_paths):
            return False
        
        return True

    def normalize_url(self, url: str) -> str:
        """规范化URL"""
        # 移除fragment
        url, _ = urldefrag(url)
        # 移除末尾斜杠（除了根路径）
        if url.endswith('/') and url != self.base_url + '/':
            url = url.rstrip('/')
        return url

    def safe_get(self, url, stream=False, timeout=30):
        """带重试与 SSL 回退的 GET 请求"""
        response = self.session.get(url, stream=stream, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        return response

    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """提取页面内容"""
        # 提取标题（清理特殊字符）
        title = ''
        if soup.title:
            title = soup.title.get_text(strip=True)
        elif soup.find('h1'):
            title = soup.find('h1').get_text(strip=True)
        elif soup.find('h2'):
            title = soup.find('h2').get_text(strip=True)
        
        # 清理标题中的特殊字符和控制字符
        if title:
            title = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', title)  # 移除控制字符
            # 尝试修复编码问题（如果标题看起来像乱码）
            title = self._fix_encoding_issues(title)
            title = title.strip()
        
        # 提取主内容
        # 尝试找到主要内容区域
        main_content = ''
        content_selectors = [
            'main', 'article', '.content', '.main-content',
            '#content', '#main', '.post-content', '.entry-content',
            '.container', '.wrapper', 'body'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                text = content_elem.get_text(separator='\n', strip=True)
                if len(text) > 50:  # 确保有足够内容
                    main_content = text
                    break
        
        # 如果没找到，使用body（但保留更多内容）
        if not main_content or len(main_content) < 50:
            # 创建副本以避免修改原始soup
            soup_copy = BeautifulSoup(str(soup), 'html.parser')
            # 移除script和style，但保留其他内容
            for script in soup_copy(["script", "style", "noscript"]):
                script.decompose()
            main_content = soup_copy.get_text(separator='\n', strip=True)
            
            # 清理空白行和控制字符
            lines = []
            for line in main_content.split('\n'):
                cleaned_line = line.strip()
                # 移除控制字符（保留可打印字符）
                cleaned_line = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', cleaned_line)
                if cleaned_line:
                    lines.append(cleaned_line)
            main_content = '\n'.join(lines)
        
        # 提取所有文本段落（清理控制字符）
        paragraphs = []
        for p in soup.find_all(['p', 'div']):
            text = p.get_text(strip=True)
            if text and len(text) > 20:  # 过滤太短的文本
                # 移除控制字符
                text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
                if text:  # 清理后可能为空
                    paragraphs.append(text)
        
        # 提取链接
        links = []
        for a in soup.find_all('a', href=True):
            href = a.get('href', '')
            link_text = a.get_text(strip=True)
            if href and link_text:
                full_url = urljoin(url, href)
                links.append({
                    'text': link_text,
                    'url': self.normalize_url(full_url)
                })
        
        # 提取元数据
        meta_description = ''
        meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_desc_tag:
            meta_description = meta_desc_tag.get('content', '').strip()
        
        return {
            'title': title,
            'content': main_content,
            'paragraphs': paragraphs[:50],  # 限制段落数量
            'links': links,
            'meta_description': meta_description,
            'word_count': len(main_content.split())
        }

    def crawl_page(self, url: str) -> bool:
        """爬取单个页面"""
        url = self.normalize_url(url)
        current_depth = self._relative_depth(url)
        
        # 检查是否已访问
        if url in self.visited_urls:
            return True
        
        # 检查深度
        if current_depth > self.max_depth:
            logger.debug(f"跳过（超过最大深度）: {url}")
            return False
        
        # 检查URL有效性
        if not self.is_valid_url(url):
            logger.debug(f"跳过（无效URL）: {url}")
            return False
        
        try:
            logger.info(f"[深度 {current_depth}] 爬取: {url}")
            
            # 使用安全GET方法（带SSL回退）
            response = self.safe_get(url, timeout=30)
            
            # 检查状态码
            if response.status_code >= 400:
                logger.warning(f"HTTP状态码 {response.status_code}: {url}")
                if response.status_code == 404:
                    return False
            
            response.raise_for_status()
            
            # 检查内容类型
            content_type = response.headers.get('Content-Type', '')
            if content_type and 'text/html' not in content_type:
                logger.debug(f"跳过（非HTML，类型: {content_type}）: {url}")
                return False
            
            # 获取响应文本（处理编码问题）
            html_content = response.content.decode('utf-8', errors='replace')
            
            # 如果包含大量乱码，尝试重新检测编码
            if len(html_content) > 100:
                non_ascii_count = sum(1 for c in html_content[:1000] if ord(c) > 127)
                weird_chars = sum(1 for c in html_content[:1000] 
                                if ord(c) > 127 and not (0x4e00 <= ord(c) <= 0x9fff))
                
                if non_ascii_count > 0 and weird_chars > non_ascii_count * 0.3:
                    # 尝试chardet检测
                    try:
                        import chardet
                        detected = chardet.detect(response.content)
                        encoding = detected.get('encoding', 'utf-8')
                        if encoding and encoding.lower() != 'utf-8':
                            html_content = response.content.decode(encoding, errors='replace')
                    except:
                        # 尝试常见中文编码
                        for enc in ['gbk', 'gb2312', 'big5']:
                            html_content = response.content.decode(enc, errors='replace')
                            break
            
            # 解析HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 提取内容
            page_data = self.extract_content(soup, url)
            page_data.update({
                'url': url,
                'depth': current_depth,
                'crawl_time': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # 保存数据
            self.visited_urls.add(url)
            self.url_data[url] = page_data
            self.stats['success'] += 1
            
            # 及时保存：每爬取一个页面就追加到JSONL文件
            self.append_to_jsonl(page_data)
            
            # 提取所有链接
            links_to_crawl = []
            for a in soup.find_all('a', href=True):
                href = a.get('href', '')
                full_url = urljoin(url, href)
                normalized_url = self.normalize_url(full_url)
                
                if (self.is_valid_url(normalized_url) and 
                    normalized_url not in self.visited_urls and
                    normalized_url.startswith(self.base_url)):
                    links_to_crawl.append(normalized_url)
            
            # 记录下一批待爬链接（仅在允许的深度范围内）
            if current_depth < self.max_depth:
                for link_url in set(links_to_crawl):
                    self._enqueue_url(link_url)
            
            return True
            
        except Exception as e:
            logger.error(f"处理失败 {url}: {e}")
            self.stats['failed'] += 1
            return False


    def _filename_from_url(self, url: str, extension: str) -> str:
        """根据URL生成安全的文件名"""
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split('/') if p]
        filename = '_'.join(path_parts) if path_parts else 'index'
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        if not filename:
            filename = 'index'
        if not filename.endswith(extension):
            filename += extension
        return filename

    def get_structured_data(self) -> Dict:
        """获取结构化数据（用于后续转换）"""
        return {
            'base_url': self.base_url,
            'total_pages': len(self.url_data),
            'pages': list(self.url_data.values())
        }

    def append_to_jsonl(self, page_data: Dict):
        """追加单条数据到JSONL文件（及时保存）"""
        cleaned_data = self._prepare_json_payload(self._clean_data(page_data))
        with open(self.jsonl_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(cleaned_data, ensure_ascii=False) + '\n')
            f.flush()
        
        if self.stats['success'] % 10 == 0:
            logger.debug(f"已保存 {self.stats['success']} 个页面")
    
    def _fix_encoding_issues(self, text: str) -> str:
        """尝试修复编码问题"""
        if not text or len(text) < 10:
            return text
        
        # 检查是否包含乱码字符（非中文、非ASCII）
        has_garbled = any(ord(c) > 127 and not (0x4e00 <= ord(c) <= 0x9fff) 
                         and not (0x3400 <= ord(c) <= 0x4dbf) for c in text[:100])
        
        if has_garbled:
            fixed = text.encode('latin1', errors='ignore').decode('utf-8', errors='ignore')
            if fixed and len(fixed) > len(text) * 0.5:
                return fixed
        
        return text
    
    def _clean_data(self, data: Dict) -> Dict:
        """清理数据中的控制字符和无效字符"""
        cleaned = {}
        for key, value in data.items():
            if isinstance(value, str):
                # 先尝试修复编码问题
                value = self._fix_encoding_issues(value)
                # 移除控制字符（保留换行符、制表符等常用字符）
                cleaned_value = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', value)
                cleaned[key] = cleaned_value
            elif isinstance(value, list):
                # 处理列表
                cleaned_list = []
                for item in value:
                    if isinstance(item, str):
                        item = self._fix_encoding_issues(item)
                        cleaned_item = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', item)
                        cleaned_list.append(cleaned_item)
                    elif isinstance(item, dict):
                        cleaned_list.append(self._clean_data(item))
                    else:
                        cleaned_list.append(item)
                cleaned[key] = cleaned_list
            else:
                cleaned[key] = value
        return cleaned
    
    def _prepare_json_payload(self, data: Dict) -> Dict:
        """准备输出到JSON/JSONL的payload，移除不需要的字段"""
        payload = dict(data)
        payload.pop('links', None)
        payload.pop('paragraphs', None)
        return payload

    def _relative_depth(self, url: str) -> int:
        """计算URL相对于基准路径的深度"""
        parsed = urlparse(url)
        segments = [seg for seg in parsed.path.strip('/').split('/') if seg]
        base_len = len(self.base_path_segments)
        relative = len(segments) - base_len
        return max(0, relative)
    

    def save_sitemap(self):
        """保存网站地图"""
        sitemap_file = self.output_dir / f"{self.output_slug}_sitemap.json"
        
        sitemap = {
            'base_url': self.base_url,
            'total_pages': len(self.url_data),
            'structure': {}
        }
        
        # 按路径组织
        for url, data in self.url_data.items():
            parsed = urlparse(url)
            path_parts = parsed.path.strip('/').split('/')
            
            current = sitemap['structure']
            for part in path_parts:
                if part:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
            
            current['_url'] = url
            current['_title'] = data['title']
        
        with open(sitemap_file, 'w', encoding='utf-8') as f:
            json.dump(sitemap, f, ensure_ascii=False, indent=2)
        
        logger.info(f"网站地图已保存到: {sitemap_file}")

    def run(self):
        """运行爬虫"""
        logger.info(f"开始爬取: {self.base_url}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"最大深度: {self.max_depth}")
        logger.info(f"最大网页数量: {self.max_pages}")
        
        start_time = time.time()
    
        self._load_state()
        if not self.to_visit:
            logger.info("没有待爬取的URL，直接退出")
            return
        
        # 开始爬取
        while self.to_visit:
            # 检查是否达到最大网页数量限制
            if self.stats['success'] >= self.max_pages:
                logger.info(f"已达到最大网页数量限制 ({self.max_pages})，停止爬取")
                break
            
            current_url = self.to_visit.popleft()
            self.pending_urls.discard(current_url)
            
            if current_url in self.visited_urls:
                self.stats['skipped'] += 1
                continue
            
            self.crawl_page(current_url)
            self._save_state()
            
            if self.delay:
                time.sleep(self.delay)
        
        elapsed_time = time.time() - start_time
        
        # 打印统计信息
        logger.info("\n" + "="*50)
        logger.info("爬取完成！")
        logger.info(f"总耗时: {elapsed_time:.2f}秒")
        logger.info(f"成功: {self.stats['success']}")
        logger.info(f"失败: {self.stats['failed']}")
        logger.info(f"跳过: {self.stats['skipped']}")
        logger.info(f"总计: {len(self.visited_urls)}")
        logger.info("="*50)

        logger.info("\n所有数据已保存！")
        logger.info(f"JSONL文件（实时保存）: {self.jsonl_file}")


def _build_slug(base_url: str) -> str:
    parsed = urlparse(base_url)
    domain_slug = parsed.netloc.lower().replace('.', '_')
    path_segs = [seg for seg in parsed.path.strip('/').split('/') if seg]
    if path_segs:
        return f"{domain_slug}_{'_'.join(path_segs)}"
    return domain_slug


def _run_crawler_task(task_config):
    base_url = task_config['base_url']
    crawler = WebsiteCrawler(
        base_url=base_url,
        output_slug=task_config['slug'],
        max_depth=task_config['depth'],
        delay=task_config['delay'],
        max_pages=task_config['max_pages']
    )
    crawler.run()


def _parse_urls_argument(urls_arg):
    if not urls_arg:
        return []
    if (isinstance(urls_arg, list)
            and all(isinstance(item, str) for item in urls_arg)
            and any(item.startswith('http') for item in urls_arg if item)):
        parsed_list = []
        for item in urls_arg:
            cleaned = item.strip()
            if cleaned:
                parsed_list.append(cleaned.rstrip('/'))
        return parsed_list
    candidate = ''.join(urls_arg) if isinstance(urls_arg, list) else str(urls_arg)
    candidate = candidate.strip()
    if not candidate:
        return []
    try:
        evaluated = ast.literal_eval(candidate)
        if isinstance(evaluated, (list, tuple)):
            return [
                str(item).strip().rstrip('/')
                for item in evaluated
                if str(item).strip()
            ]
        if isinstance(evaluated, str):
            cleaned = evaluated.strip().rstrip('/')
            return [cleaned] if cleaned else []
    except (ValueError, SyntaxError):
        pass
    cleaned = candidate.rstrip('/')
    return [cleaned] if cleaned else []


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--urls',
        type=list,
        default=DEFAULT_SEED_URLS
    )
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--delay', type=float, default=0.00001)
    parser.add_argument('--max-pages', type=int, default=2000, help='最大爬取网页数量')
    parser.add_argument('--workers', type=int, default=None)
    
    args = parser.parse_args()
    
    url_candidates = _parse_urls_argument(args.urls)
    
    # 去重同时保持顺序
    seen = set()
    base_urls = []
    for url in url_candidates:
        normalized = url.rstrip('/')
        if normalized and normalized not in seen:
            seen.add(normalized)
            base_urls.append(normalized)
    
    tasks = [{
        'base_url': base_url,
        'slug': _build_slug(base_url),
        'depth': args.depth,
        'delay': args.delay,
        'max_pages': args.max_pages
    } for base_url in base_urls]
    
    if len(tasks) == 1:
        _run_crawler_task(tasks[0])
        return
    
    worker_count = args.workers or min(len(tasks), cpu_count())
    worker_count = max(1, min(worker_count, len(tasks)))
    logger.info(f"将使用 {worker_count} 个进程并行爬取 {len(tasks)} 个站点")
    
    with Pool(processes=worker_count) as pool:
        pool.map(_run_crawler_task, tasks)


if __name__ == '__main__':
    main()

