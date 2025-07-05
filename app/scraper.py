"""
Advanced web scraper for building knowledge base.

Features:
- Depth-first crawling within domain
- Binary file detection and skipping  
- HTML cleaning with trafilatura
- Content persistence as .md files
- Metadata JSON storage
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
from typing import Set, Dict, Optional, List
import mimetypes

import requests
from bs4 import BeautifulSoup
import trafilatura


class WebScraper:
    """Advanced web scraper with depth-first crawling capabilities."""
    
    def __init__(self, max_depth: int = 3, delay: float = 1.0, respect_robots: bool = True):
        self.max_depth = max_depth
        self.delay = delay
        self.respect_robots = respect_robots
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (WebScraper/1.0; +https://example.com/bot)'
        })
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Binary file extensions to skip
        self.binary_extensions = {
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.zip', '.tar', '.gz', '.rar', '.7z',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp',
            '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
            '.exe', '.dmg', '.pkg', '.deb', '.rpm'
        }
        
        # MIME types to skip
        self.binary_mimes = {
            'application/pdf', 'application/msword', 'application/vnd.ms-excel',
            'application/vnd.ms-powerpoint', 'application/zip', 'application/octet-stream',
            'image/', 'video/', 'audio/'
        }
    
    def is_binary_url(self, url: str) -> bool:
        """Check if URL points to a binary file."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Check file extension
        for ext in self.binary_extensions:
            if path.endswith(ext):
                return True
        
        # Check MIME type via HEAD request
        try:
            response = self.session.head(url, timeout=10, allow_redirects=True)
            content_type = response.headers.get('content-type', '').lower()
            
            for mime_prefix in self.binary_mimes:
                if content_type.startswith(mime_prefix):
                    return True
                    
        except Exception as e:
            self.logger.warning(f"Could not check MIME type for {url}: {e}")
        
        return False
    
    def can_fetch(self, url: str, domain: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        if not self.respect_robots:
            return True
            
        try:
            robots_url = f"https://{domain}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            return rp.can_fetch(self.session.headers['User-Agent'], url)
        except Exception as e:
            self.logger.warning(f"Could not check robots.txt for {domain}: {e}")
            return True
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and query parameters."""
        parsed = urlparse(url)
        # Remove fragment and normalize
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            ''  # Remove fragment
        ))
        return normalized.rstrip('/')
    
    def extract_links(self, html_content: str, base_url: str, target_domain: str) -> List[str]:
        """Extract internal links from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            normalized_url = self.normalize_url(absolute_url)
            
            # Only include links within the target domain
            parsed = urlparse(normalized_url)
            if parsed.netloc == target_domain and normalized_url not in self.visited_urls:
                links.append(normalized_url)
        
        return links
    
    def clean_content(self, html_content: str, url: str) -> Optional[Dict[str, str]]:
        """Extract and clean content from HTML using trafilatura."""
        try:
            # Use trafilatura for better content extraction
            extracted = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=True,
                include_links=True,
                url=url
            )
            
            if not extracted:
                self.logger.warning(f"No content extracted from {url}")
                return None
            
            # Also extract metadata
            metadata = trafilatura.extract_metadata(html_content)
            
            return {
                'content': extracted,
                'title': metadata.title if metadata and metadata.title else self.extract_title_fallback(html_content),
                'description': metadata.description if metadata and metadata.description else '',
                'author': metadata.author if metadata and metadata.author else '',
                'date': metadata.date if metadata and metadata.date else ''
            }
            
        except Exception as e:
            self.logger.error(f"Error cleaning content from {url}: {e}")
            return None
    
    def extract_title_fallback(self, html_content: str) -> str:
        """Fallback method to extract title from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find('title')
            if title_tag:
                return title_tag.get_text().strip()
            
            # Try h1 as fallback
            h1_tag = soup.find('h1')
            if h1_tag:
                return h1_tag.get_text().strip()
                
            return "Untitled"
        except Exception:
            return "Untitled"
    
    def save_content(self, url: str, content_data: Dict[str, str], output_dir: Path) -> None:
        """Save content as markdown file with metadata JSON."""
        try:
            # Create safe filename from URL
            parsed = urlparse(url)
            safe_path = parsed.path.strip('/').replace('/', '_')
            if not safe_path:
                safe_path = 'index'
            
            # Remove or replace unsafe characters
            safe_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.')
            safe_path = ''.join(c if c in safe_chars else '_' for c in safe_path)
            
            # Ensure reasonable filename length
            if len(safe_path) > 100:
                safe_path = safe_path[:100]
            
            base_filename = f"{parsed.netloc}_{safe_path}"
            
            # Save markdown content
            md_path = output_dir / f"{base_filename}.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# {content_data['title']}\n\n")
                if content_data.get('description'):
                    f.write(f"*{content_data['description']}*\n\n")
                f.write(f"**Source:** {url}\n\n")
                f.write("---\n\n")
                f.write(content_data['content'])
            
            # Save metadata JSON
            metadata = {
                'title': content_data['title'],
                'url': url,
                'fetched_at': datetime.now().isoformat(),
                'description': content_data.get('description', ''),
                'author': content_data.get('author', ''),
                'date': content_data.get('date', ''),
                'markdown_file': str(md_path)
            }
            
            json_path = output_dir / f"{base_filename}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved content from {url} to {md_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving content from {url}: {e}")
    
    def crawl_url(self, url: str, depth: int, target_domain: str, output_dir: Path) -> List[str]:
        """Crawl a single URL and return discovered links."""
        if depth > self.max_depth or url in self.visited_urls:
            return []
        
        if self.is_binary_url(url):
            self.logger.info(f"Skipping binary file: {url}")
            return []
        
        if not self.can_fetch(url, target_domain):
            self.logger.info(f"Robots.txt disallows crawling: {url}")
            return []
        
        self.visited_urls.add(url)
        
        try:
            self.logger.info(f"Crawling (depth {depth}): {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Check if response is HTML
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('text/html'):
                self.logger.info(f"Skipping non-HTML content: {url} (type: {content_type})")
                return []
            
            # Clean and extract content
            content_data = self.clean_content(response.text, url)
            if content_data:
                self.save_content(url, content_data, output_dir)
            
            # Extract links for further crawling
            links = self.extract_links(response.text, url, target_domain)
            
            # Respect crawl delay
            time.sleep(self.delay)
            
            return links
            
        except Exception as e:
            self.logger.error(f"Error crawling {url}: {e}")
            return []
    
    def crawl(self, start_url: str, output_dir: str = "data/raw") -> None:
        """
        Perform depth-first crawl starting from the given URL.
        
        Args:
            start_url: The starting URL to crawl
            output_dir: Directory to save scraped content
        """
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Parse target domain
        parsed_start = urlparse(start_url)
        target_domain = parsed_start.netloc
        
        self.logger.info(f"Starting crawl of {target_domain} from {start_url}")
        self.logger.info(f"Max depth: {self.max_depth}, Output: {output_path}")
        
        # Initialize crawl queue with start URL
        crawl_queue = [(start_url, 0)]  # (url, depth)
        
        while crawl_queue:
            current_url, depth = crawl_queue.pop(0)  # Depth-first: use pop(0)
            
            # Crawl current URL and get discovered links
            discovered_links = self.crawl_url(current_url, depth, target_domain, output_path)
            
            # Add discovered links to queue for next depth level
            for link in discovered_links:
                if link not in self.visited_urls:
                    crawl_queue.append((link, depth + 1))
        
        self.logger.info(f"Crawl completed. Visited {len(self.visited_urls)} URLs.")
        self.logger.info(f"Content saved to: {output_path}")


def scrape_website(url: str, max_depth: int = 2, delay: float = 1.0) -> None:
    """
    High-level function to scrape a website.
    
    Args:
        url: Starting URL to scrape
        max_depth: Maximum crawl depth
        delay: Delay between requests in seconds
    """
    scraper = WebScraper(max_depth=max_depth, delay=delay)
    scraper.crawl(url)
