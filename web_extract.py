import httpx
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from bs4 import BeautifulSoup
import re
import io
from PIL import Image
import pytesseract
import logging
from urllib.parse import urlparse
import json
import time
import cachetools
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebExtractor")

class WebExtractor:
    def __init__(
        self, 
        include_images: bool = False, 
        timeout: int = 15,
        max_text_length: int = 10000,
        min_content_length: int = 100,  # Minimum content length to consider valid
        pytesseract_path: Optional[str] = None,
        thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None,
        cache_size: int = 100,  # Cache size for results
        connection_pool_size: int = 20  # Connection pool size for HTTP requests
    ):
        self.include_images = include_images
        self.timeout = timeout
        self.max_text_length = max_text_length
        self.min_content_length = min_content_length
        self.connection_pool_size = connection_pool_size
        
        # Use provided thread pool or create a new one
        self.thread_pool = thread_pool or concurrent.futures.ThreadPoolExecutor(max_workers=8)
        
        # Set up a cache for results to avoid redundant requests
        self.cache = cachetools.LRUCache(maxsize=cache_size)
        
        # Configure pytesseract path if provided
        if pytesseract_path:
            pytesseract.pytesseract.tesseract_cmd = pytesseract_path
    
    async def extract_information(self, url: str, title: str = "", snippet: str = "") -> Dict[str, Any]:
        """Extract detailed information from a URL"""
        # Check cache first
        cache_key = f"extract:{url}"
        if cache_key in self.cache:
            logger.info(f"Cache hit for URL: {url}")
            return self.cache[cache_key]
        
        logger.info(f"Extracting information from: {url}")
        start_time = time.time()
        
        result = {
            "url": url,
            "title": title,
            "snippet": snippet,
            "content": "",
            "metadata": {},
            "images": [],
            "tables": [],
            "key_points": [],
            "timestamp": time.time(),
            "has_valid_content": False  # Track if this result has valid content
        }
        
        try:
            # Set up limits for the HTTP client
            limits = httpx.Limits(max_connections=self.connection_pool_size)
            
            async with httpx.AsyncClient(
                timeout=self.timeout, 
                follow_redirects=True,
                limits=limits
            ) as client:
                # Fetch the page with timeout
                response = await client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                })
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get("content-type", "")
                
                if "text/html" in content_type:
                    # Process HTML content in the thread pool for CPU-bound operations
                    loop = asyncio.get_event_loop()
                    
                    # Parse HTML
                    html_content = response.text
                    soup = await loop.run_in_executor(
                        self.thread_pool,
                        lambda: BeautifulSoup(html_content, "html.parser")
                    )
                    
                    # Extract or update title if not provided
                    if not title and soup.title:
                        result["title"] = soup.title.string.strip() if soup.title.string else ""
                    
                    # Extract metadata (light operation)
                    result["metadata"] = await loop.run_in_executor(
                        self.thread_pool,
                        lambda: self._extract_metadata(soup)
                    )
                    
                    # Extract main content (CPU intensive)
                    result["content"] = await loop.run_in_executor(
                        self.thread_pool,
                        lambda: self._extract_main_content(soup)
                    )
                    
                    # Check if content is valid
                    if result["content"] and len(result["content"].strip()) >= self.min_content_length:
                        result["has_valid_content"] = True
                    
                    # If content is not valid, try harder to find content with a different method
                    if not result["has_valid_content"]:
                        result["content"] = await loop.run_in_executor(
                            self.thread_pool,
                            lambda: self._extract_fallback_content(soup)
                        )
                        if result["content"] and len(result["content"].strip()) >= self.min_content_length:
                            result["has_valid_content"] = True
                    
                    # Only continue processing if we have valid content to save time
                    if result["has_valid_content"]:
                        # Process tables and key points in parallel
                        tables_task = loop.run_in_executor(
                            self.thread_pool,
                            lambda: self._extract_tables(soup)
                        )
                        
                        key_points_task = loop.run_in_executor(
                            self.thread_pool,
                            lambda: self._generate_key_points(result["content"])
                        )
                        
                        # Extract images if requested - do this in parallel
                        images_task = None
                        if self.include_images:
                            images_task = self._extract_images(soup, url, client)
                        
                        # Wait for all tasks to complete
                        result["tables"] = await tables_task
                        result["key_points"] = await key_points_task
                        
                        if images_task:
                            result["images"] = await images_task
                    
                    # Add extraction time for performance monitoring
                    result["extraction_time"] = time.time() - start_time
                    
                elif "application/json" in content_type:
                    # Process JSON content in thread pool
                    loop = asyncio.get_event_loop()
                    
                    json_data = response.json()
                    content_task = loop.run_in_executor(
                        self.thread_pool,
                        lambda: json.dumps(json_data, indent=2)
                    )
                    
                    key_points_task = loop.run_in_executor(
                        self.thread_pool,
                        lambda: self._summarize_json(json_data)
                    )
                    
                    # Get results
                    result["content"] = await content_task
                    result["key_points"] = await key_points_task
                    result["has_valid_content"] = True
                    
                elif "image/" in content_type and self.include_images:
                    # Process image content
                    image_data = response.content
                    
                    # Use thread pool for OCR (CPU-intensive)
                    loop = asyncio.get_event_loop()
                    img_text = await loop.run_in_executor(
                        self.thread_pool,
                        lambda: self._process_image_sync(image_data)
                    )
                    
                    result["images"] = [{
                        "src": url,
                        "alt": "Direct image",
                        "text": img_text
                    }]
                    result["has_valid_content"] = True
                
                elif "text/" in content_type:
                    # Process plain text content
                    text_content = response.text[:self.max_text_length]
                    result["content"] = text_content
                    
                    if text_content and len(text_content.strip()) >= self.min_content_length:
                        result["has_valid_content"] = True
                        
                        # Generate key points in thread pool
                        loop = asyncio.get_event_loop()
                        result["key_points"] = await loop.run_in_executor(
                            self.thread_pool,
                            lambda: self._generate_key_points(text_content)
                        )
                
                # Cache the result if it's valid
                if result["has_valid_content"]:
                    self.cache[cache_key] = result
                
                return result
                
        except Exception as e:
            logger.error(f"Error extracting information from {url}: {str(e)}")
            result["error"] = str(e)
            return result
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract metadata from HTML"""
        metadata = {}
        
        # Extract meta tags (limit to important ones)
        meta_tags = ["description", "keywords", "author", "og:title", "og:description", "og:image", "twitter:title", "twitter:description"]
        
        for meta in soup.find_all("meta"):
            name = meta.get("name") or meta.get("property")
            content = meta.get("content")
            if name and content and (name in meta_tags or name.startswith("og:") or name.startswith("twitter:")):
                metadata[name] = content
        
        # Extract schema.org structured data (only first one to save time)
        structured_data_found = False
        for script in soup.find_all("script", {"type": "application/ld+json"}, limit=1):
            try:
                if script.string and not structured_data_found:
                    json_data = json.loads(script.string)
                    metadata["structured_data"] = json_data
                    structured_data_found = True
            except Exception as e:
                logger.error(f"Error parsing structured data: {e}")
        
        return metadata
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract the main content from HTML using optimized selectors"""
        # Create a copy of the soup to manipulate for better performance
        content_soup = BeautifulSoup(str(soup), "html.parser")
        
        # Remove script and style elements (limit for speed)
        for script in content_soup(["script", "style", "nav", "footer", "header", "aside"], limit=50):
            script.decompose()
        
        # Try to find main content in common containers - short list for speed
        content_selectors = [
            "article", "main", ".content", "#content", ".post", ".article"
        ]
        
        for selector in content_selectors:
            content_elem = content_soup.select_one(selector)
            if content_elem:
                return content_elem.get_text(separator=" ", strip=True)[:self.max_text_length]
        
        # If no content found in common containers, use body text (limit size)
        if content_soup.body:
            return content_soup.body.get_text(separator=" ", strip=True)[:self.max_text_length]
        
        return ""
    
    def _extract_fallback_content(self, soup: BeautifulSoup) -> str:
        """Try harder to extract content when main methods fail - optimized version"""
        # Look for elements with significant text content (limit to 20 biggest divs)
        divs = soup.find_all("div", limit=20)
        best_div = None
        most_text = 0
        
        for div in divs:
            text = div.get_text(strip=True)
            if len(text) > most_text:
                most_text = len(text)
                best_div = div
        
        if best_div and most_text >= self.min_content_length:
            return best_div.get_text(separator=" ", strip=True)[:self.max_text_length]
        
        # If still no content, get all paragraph text
        paragraphs = soup.find_all("p", limit=20)
        if paragraphs:
            text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
            return text[:self.max_text_length]
        
        return ""
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[List[List[str]]]:
        """Extract tables from HTML - optimized to process fewer tables"""
        tables = []
        
        # Limit to 5 tables for performance
        for table in soup.find_all("table", limit=5):
            table_data = []
            rows = table.find_all("tr", limit=20)  # Limit to 20 rows
            
            for row in rows:
                row_data = []
                cells = row.find_all(["td", "th"], limit=10)  # Limit to 10 columns
                
                for cell in cells:
                    row_data.append(cell.get_text(strip=True))
                
                if row_data:
                    table_data.append(row_data)
            
            if table_data:
                tables.append(table_data)
        
        return tables
    
    async def _extract_images(self, soup: BeautifulSoup, base_url: str, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
        """Extract images and their text content with improved efficiency"""
        images = []
        seen_urls = set()
        
        # Only process the first 5 images for performance
        img_tags = soup.find_all("img", limit=5)
        
        # Create tasks for parallel image processing
        tasks = []
        
        for img in img_tags:
            src = img.get("src", "")
            alt = img.get("alt", "")
            
            if not src:
                continue
            
            # Make URL absolute
            if not src.startswith(("http://", "https://")):
                src = urljoin(base_url, src)
            
            # Skip if already processed or if it's a tiny icon/button
            if src in seen_urls or "icon" in src.lower() or "logo" in src.lower():
                continue
            
            seen_urls.add(src)
            
            # Create task for this image
            tasks.append(self._process_image(client, src, alt))
        
        # Process all images in parallel
        image_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter valid results
        for result in image_results:
            if isinstance(result, Exception):
                continue
            if result:  # Skip None results
                images.append(result)
        
        return images
    
    async def _process_image(self, client: httpx.AsyncClient, src: str, alt: str) -> Optional[Dict[str, Any]]:
        """Process a single image"""
        try:
            # Check cache
            cache_key = f"image:{src}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Download image with timeout
            try:
                img_response = await client.get(src, timeout=5)  # Shorter timeout for images
                img_response.raise_for_status()
            except Exception as e:
                logger.error(f"Error fetching image {src}: {e}")
                return None
            
            # Process image only if it's an actual image
            content_type = img_response.headers.get("content-type", "")
            if "image/" not in content_type:
                return None
            
            image_data = img_response.content
            
            # Extract text using OCR if the image is large enough
            img_text = ""
            try:
                loop = asyncio.get_event_loop()
                
                # First check the image size
                image_size = await loop.run_in_executor(
                    self.thread_pool,
                    lambda: self._check_image_size(image_data)
                )
                
                width, height = image_size
                
                # Only process images that are reasonably large and not too large (to avoid memory issues)
                if width > 100 and height > 100 and width * height < 4000000:  # Limit to 4MP
                    img_text = await loop.run_in_executor(
                        self.thread_pool,
                        lambda: self._process_image_sync(image_data)
                    )
            except Exception as e:
                logger.error(f"Error processing image: {e}")
            
            result = {
                "src": src,
                "alt": alt,
                "text": img_text
            }
            
            # Cache the result
            self.cache[cache_key] = result
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing image {src}: {e}")
            return None
    
    def _check_image_size(self, image_data: bytes) -> Tuple[int, int]:
        """Check the dimensions of an image"""
        try:
            image = Image.open(io.BytesIO(image_data))
            return image.size
        except Exception:
            return (0, 0)
    
    def _process_image_sync(self, image_data: bytes) -> str:
        """Process an image synchronously for OCR"""
        try:
            image = Image.open(io.BytesIO(image_data))
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    def _generate_key_points(self, text: str) -> List[str]:
        """Generate key points from text - optimized for speed"""
        if not text or len(text) < self.min_content_length:
            return []
        
        # Simple extraction of sentences ending with periods
        sentences = re.findall(r'[^.!?]+[.!?]', text)
        
        # Filter sentences by length and content
        good_sentences = []
        for sentence in sentences[:30]:  # Only process first 30 sentences
            sentence = sentence.strip()
            # Only include non-trivial sentences
            if len(sentence) > 20 and len(sentence.split()) > 5:
                good_sentences.append(sentence)
        
        # Select up to 5 key sentences
        if len(good_sentences) <= 5:
            return good_sentences
        
        # Pick 5 well-distributed sentences
        step = len(good_sentences) // 5
        return [good_sentences[i * step] for i in range(5)]
    
    def _summarize_json(self, json_data: Any) -> List[str]:
        """Extract key points from JSON data - simplified for speed"""
        points = []
        
        if isinstance(json_data, dict):
            # For objects, include top-level keys and some values (limit to 5)
            for key, value in list(json_data.items())[:5]:
                if isinstance(value, (str, int, float, bool)):
                    points.append(f"{key}: {value}")
                elif isinstance(value, (list, dict)):
                    points.append(f"{key}: Contains {type(value).__name__} data")
        
        elif isinstance(json_data, list):
            # For arrays, summarize length and sample items
            points.append(f"Contains {len(json_data)} items")
            
            if json_data and len(json_data) > 0:
                sample = json_data[0]
                if isinstance(sample, dict):
                    keys = ", ".join(list(sample.keys())[:3])  # Limit to 3 keys
                    points.append(f"Each item contains keys: {keys}...")
                else:
                    points.append(f"Items are of type: {type(sample).__name__}")
        
        return points
    
    async def extract_from_url(self, url: str) -> Dict[str, Any]:
        """Extract information from a single URL"""
        result = await self.extract_information(url)
        
        # Only return if it has valid content
        if not result.get("has_valid_content", False):
            logger.warning(f"No valid content found at URL: {url}")
            result["error"] = "No valid content could be extracted from this URL"
        
        return result
    
    async def extract_from_image(self, image_data: bytes) -> Dict[str, Any]:
        """Extract text from a single image"""
        try:
            # Process in thread pool
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                self.thread_pool,
                lambda: self._process_image_sync(image_data)
            )
            
            return {
                "success": True,
                "text": text,
                "length": len(text),
                "has_valid_content": len(text.strip()) > self.min_content_length
            }
        except Exception as e:
            logger.error(f"Error extracting from image: {e}")
            return {
                "success": False,
                "error": str(e),
                "has_valid_content": False
            }

# Helper function for resolving relative URLs
def urljoin(base: str, url: str) -> str:
    """Join a base URL and a possibly relative URL to form an absolute URL"""
    if not base.endswith('/'):
        base += '/'
    
    if url.startswith('/'):
        # URL is absolute path on the same domain
        parsed_base = urlparse(base)
        return f"{parsed_base.scheme}://{parsed_base.netloc}{url}"
    elif url.startswith(('http://', 'https://')):
        # URL is already absolute
        return url
    else:
        # URL is relative to the base path
        return base + url