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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebExtractor")

class WebExtractor:
    def __init__(
        self, 
        include_images: bool = False, 
        timeout: int = 30,
        max_text_length: int = 10000,
        min_content_length: int = 100,  # Minimum content length to consider valid
        pytesseract_path: Optional[str] = None
    ):
        self.include_images = include_images
        self.timeout = timeout
        self.max_text_length = max_text_length
        self.min_content_length = min_content_length
        
        # Configure pytesseract path if provided
        if pytesseract_path:
            pytesseract.pytesseract.tesseract_cmd = pytesseract_path
    
    async def extract_information(self, url: str, title: str = "", snippet: str = "") -> Dict[str, Any]:
        """Extract detailed information from a URL"""
        logger.info(f"Extracting information from: {url}")
        
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
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get("content-type", "")
                
                if "text/html" in content_type:
                    # Process HTML content
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Extract or update title if not provided
                    if not title and soup.title:
                        result["title"] = soup.title.string.strip() if soup.title.string else ""
                    
                    # Extract metadata
                    result["metadata"] = self._extract_metadata(soup)
                    
                    # Extract main content
                    result["content"] = self._extract_main_content(soup)
                    
                    # Check if content is valid
                    if result["content"] and len(result["content"].strip()) >= self.min_content_length:
                        result["has_valid_content"] = True
                    
                    # If content is not valid, try harder to find content
                    if not result["has_valid_content"]:
                        result["content"] = self._extract_fallback_content(soup)
                        if result["content"] and len(result["content"].strip()) >= self.min_content_length:
                            result["has_valid_content"] = True
                    
                    # Only continue processing if we have valid content
                    if result["has_valid_content"]:
                        # Extract tables if any
                        result["tables"] = self._extract_tables(soup)
                        
                        # Extract images if requested
                        if self.include_images:
                            result["images"] = await self._extract_images(soup, url, client)
                        
                        # Generate key points
                        result["key_points"] = self._generate_key_points(result["content"])
                    
                elif "application/json" in content_type:
                    # Process JSON content
                    json_data = response.json()
                    result["content"] = json.dumps(json_data, indent=2)
                    result["key_points"] = self._summarize_json(json_data)
                    result["has_valid_content"] = True
                    
                elif "image/" in content_type and self.include_images:
                    # Process image content
                    image_data = response.content
                    result["images"] = [{
                        "src": url,
                        "alt": "Direct image",
                        "text": await self._extract_text_from_image(image_data)
                    }]
                    result["has_valid_content"] = True
                
                elif "text/" in content_type:
                    # Process plain text content
                    result["content"] = response.text[:self.max_text_length]
                    if result["content"] and len(result["content"].strip()) >= self.min_content_length:
                        result["has_valid_content"] = True
                        result["key_points"] = self._generate_key_points(result["content"])
                
                return result
                
        except Exception as e:
            logger.error(f"Error extracting information from {url}: {str(e)}")
            result["error"] = str(e)
            return result
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract metadata from HTML"""
        metadata = {}
        
        # Extract meta tags
        for meta in soup.find_all("meta"):
            name = meta.get("name") or meta.get("property")
            content = meta.get("content")
            if name and content:
                metadata[name] = content
        
        # Extract schema.org structured data
        for script in soup.find_all("script", {"type": "application/ld+json"}):
            try:
                if script.string:
                    json_data = json.loads(script.string)
                    metadata["structured_data"] = json_data
            except Exception as e:
                logger.error(f"Error parsing structured data: {e}")
        
        return metadata
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract the main content from HTML"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Try to find main content in common containers
        content_selectors = [
            "article", "main", ".content", "#content", ".post", ".article", 
            ".entry-content", "#main-content", ".main-content"
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                return content_elem.get_text(separator=" ", strip=True)[:self.max_text_length]
        
        # If no content found, use the body text
        if soup.body:
            return soup.body.get_text(separator=" ", strip=True)[:self.max_text_length]
        
        return ""
    
    def _extract_fallback_content(self, soup: BeautifulSoup) -> str:
        """Try harder to extract content when main methods fail"""
        # Create a copy of the soup to manipulate
        content_soup = BeautifulSoup(str(soup), "html.parser")
        
        # Remove clearly non-content elements
        for element in content_soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "form"]):
            element.decompose()
        
        # Look for elements with significant text content
        candidates = []
        
        for elem in content_soup.find_all(["div", "section", "p"]):
            text = elem.get_text(strip=True)
            if len(text) > self.min_content_length:
                # Calculate text density (text length / HTML length)
                text_density = len(text) / (len(str(elem)) + 1)  # +1 to avoid division by zero
                candidates.append((elem, len(text), text_density))
        
        # Sort candidates by text length and density
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Get the best candidate
        if candidates:
            best_elem = candidates[0][0]
            return best_elem.get_text(separator=" ", strip=True)[:self.max_text_length]
        
        # If still no content, try getting all paragraph text
        paragraphs = content_soup.find_all("p")
        if paragraphs:
            texts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20]
            if texts:
                return " ".join(texts)[:self.max_text_length]
        
        return ""
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[List[List[str]]]:
        """Extract tables from HTML"""
        tables = []
        
        for table in soup.find_all("table"):
            table_data = []
            rows = table.find_all("tr")
            
            for row in rows:
                row_data = []
                cells = row.find_all(["td", "th"])
                
                for cell in cells:
                    row_data.append(cell.get_text(strip=True))
                
                if row_data:
                    table_data.append(row_data)
            
            if table_data:
                tables.append(table_data)
        
        return tables
    
    async def _extract_images(self, soup: BeautifulSoup, base_url: str, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
        """Extract images and their text content"""
        images = []
        seen_urls = set()
        
        for img in soup.find_all("img"):
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
            
            try:
                # Download image
                img_response = await client.get(src)
                img_response.raise_for_status()
                
                # Process image only if it's an actual image
                content_type = img_response.headers.get("content-type", "")
                if "image/" in content_type:
                    image_data = img_response.content
                    
                    # Extract text using OCR if the image is large enough
                    img_text = ""
                    try:
                        image = Image.open(io.BytesIO(image_data))
                        width, height = image.size
                        
                        # Only process images that are reasonably large
                        if width > 100 and height > 100:
                            img_text = await self._extract_text_from_image(image_data)
                    except Exception as e:
                        logger.error(f"Error processing image: {e}")
                    
                    images.append({
                        "src": src,
                        "alt": alt,
                        "text": img_text
                    })
                    
                    # Limit to 5 images to avoid long processing times
                    if len(images) >= 5:
                        break
            
            except Exception as e:
                logger.error(f"Error fetching image {src}: {e}")
        
        return images
    
    async def _extract_text_from_image(self, image_data: bytes) -> str:
        """Extract text from image using OCR"""
        try:
            # Run OCR in a thread pool to avoid blocking
            image = Image.open(io.BytesIO(image_data))
            
            # Run in thread pool executor
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None, lambda: pytesseract.image_to_string(image)
            )
            
            return text.strip()
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    def _generate_key_points(self, text: str) -> List[str]:
        """Generate key points from text"""
        if not text:
            return []
        
        # Simple extraction of sentences ending with periods
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        
        # Filter sentences by length and content
        good_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Only include non-trivial sentences
            if len(sentence) > 20 and len(sentence.split()) > 5:
                good_sentences.append(sentence)
        
        # Select up to 5 key sentences
        step = max(1, len(good_sentences) // 5)
        key_points = good_sentences[::step][:5]
        
        return key_points
    
    def _summarize_json(self, json_data: Any) -> List[str]:
        """Extract key points from JSON data"""
        points = []
        
        if isinstance(json_data, dict):
            # For objects, include top-level keys and some values
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
                    keys = ", ".join(list(sample.keys())[:5])
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
            text = await self._extract_text_from_image(image_data)
            
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