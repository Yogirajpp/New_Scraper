import httpx
import asyncio
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import logging
import random
import time
import cachetools
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebScraper")

class WebScraper:
    def __init__(
        self, 
        max_results: int = 5, 
        search_depth: int = 2, 
        timeout: int = 15,
        exclude_domains: List[str] = None,
        user_agents: List[str] = None,
        min_content_length: int = 100,  # Minimum content length to consider valid
        connection_pool_size: int = 20,  # Connection pool size for HTTP requests
        cache_size: int = 100  # Cache size for results
    ):
        self.max_results = max_results
        self.search_depth = search_depth
        self.timeout = timeout
        self.exclude_domains = exclude_domains or []
        self.min_content_length = min_content_length
        self.connection_pool_size = connection_pool_size
        
        # Set up a cache for results to avoid redundant requests
        self.cache = cachetools.LRUCache(maxsize=cache_size)
        
        # Default user agents list for rotating to avoid detection
        self.user_agents = user_agents or [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
        ]
        
        # Search engines to use
        self.search_engines = [
            {
                "name": "Google",
                "url": "https://www.google.com/search?q={query}&num={num}",
                "result_selector": "div.g",
                "title_selector": "h3",
                "url_selector": "a",
                "snippet_selector": "div.VwiC3b"
            },
            {
                "name": "Bing", 
                "url": "https://www.bing.com/search?q={query}&count={num}",
                "result_selector": "li.b_algo",
                "title_selector": "h2",
                "url_selector": "h2 a",
                "snippet_selector": "div.b_caption p"
            },
            {
                "name": "DuckDuckGo",
                "url": "https://html.duckduckgo.com/html/?q={query}",
                "result_selector": "div.result",
                "title_selector": "h2.result__title",
                "url_selector": "a.result__url",
                "snippet_selector": "a.result__snippet"
            }
        ]
        
        # Flag to track search task completion
        self.search_complete = asyncio.Event()
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Execute a search across multiple search engines and aggregate results"""
        logger.info(f"Searching for: {query}")
        start_time = time.time()
        
        # Check cache first
        cache_key = f"search:{query}:{self.max_results}"
        if cache_key in self.cache:
            logger.info(f"Cache hit for query: {query}")
            return self.cache[cache_key]
        
        # Create shared data structures for collecting results
        all_results = []
        unique_urls = set()
        
        # Semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(self.connection_pool_size)
        
        # Function to get a random user agent
        def get_random_user_agent():
            return random.choice(self.user_agents)
        
        # Create an HTTP client to be shared across all requests
        limits = httpx.Limits(max_connections=self.connection_pool_size)
        async with httpx.AsyncClient(
            timeout=self.timeout, 
            follow_redirects=True,
            limits=limits
        ) as client:
            # Search engines in parallel
            engine_tasks = []
            for engine in self.search_engines:
                engine_tasks.append(
                    self._search_engine(client, query, engine, semaphore, get_random_user_agent)
                )
            
            # Run all search engine tasks concurrently
            engine_results = await asyncio.gather(*engine_tasks, return_exceptions=True)
            
            # Process results
            for results in engine_results:
                if isinstance(results, Exception):
                    logger.error(f"Search engine error: {results}")
                    continue
                
                # Add unique results to the collection
                for result in results:
                    url = result.get("url", "")
                    if url and url not in unique_urls:
                        unique_urls.add(url)
                        all_results.append(result)
            
            # Efficient sorting and limiting
            all_results = sorted(all_results, key=lambda x: len(x.get("snippet", "")), reverse=True)
            desired_count = min(self.max_results * 2, len(all_results))
            limited_results = all_results[:desired_count]
            
            # If search depth > 1, enrich results in parallel
            if self.search_depth > 1 and limited_results:
                # Create enrichment tasks for all results
                enrich_tasks = []
                for result in limited_results:
                    enrich_tasks.append(
                        self._fetch_page_content(client, result, semaphore, get_random_user_agent)
                    )
                
                # Run all enrichment tasks concurrently with a timeout
                enriched_results = await asyncio.gather(*enrich_tasks, return_exceptions=True)
                
                # Filter valid results
                valid_results = []
                for res in enriched_results:
                    if isinstance(res, Exception):
                        logger.error(f"Enrichment error: {res}")
                        continue
                    if res.get("has_content", False):
                        valid_results.append(res)
                
                # If we don't have enough valid results, try more from the remaining candidates
                if len(valid_results) < self.max_results and len(all_results) > desired_count:
                    additional_candidates = all_results[desired_count:desired_count + (self.max_results - len(valid_results))]
                    
                    if additional_candidates:
                        # Create tasks for additional candidates
                        add_tasks = []
                        for result in additional_candidates:
                            add_tasks.append(
                                self._fetch_page_content(client, result, semaphore, get_random_user_agent)
                            )
                        
                        # Process additional candidates
                        add_results = await asyncio.gather(*add_tasks, return_exceptions=True)
                        
                        # Add valid results
                        for res in add_results:
                            if isinstance(res, Exception):
                                continue
                            if res.get("has_content", False):
                                valid_results.append(res)
                
                # Get the final results
                final_results = valid_results[:self.max_results]
                
                # Cache the results
                self.cache[cache_key] = final_results
                
                logger.info(f"Search completed in {time.time() - start_time:.2f}s with {len(final_results)} valid results")
                return final_results
            
            # If search depth is 1, just return the limited results
            logger.info(f"Basic search completed in {time.time() - start_time:.2f}s with {len(limited_results)} results")
            self.cache[cache_key] = limited_results[:self.max_results]
            return limited_results[:self.max_results]
    
    async def _search_engine(
        self, 
        client: httpx.AsyncClient, 
        query: str, 
        engine: Dict[str, str],
        semaphore: asyncio.Semaphore,
        get_user_agent: callable
    ) -> List[Dict[str, Any]]:
        """Execute a search on a specific search engine with resource limiting"""
        async with semaphore:
            results = []
            
            try:
                # Format the search URL - request more results than needed
                search_url = engine["url"].format(query=query, num=self.max_results * 3)
                
                # Use random user agent
                headers = {"User-Agent": get_user_agent()}
                
                # Execute the search request with timeout
                response = await client.get(search_url, headers=headers)
                response.raise_for_status()
                
                # Fast HTML parsing with features limited to what we need
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Extract results
                result_elements = soup.select(engine["result_selector"])
                
                # Process in batch for efficiency
                for element in result_elements[:self.max_results * 3]:  # Limit processing to avoid excessive work
                    try:
                        # Extract title
                        title_elem = element.select_one(engine["title_selector"])
                        title = title_elem.get_text(strip=True) if title_elem else ""
                        
                        # Extract URL
                        url_elem = element.select_one(engine["url_selector"])
                        url = url_elem.get("href") if url_elem else ""
                        
                        # Clean URL (some engines add tracking parameters or use relative URLs)
                        if url.startswith("/url?q="):
                            url = url.split("/url?q=")[1].split("&")[0]
                        
                        # Ensure absolute URL
                        if not url.startswith(("http://", "https://")):
                            base_url = "/".join(search_url.split("/")[:3])
                            url = urljoin(base_url, url)
                        
                        # Check if domain is excluded
                        domain = urlparse(url).netloc
                        if any(excluded in domain for excluded in self.exclude_domains):
                            continue
                        
                        # Extract snippet
                        snippet_elem = element.select_one(engine["snippet_selector"])
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                        
                        # Only add results with non-empty title and URL
                        if title and url:
                            # Add to results
                            results.append({
                                "title": title,
                                "url": url,
                                "snippet": snippet,
                                "source": engine["name"]
                            })
                        
                    except Exception as e:
                        logger.error(f"Error parsing search result: {e}")
                
                # Add a small delay to avoid rate limiting (smaller delay)
                await asyncio.sleep(random.uniform(0.2, 0.5))
                
                return results
            
            except Exception as e:
                logger.error(f"Error searching {engine['name']}: {str(e)}")
                return []
    
    async def _fetch_page_content(
        self, 
        client: httpx.AsyncClient, 
        result: Dict[str, Any],
        semaphore: asyncio.Semaphore,
        get_user_agent: callable
    ) -> Dict[str, Any]:
        """Fetch the content of a page with resource limiting and optimized parsing"""
        async with semaphore:
            url = result.get("url", "")
            
            if not url:
                result["has_content"] = False
                return result
            
            # Check cache first
            cache_key = f"content:{url}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            try:
                # Use random user agent
                headers = {"User-Agent": get_user_agent()}
                
                # Fetch the page with timeout
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                # Check content type - only process HTML
                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type:
                    result["has_content"] = False
                    return result
                
                # Parse the HTML with minimal features for speed
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Extract metadata quickly
                meta_tags = {}
                for meta in soup.find_all("meta", limit=10):  # Limit to 10 most important meta tags
                    name = meta.get("name") or meta.get("property")
                    content = meta.get("content")
                    if name and content:
                        meta_tags[name] = content
                
                # Extract main content efficiently
                main_content = self._extract_content_fast(soup)
                
                # Determine if content is valid
                has_content = bool(main_content and len(main_content.strip()) >= self.min_content_length)
                
                # Update the result with additional information
                result.update({
                    "meta_tags": meta_tags,
                    "main_content": main_content,
                    "has_content": has_content,
                    "last_updated": time.time()
                })
                
                # Cache the result
                self.cache[cache_key] = result
                
                return result
            
            except Exception as e:
                logger.error(f"Error fetching page content for {url}: {str(e)}")
                # Mark this result as having no content
                result.update({
                    "error": str(e),
                    "has_content": False,
                    "main_content": ""
                })
                return result
    
    def _extract_content_fast(self, soup: BeautifulSoup) -> str:
        """Extract the main content from HTML using a fast approach"""
        # Remove script and style elements to clean up the content
        for script in soup(["script", "style", "nav", "footer", "header", "aside"], limit=50):
            script.decompose()
        
        # Try to find main content in common containers - check only a few most common ones
        content_selectors = [
            "article", "main", ".content", "#content", ".post", ".article"
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                return content_elem.get_text(separator=" ", strip=True)[:5000]
        
        # If no content found, look for the div with the most text
        divs = soup.find_all("div", limit=30)  # Limit to top 30 divs to avoid excessive processing
        best_div = None
        most_text = 0
        
        for div in divs:
            text = div.get_text(strip=True)
            if len(text) > most_text:
                most_text = len(text)
                best_div = div
        
        if best_div and most_text >= self.min_content_length:
            return best_div.get_text(separator=" ", strip=True)[:5000]
        
        # If still no content, get all paragraph text
        paragraphs = soup.find_all("p", limit=20)
        if paragraphs:
            text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
            return text[:5000]
        
        # Last resort: get body text
        if soup.body:
            return soup.body.get_text(separator=" ", strip=True)[:5000]
        
        return ""
    
    async def search_by_domain(self, query: str, domain: str) -> List[Dict[str, Any]]:
        """Search for information within a specific domain"""
        site_query = f"{query} site:{domain}"
        return await self.search(site_query)