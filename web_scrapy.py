import httpx
import asyncio
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import logging
import random
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebScraper")

class WebScraper:
    def __init__(
        self, 
        max_results: int = 5, 
        search_depth: int = 2, 
        timeout: int = 30,
        exclude_domains: List[str] = None,
        user_agents: List[str] = None,
        min_content_length: int = 100  # Minimum content length to consider valid
    ):
        self.max_results = max_results
        self.search_depth = search_depth
        self.timeout = timeout
        self.exclude_domains = exclude_domains or []
        self.min_content_length = min_content_length
        
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
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Execute a search across multiple search engines and aggregate results"""
        logger.info(f"Searching for: {query}")
        all_results = []
        tasks = []
        
        # Create a client session with limits
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            # Search across all engines
            for engine in self.search_engines:
                tasks.append(self._search_engine(client, query, engine))
            
            engine_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine and deduplicate results
            for results in engine_results:
                if isinstance(results, Exception):
                    logger.error(f"Search engine error: {results}")
                    continue
                
                all_results.extend(results)
            
            # Remove duplicates based on URL
            unique_results = self._deduplicate_results(all_results)
            
            # Limit results to max_results or request more if search_depth > 1
            desired_count = self.max_results * (2 if self.search_depth > 1 else 1)
            limited_results = unique_results[:desired_count]
            
            # If search depth > 1, follow links and extract additional information
            if self.search_depth > 1:
                limited_results = await self._enrich_results(client, limited_results)
                
                # Filter results to ensure they have valid content
                valid_results = self._filter_valid_results(limited_results)
                
                # If we have fewer valid results than max_results, try to get more
                if len(valid_results) < self.max_results and len(limited_results) < len(unique_results):
                    additional_results = unique_results[len(limited_results):len(limited_results) + (self.max_results - len(valid_results))]
                    
                    if additional_results:
                        enriched_additional = await self._enrich_results(client, additional_results)
                        valid_additional = self._filter_valid_results(enriched_additional)
                        valid_results.extend(valid_additional)
                
                return valid_results[:self.max_results]
            
            return limited_results[:self.max_results]
    
    def _filter_valid_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results to only include those with valid content"""
        valid_results = []
        
        for result in results:
            # Check if the result has non-empty main content
            main_content = result.get("main_content", "")
            
            if main_content and len(main_content.strip()) >= self.min_content_length:
                valid_results.append(result)
        
        return valid_results
    
    async def _search_engine(self, client: httpx.AsyncClient, query: str, engine: Dict[str, str]) -> List[Dict[str, Any]]:
        """Execute a search on a specific search engine"""
        results = []
        
        try:
            # Format the search URL - request more results than needed
            search_url = engine["url"].format(query=query, num=self.max_results * 3)
            
            # Use random user agent
            headers = {"User-Agent": random.choice(self.user_agents)}
            
            # Execute the search request
            response = await client.get(search_url, headers=headers)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract results
            result_elements = soup.select(engine["result_selector"])
            
            for element in result_elements:
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
            
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(random.uniform(1.0, 3.0))
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching {engine['name']}: {str(e)}")
            return []
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on URL"""
        unique_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get("url", "")
            if url and url not in unique_urls:
                unique_urls.add(url)
                unique_results.append(result)
        
        return unique_results
    
    async def _enrich_results(self, client: httpx.AsyncClient, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Follow links and extract additional information"""
        tasks = []
        
        for result in results:
            tasks.append(self._fetch_page_content(client, result))
        
        enriched_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and keep track of the results
        final_results = []
        for i, r in enumerate(enriched_results):
            if isinstance(r, Exception):
                logger.error(f"Error enriching result: {r}")
            else:
                # Include the result only if it has content
                if r.get("main_content", ""):
                    final_results.append(r)
                else:
                    logger.info(f"Skipping result without content: {r.get('url', 'unknown URL')}")
        
        return final_results
    
    async def _fetch_page_content(self, client: httpx.AsyncClient, result: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch the content of a page and extract more information"""
        url = result.get("url", "")
        
        if not url:
            return result
        
        try:
            # Use random user agent
            headers = {"User-Agent": random.choice(self.user_agents)}
            
            # Fetch the page
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract metadata
            meta_tags = {}
            for meta in soup.find_all("meta"):
                name = meta.get("name") or meta.get("property")
                content = meta.get("content")
                if name and content:
                    meta_tags[name] = content
            
            # Extract main content
            main_content = ""
            
            # Try to find main content in common containers
            content_selectors = [
                "article", "main", ".content", "#content", ".post", ".article", 
                ".entry-content", "#main-content"
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Get the text but limit to 5000 characters
                    main_content = content_elem.get_text(separator=" ", strip=True)[:5000]
                    break
            
            # If no content found, use the body text
            if not main_content and soup.body:
                main_content = soup.body.get_text(separator=" ", strip=True)[:5000]
            
            # Get internal links for potential further crawling
            internal_links = []
            domain = urlparse(url).netloc
            
            for a in soup.find_all("a", href=True):
                link_url = a.get("href")
                if link_url:
                    # Make URL absolute
                    if not link_url.startswith(("http://", "https://")):
                        link_url = urljoin(url, link_url)
                    
                    # Check if it's an internal link
                    link_domain = urlparse(link_url).netloc
                    if link_domain == domain:
                        internal_links.append({
                            "url": link_url,
                            "text": a.get_text(strip=True)
                        })
            
            # Update the result with additional information
            result.update({
                "meta_tags": meta_tags,
                "main_content": main_content,
                "internal_links": internal_links[:10],  # Limit to 10 internal links
                "last_updated": time.time(),
                "has_content": bool(main_content and len(main_content.strip()) >= self.min_content_length)
            })
            
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
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

    async def search_by_domain(self, query: str, domain: str) -> List[Dict[str, Any]]:
        """Search for information within a specific domain"""
        site_query = f"{query} site:{domain}"
        return await self.search(site_query)