from fastapi import FastAPI, Query, HTTPException, Form, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import asyncio
import time
from web_scrapy import WebScraper
from web_extract import WebExtractor

app = FastAPI(
    title="Information Gathering API",
    description="An API for gathering information from the internet based on queries",
    version="1.0.0"
)

class SearchQuery(BaseModel):
    query: str
    max_results: Optional[int] = 5
    search_depth: Optional[int] = 2
    include_images: Optional[bool] = False
    exclude_domains: Optional[List[str]] = []

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    execution_time: float
    total_sources: int

@app.get("/")
async def root():
    return {"message": "Welcome to the Information Gathering API"}

@app.post("/search", response_model=SearchResponse)
async def search(search_query: SearchQuery):
    start_time = time.time()
    
    try:
        # Initialize web scraper with a higher max_results to account for filtering
        # Request more results than needed since we'll filter out empty ones
        scraper = WebScraper(
            max_results=search_query.max_results * 2,  # Request more results to account for filtering
            search_depth=search_query.search_depth,
            exclude_domains=search_query.exclude_domains
        )
        
        # Search for the query
        search_results = await scraper.search(search_query.query)
        
        # Initialize web extractor and process results
        extractor = WebExtractor(include_images=search_query.include_images)
        processed_results = []
        
        # Process each result and filter out those without content
        for result in search_results:
            extracted_data = await extractor.extract_information(
                url=result.get("url", ""),
                title=result.get("title", ""),
                snippet=result.get("snippet", "")
            )
            
            # Only include results with non-empty content
            if extracted_data.get("content"):
                processed_results.append(extracted_data)
                
                # Stop once we have enough valid results
                if len(processed_results) >= search_query.max_results:
                    break
        
        execution_time = time.time() - start_time
        
        return SearchResponse(
            query=search_query.query,
            results=processed_results,
            execution_time=round(execution_time, 2),
            total_sources=len(processed_results)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/extract-from-url")
async def extract_from_url(url: str = Form(...)):
    try:
        extractor = WebExtractor(include_images=True)
        result = await extractor.extract_from_url(url)
        
        # Make sure result has content
        if not result.get("content"):
            return JSONResponse(content={"error": "No content could be extracted from the URL"})
            
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting from URL: {str(e)}")

@app.post("/extract-from-image")
async def extract_from_image(image: UploadFile = File(...)):
    try:
        extractor = WebExtractor(include_images=True)
        content = await image.read()
        result = await extractor.extract_from_image(content)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting from image: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)