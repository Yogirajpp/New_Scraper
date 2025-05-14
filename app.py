from fastapi import FastAPI, Query, HTTPException, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import asyncio
import time
import concurrent.futures
from web_scrapy import WebScraper
from web_extract import WebExtractor
import functools

app = FastAPI(
    title="Information Gathering API",
    description="An API for gathering information from the internet based on queries",
    version="1.0.0"
)

# Create a thread pool executor for CPU-bound tasks
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

# Create a connection pool for HTTP requests
connection_pool_size = 20
connection_timeout = 15  # Reduced timeout for faster responses

class SearchQuery(BaseModel):
    query: str
    max_results: Optional[int] = 5
    search_depth: Optional[int] = 2
    include_images: Optional[bool] = False
    exclude_domains: Optional[List[str]] = []
    parallel_processing: Optional[bool] = True  # Enable by default
    timeout: Optional[int] = 15  # Default timeout in seconds

class SearchResult(BaseModel):
    id: str
    status: str
    progress: float
    results: Optional[List[Dict[str, Any]]] = None
    execution_time: Optional[float] = None
    total_sources: Optional[int] = None
    error: Optional[str] = None

# Store for background search results
search_results_store = {}

@app.get("/")
async def root():
    return {"message": "Welcome to the Information Gathering API"}

@app.post("/search", response_model=dict)
async def search(search_query: SearchQuery, background_tasks: BackgroundTasks):
    """
    Start a search in the background and return a job ID
    """
    import uuid
    
    job_id = str(uuid.uuid4())
    
    # Initialize the search job
    search_results_store[job_id] = {
        "id": job_id,
        "status": "processing",
        "progress": 0.0,
        "results": None,
        "execution_time": None,
        "total_sources": None,
        "error": None
    }
    
    # Add the task to the background
    background_tasks.add_task(
        perform_search_task, 
        job_id=job_id, 
        search_query=search_query
    )
    
    return {"job_id": job_id, "status": "processing"}

@app.get("/search/{job_id}", response_model=dict)
async def get_search_result(job_id: str):
    """
    Get the current status or result of a search job
    """
    if job_id not in search_results_store:
        raise HTTPException(status_code=404, detail="Search job not found")
    
    result = search_results_store[job_id]
    
    # Clean up completed jobs after an hour
    if result["status"] in ["completed", "failed"] and time.time() - result.get("completed_at", 0) > 3600:
        background_tasks = BackgroundTasks()
        background_tasks.add_task(clean_up_job, job_id)
    
    return result

async def perform_search_task(job_id: str, search_query: SearchQuery):
    """
    Perform the actual search in the background
    """
    start_time = time.time()
    
    try:
        # Initialize web scraper with optimized settings
        scraper = WebScraper(
            max_results=search_query.max_results * 2,  # Request more results to account for filtering
            search_depth=search_query.search_depth,
            exclude_domains=search_query.exclude_domains,
            timeout=search_query.timeout,
            connection_pool_size=connection_pool_size
        )
        
        # Update progress
        search_results_store[job_id]["progress"] = 0.2
        
        # Search for the query
        search_results = await scraper.search(search_query.query)
        
        # Update progress
        search_results_store[job_id]["progress"] = 0.5
        
        # Initialize web extractor
        extractor = WebExtractor(
            include_images=search_query.include_images,
            timeout=search_query.timeout,
            thread_pool=thread_pool
        )
        
        processed_results = []
        
        # Process results in parallel if enabled
        if search_query.parallel_processing:
            # Create processing tasks for all results
            tasks = []
            for i, result in enumerate(search_results):
                tasks.append(
                    extractor.extract_information(
                        url=result.get("url", ""),
                        title=result.get("title", ""),
                        snippet=result.get("snippet", "")
                    )
                )
            
            # Process all results in parallel
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter valid results and handle exceptions
            valid_results = []
            for i, res in enumerate(all_results):
                if isinstance(res, Exception):
                    continue
                if res.get("content") and res.get("has_valid_content", False):
                    valid_results.append(res)
                    
                # Update progress periodically
                if i % 2 == 0:
                    progress = 0.5 + 0.45 * (i / len(all_results))
                    search_results_store[job_id]["progress"] = min(0.95, progress)
            
            # Take only the requested number of valid results
            processed_results = valid_results[:search_query.max_results]
            
        else:
            # Process sequentially for lower resource usage
            for i, result in enumerate(search_results):
                try:
                    extracted_data = await extractor.extract_information(
                        url=result.get("url", ""),
                        title=result.get("title", ""),
                        snippet=result.get("snippet", "")
                    )
                    
                    # Only include results with non-empty content
                    if extracted_data.get("content") and extracted_data.get("has_valid_content", False):
                        processed_results.append(extracted_data)
                        
                        # Stop once we have enough valid results
                        if len(processed_results) >= search_query.max_results:
                            break
                    
                    # Update progress
                    progress = 0.5 + 0.45 * (i / len(search_results))
                    search_results_store[job_id]["progress"] = min(0.95, progress)
                    
                except Exception as e:
                    continue
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Store the results
        search_results_store[job_id].update({
            "status": "completed",
            "progress": 1.0,
            "results": processed_results,
            "execution_time": round(execution_time, 2),
            "total_sources": len(processed_results),
            "completed_at": time.time()
        })
    
    except Exception as e:
        # Handle errors
        search_results_store[job_id].update({
            "status": "failed",
            "progress": 1.0,
            "error": f"An error occurred: {str(e)}",
            "completed_at": time.time()
        })

async def clean_up_job(job_id: str):
    """
    Remove a completed job from the store after it's been retrieved
    """
    if job_id in search_results_store:
        del search_results_store[job_id]

@app.post("/extract-from-url")
async def extract_from_url(url: str = Form(...), timeout: int = Form(15)):
    try:
        extractor = WebExtractor(include_images=True, timeout=timeout, thread_pool=thread_pool)
        result = await extractor.extract_from_url(url)
        
        # Make sure result has content
        if not result.get("content") or not result.get("has_valid_content", False):
            return JSONResponse(content={"error": "No content could be extracted from the URL"})
            
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting from URL: {str(e)}")

@app.post("/extract-from-image")
async def extract_from_image(image: UploadFile = File(...)):
    try:
        extractor = WebExtractor(include_images=True, thread_pool=thread_pool)
        content = await image.read()
        result = await extractor.extract_from_image(content)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting from image: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "workers": thread_pool._max_workers}

if __name__ == "__main__":
    import multiprocessing
    
    # Determine optimal number of workers based on CPU cores
    workers = min(multiprocessing.cpu_count() + 1, 8)
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=workers)