"""
MCP crawling tool implementations for Crawl4AI RAG.

This module provides the crawling-focused MCP tools that gather new content
from websites and store it in the database.
"""
from mcp.server.fastmcp import Context
from urllib.parse import urlparse
import json
import asyncio
import concurrent.futures
import os

from crawl4ai import CrawlerRunConfig, CacheMode

from mcp_server import mcp
from content_processing import (
    smart_chunk_markdown, 
    extract_section_info, 
    process_code_example
)
from crawl_strategies import (
    is_sitemap,
    is_txt, 
    parse_sitemap,
    crawl_markdown_file,
    crawl_batch,
    crawl_recursive_internal_links
)
from utils import (
    add_documents_to_postgres,
    extract_code_blocks,
    add_code_examples_to_postgres,
    update_source_info,
    extract_source_summary,
    create_embeddings_batch
)
from logging_config import log_mcp_tool_execution

__all__ = [
    "crawl_single_page",
    "smart_crawl_url"
]


@mcp.tool()
@log_mcp_tool_execution("crawl_single_page")
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in PostgreSQL for later retrieval.
    
    USE WHEN: User wants content from ONE specific webpage without following links.
    DON'T USE: For sitemaps, multiple pages, or recursive crawling (use smart_crawl_url instead).
    
    WORKFLOW: This tool STORES content. Follow with perform_rag_query to search stored content.
    TYPICAL PATTERN: crawl_single_page → get_available_sources → perform_rag_query
    
    Args:
        ctx: The MCP server provided context
        url: Target webpage URL (e.g., "https://docs.python.org/3/tutorial/introduction.html")
    
    Returns:
        JSON with success status, chunks_stored count, content_length, source_id, and processing summary
        
    ERRORS: Network failures, invalid URLs, content extraction failures, database storage issues
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        postgres_pool = ctx.request_context.lifespan_context.postgres_pool
        
        # Configure the crawl
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        
        # Crawl the page
        result = await crawler.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            # Extract source_id
            parsed_url = urlparse(url)
            source_id = parsed_url.netloc or parsed_url.path
            
            # Chunk the content
            chunks = smart_chunk_markdown(result.markdown)
            
            # Prepare data for PostgreSQL
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            total_word_count = 0
            
            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = source_id
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)
                
                # Accumulate word count
                total_word_count += meta.get("word_count", 0)
            
            # Update source information FIRST (before inserting documents)
            source_summary = extract_source_summary(source_id, result.markdown[:5000])  # Use first 5000 chars for summary
            await update_source_info(postgres_pool, source_id, source_summary, total_word_count)
            
            # Generate embeddings for the content chunks
            embeddings = create_embeddings_batch(contents)
            
            # Add documentation chunks to PostgreSQL (AFTER source exists)
            await add_documents_to_postgres(postgres_pool, urls, chunk_numbers, contents, metadatas, embeddings)
            
            # Extract and process code examples only if enabled
            extract_code_examples = os.getenv("USE_AGENTIC_RAG", "false") == "true"
            code_blocks = []
            if extract_code_examples:
                code_blocks = extract_code_blocks(result.markdown)
                if code_blocks:
                    code_urls = []
                    code_chunk_numbers = []
                    code_examples = []
                    code_summaries = []
                    code_metadatas = []
                    
                    # Process code examples in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        # Prepare arguments for parallel processing
                        summary_args = [(block['code'], block['context_before'], block['context_after']) 
                                        for block in code_blocks]
                        
                        # Generate summaries in parallel
                        summaries = list(executor.map(process_code_example, summary_args))
                    
                    # Prepare code example data
                    for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                        code_urls.append(url)
                        code_chunk_numbers.append(i)
                        code_examples.append(block['code'])
                        code_summaries.append(summary)
                        
                        # Create metadata for code example
                        code_meta = {
                            "chunk_index": i,
                            "url": url,
                            "source": source_id,
                            "char_count": len(block['code']),
                            "word_count": len(block['code'].split())
                        }
                        code_metadatas.append(code_meta)
                    
                    # Generate embeddings for code examples
                    code_embeddings = create_embeddings_batch(code_examples)
                    
                    # Add code examples to PostgreSQL
                    await add_code_examples_to_postgres(
                        postgres_pool, 
                        code_urls, 
                        code_chunk_numbers, 
                        code_examples, 
                        code_summaries, 
                        code_metadatas,
                        code_embeddings
                    )
            
            return json.dumps({
                "success": True,
                "url": url,
                "chunks_stored": len(chunks),
                "code_examples_stored": len(code_blocks) if code_blocks else 0,
                "content_length": len(result.markdown),
                "total_word_count": total_word_count,
                "source_id": source_id,
                "links_count": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", []))
                }
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "url": url,
                "error": result.error_message
            }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)


@mcp.tool()
@log_mcp_tool_execution("smart_crawl_url")
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000) -> str:
    """
    Intelligently crawl web content with automatic URL type detection and store in PostgreSQL.
    
    USE WHEN: User wants to gather NEW content from websites, sitemaps, or documentation sites.
    DON'T USE: If user is asking questions about EXISTING content (use perform_rag_query instead).
    
    AUTOMATIC DETECTION & STRATEGY:
    - Sitemap URLs (*.xml, */sitemap*): Extracts all URLs and crawls in parallel
    - Text files (*.txt, *.md): Direct content retrieval without browser rendering
    - Regular webpages: Recursive crawling following internal links up to max_depth
    
    WORKFLOW: This tool STORES content. Follow with get_available_sources → perform_rag_query to search it.
    TYPICAL PATTERN: smart_crawl_url → get_available_sources → perform_rag_query/search_code_examples
    
    Args:
        ctx: The MCP server provided context
        url: Target URL - examples:
             • "https://docs.python.org" (recursive webpage crawling)
             • "https://site.com/sitemap.xml" (sitemap extraction) 
             • "https://raw.github.com/user/repo/README.md" (direct text file)
        max_depth: Link recursion depth for webpages (1-5, default: 3, higher=more pages)
        max_concurrent: Parallel browser sessions (1-20, default: 10, higher=faster but more memory)
        chunk_size: Content chunk size in characters (1000-10000, default: 5000)
    
    Returns:
        JSON with crawl_type, pages_crawled, chunks_stored, code_examples_stored, urls_crawled[]
        
    ERRORS: Network timeouts, invalid URLs, rate limiting, browser failures, storage errors
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        postgres_pool = ctx.request_context.lifespan_context.postgres_pool
        
        # Determine the crawl strategy
        crawl_results = []
        crawl_type = None
        
        if is_txt(url):
            # For text files, use simple crawl
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            # For sitemaps, extract URLs and crawl in parallel
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "No URLs found in sitemap"
                }, indent=2)
            crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
            crawl_type = "sitemap"
        else:
            # For regular URLs, use recursive crawl
            crawl_results = await crawl_recursive_internal_links(crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent)
            crawl_type = "webpage"
        
        if not crawl_results:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "No content found"
            }, indent=2)
        
        # Process results and store in PostgreSQL
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        chunk_count = 0
        
        # Track sources and their content
        source_content_map = {}
        source_word_counts = {}
        
        # Process documentation chunks
        for doc in crawl_results:
            source_url = doc['url']
            md = doc['markdown']
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
            
            # Extract source_id
            parsed_url = urlparse(source_url)
            source_id = parsed_url.netloc or parsed_url.path
            
            # Store content for source summary generation
            if source_id not in source_content_map:
                source_content_map[source_id] = md[:5000]  # Store first 5000 chars
                source_word_counts[source_id] = 0
            
            for i, chunk in enumerate(chunks):
                urls.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = source_url
                meta["source"] = source_id
                meta["crawl_type"] = crawl_type
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)
                
                # Accumulate word count
                source_word_counts[source_id] += meta.get("word_count", 0)
                
                chunk_count += 1
        
        # Create url_to_full_document mapping
        url_to_full_document = {}
        for doc in crawl_results:
            url_to_full_document[doc['url']] = doc['markdown']
        
        # Update source information for each unique source FIRST (before inserting documents)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            source_summary_args = [(source_id, content) for source_id, content in source_content_map.items()]
            source_summaries = list(executor.map(lambda args: extract_source_summary(args[0], args[1]), source_summary_args))
        
        for (source_id, _), summary in zip(source_summary_args, source_summaries):
            word_count = source_word_counts.get(source_id, 0)
            await update_source_info(postgres_pool, source_id, summary, word_count)
        
        # Generate embeddings for the content chunks
        embeddings = create_embeddings_batch(contents)
        
        # Add documentation chunks to PostgreSQL (AFTER sources exist)
        batch_size = 20
        await add_documents_to_postgres(postgres_pool, urls, chunk_numbers, contents, metadatas, embeddings, batch_size=batch_size)
        
        # Extract and process code examples from all documents only if enabled
        extract_code_examples_enabled = os.getenv("USE_AGENTIC_RAG", "false") == "true"
        code_examples = []  # Initialize to empty list
        if extract_code_examples_enabled:
            code_urls = []
            code_chunk_numbers = []
            code_examples = []
            code_summaries = []
            code_metadatas = []
            
            # Extract code blocks from all documents
            for doc in crawl_results:
                source_url = doc['url']
                md = doc['markdown']
                code_blocks = extract_code_blocks(md)
                
                if code_blocks:
                    # Process code examples in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        # Prepare arguments for parallel processing
                        summary_args = [(block['code'], block['context_before'], block['context_after']) 
                                        for block in code_blocks]
                        
                        # Generate summaries in parallel
                        summaries = list(executor.map(process_code_example, summary_args))
                    
                    # Prepare code example data
                    parsed_url = urlparse(source_url)
                    source_id = parsed_url.netloc or parsed_url.path
                    
                    for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                        code_urls.append(source_url)
                        code_chunk_numbers.append(len(code_examples))  # Use global code example index
                        code_examples.append(block['code'])
                        code_summaries.append(summary)
                        
                        # Create metadata for code example
                        code_meta = {
                            "chunk_index": len(code_examples) - 1,
                            "url": source_url,
                            "source": source_id,
                            "char_count": len(block['code']),
                            "word_count": len(block['code'].split())
                        }
                        code_metadatas.append(code_meta)
            
            # Add all code examples to PostgreSQL
            if code_examples:
                # Generate embeddings for code examples
                code_embeddings = create_embeddings_batch(code_examples)
                
                await add_code_examples_to_postgres(
                    postgres_pool, 
                    code_urls, 
                    code_chunk_numbers, 
                    code_examples, 
                    code_summaries, 
                    code_metadatas,
                    code_embeddings,
                    batch_size=batch_size
                )
        
        return json.dumps({
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_crawled": len(crawl_results),
            "chunks_stored": chunk_count,
            "code_examples_stored": len(code_examples),
            "sources_updated": len(source_content_map),
            "urls_crawled": [doc['url'] for doc in crawl_results][:5] + (["..."] if len(crawl_results) > 5 else [])
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)