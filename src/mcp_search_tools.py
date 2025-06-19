"""
MCP search tool implementations for Crawl4AI RAG.

This module provides the search-focused MCP tools that query and retrieve
stored content from the database using various search strategies.
"""
from mcp.server.fastmcp import Context
from typing import List, Dict, Any
import json
import os

from mcp_server import mcp
from content_processing import rerank_results
from utils import (
    search_documents,
    search_code_examples as search_code_examples_util
)
from logging_config import log_mcp_tool_execution

__all__ = [
    "get_available_sources",
    "perform_rag_query",
    "search_code_examples"
]


@mcp.tool()
@log_mcp_tool_execution("get_available_sources")
async def get_available_sources(ctx: Context) -> str:
    """
    List all crawled content sources available for searching and querying.
    
    USE WHEN: Before performing searches to see what content is indexed, or when user asks "what do you know about?"
    DON'T USE: For actually searching content (use perform_rag_query or search_code_examples instead).
    
    WORKFLOW: This is typically the FIRST step in search workflows to understand available content.
    TYPICAL PATTERNS: 
    • get_available_sources → perform_rag_query (with source filter)
    • get_available_sources → search_code_examples (with source_id filter)
    
    ALWAYS USE: Before calling perform_rag_query or search_code_examples with source filtering!
    
    Args:
        ctx: The MCP server provided context (no parameters needed)
    
    Returns:
        JSON with sources[] array containing source_id, summary, total_words, created_at, updated_at
        
    ERRORS: Database connection failures, empty database (returns empty sources array)
    """
    try:
        # Get the PostgreSQL pool from the context
        postgres_pool = ctx.request_context.lifespan_context.postgres_pool
        
        # Query the sources table directly
        async with postgres_pool.acquire() as conn:
            result = await conn.fetch(
                "SELECT source_id, summary, total_word_count, created_at, updated_at FROM sources ORDER BY source_id"
            )
        
        # Format the sources with their details
        sources = []
        if result:
            for source in result:
                sources.append({
                    "source_id": source["source_id"],
                    "summary": source["summary"],
                    "total_words": source["total_word_count"],
                    "created_at": source["created_at"].isoformat() if source["created_at"] else None,
                    "updated_at": source["updated_at"].isoformat() if source["updated_at"] else None
                })
        
        return json.dumps({
            "success": True,
            "sources": sources,
            "count": len(sources)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


@mcp.tool()
@log_mcp_tool_execution("perform_rag_query")
async def perform_rag_query(ctx: Context, query: str, source: str = None, match_count: int = 5) -> str:
    """
    Search stored content using semantic similarity to answer questions and find relevant information.
    
    USE WHEN: User asks questions about topics, wants to find information, or needs content summaries.
    DON'T USE: For gathering new content from web (use smart_crawl_url), or specifically for code (use search_code_examples).
    
    SEARCH CAPABILITIES:
    • Vector similarity search using OpenAI embeddings
    • Optional hybrid search (vector + keyword matching) if enabled
    • Source filtering to search within specific domains
    • Reranking for improved relevance if enabled
    
    WORKFLOW: Use get_available_sources first to see what content is available, then search it.
    TYPICAL PATTERN: get_available_sources → perform_rag_query → (optional follow-up searches)
    
    Args:
        ctx: The MCP server provided context
        query: Natural language search query (e.g., "Python async programming", "how to handle errors")
        source: Optional source filter from get_available_sources (e.g., "docs.python.org", "github.com")
        match_count: Number of results to return (1-20, default: 5, higher=more comprehensive)
    
    Returns:
        JSON with query, search_mode, results[] containing url, content, metadata, similarity scores
        
    ERRORS: Empty database, no matches found, embedding generation failures, database errors
    """
    try:
        # Get the PostgreSQL pool from the context
        postgres_pool = ctx.request_context.lifespan_context.postgres_pool
        
        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"
        
        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source}
        
        if use_hybrid_search:
            # Hybrid search: combine vector and keyword search
            
            # 1. Get vector search results (get more to account for filtering)
            vector_results = await search_documents(
                pool=postgres_pool,
                query=query,
                match_count=match_count * 2,  # Get double to have room for filtering
                filter_metadata=filter_metadata
            )
            
            # 2. Get keyword search results using ILIKE
            async with postgres_pool.acquire() as conn:
                keyword_query = """
                    SELECT id, url, chunk_number, content, metadata, source_id 
                    FROM crawled_pages 
                    WHERE content ILIKE $1
                """
                query_params = [f'%{query}%']
                
                # Apply source filter if provided
                if source and source.strip():
                    keyword_query += " AND source_id = $2"
                    query_params.append(source)
                
                keyword_query += f" LIMIT {match_count * 2}"
                
                # Execute keyword search
                keyword_response = await conn.fetch(keyword_query, *query_params)
                keyword_results = [dict(record) for record in keyword_response]
            
            # 3. Combine results with preference for items appearing in both
            seen_ids = set()
            combined_results = []
            
            # First, add items that appear in both searches (these are the best matches)
            vector_ids = {r.get('id') for r in vector_results if r.get('id')}
            for kr in keyword_results:
                if kr['id'] in vector_ids and kr['id'] not in seen_ids:
                    # Find the vector result to get similarity score
                    for vr in vector_results:
                        if vr.get('id') == kr['id']:
                            # Boost similarity score for items in both results
                            vr['similarity'] = min(1.0, vr.get('similarity', 0) * 1.2)
                            combined_results.append(vr)
                            seen_ids.add(kr['id'])
                            break
            
            # Then add remaining vector results (semantic matches without exact keyword)
            for vr in vector_results:
                if vr.get('id') and vr['id'] not in seen_ids and len(combined_results) < match_count:
                    combined_results.append(vr)
                    seen_ids.add(vr['id'])
            
            # Finally, add pure keyword matches if we still need more results
            for kr in keyword_results:
                if kr['id'] not in seen_ids and len(combined_results) < match_count:
                    # Convert keyword result to match vector result format
                    combined_results.append({
                        'id': kr['id'],
                        'url': kr['url'],
                        'chunk_number': kr['chunk_number'],
                        'content': kr['content'],
                        'metadata': kr['metadata'],
                        'source_id': kr['source_id'],
                        'similarity': 0.5  # Default similarity for keyword-only matches
                    })
                    seen_ids.add(kr['id'])
            
            # Use combined results
            results = combined_results[:match_count]
            
        else:
            # Standard vector search only
            results = await search_documents(
                pool=postgres_pool,
                query=query,
                match_count=match_count,
                filter_metadata=filter_metadata
            )
        
        # Apply reranking if enabled
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        if use_reranking and ctx.request_context.lifespan_context.reranking_model:
            results = rerank_results(ctx.request_context.lifespan_context.reranking_model, query, results, content_key="content")
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = {
                "url": result.get("url"),
                "content": result.get("content"),
                "metadata": result.get("metadata"),
                "similarity": result.get("similarity")
            }
            # Include rerank score if available
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            formatted_results.append(formatted_result)
        
        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source,
            "search_mode": "hybrid" if use_hybrid_search else "vector",
            "reranking_applied": use_reranking and ctx.request_context.lifespan_context.reranking_model is not None,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)


@mcp.tool()
@log_mcp_tool_execution("search_code_examples")
async def search_code_examples(ctx: Context, query: str, source_id: str = None, match_count: int = 5) -> str:
    """
    Search specifically for code examples and programming snippets with AI-generated summaries.
    
    USE WHEN: User specifically asks for code examples, programming snippets, or implementation details.
    DON'T USE: For general information searches (use perform_rag_query), or if USE_AGENTIC_RAG=false.
    
    SPECIALIZED SEARCH:
    • Targets extracted code blocks from crawled content
    • Includes AI-generated summaries explaining what each code does
    • Searches both code content and summaries for better matching
    • Optional hybrid search combining vector and keyword matching
    
    AVAILABILITY: Only works if USE_AGENTIC_RAG=true in environment configuration.
    WORKFLOW: Use get_available_sources first, then search for code within specific sources.
    TYPICAL PATTERN: get_available_sources → search_code_examples (with source_id filter)
    
    Args:
        ctx: The MCP server provided context
        query: Code-focused search query (e.g., "async function example", "error handling try catch", "class definition")
        source_id: Optional source filter from get_available_sources (e.g., "docs.python.org", "github.com")
        match_count: Number of code examples to return (1-15, default: 5)
    
    Returns:
        JSON with query, results[] containing url, code, summary, metadata, similarity scores
        
    ERRORS: Feature disabled (USE_AGENTIC_RAG=false), no code examples found, database errors
    """
    # Check if code example extraction is enabled
    extract_code_examples_enabled = os.getenv("USE_AGENTIC_RAG", "false") == "true"
    if not extract_code_examples_enabled:
        return json.dumps({
            "success": False,
            "error": "Code example extraction is disabled. Perform a normal RAG search."
        }, indent=2)
    
    try:
        # Get the PostgreSQL pool from the context
        postgres_pool = ctx.request_context.lifespan_context.postgres_pool
        
        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"
        
        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source_id and source_id.strip():
            filter_metadata = {"source": source_id}
        
        if use_hybrid_search:
            # Hybrid search: combine vector and keyword search
            
            # 1. Get vector search results (get more to account for filtering)
            vector_results = await search_code_examples_util(
                pool=postgres_pool,
                query=query,
                match_count=match_count * 2,  # Get double to have room for filtering
                filter_metadata=filter_metadata,
                source_id=source_id
            )
            
            # 2. Get keyword search results using ILIKE on both content and summary
            async with postgres_pool.acquire() as conn:
                keyword_query = """
                    SELECT id, url, chunk_number, content, summary, metadata, source_id 
                    FROM code_examples 
                    WHERE content ILIKE $1 OR summary ILIKE $1
                """
                query_params = [f'%{query}%']
                
                # Apply source filter if provided
                if source_id and source_id.strip():
                    keyword_query += " AND source_id = $2"
                    query_params.append(source_id)
                
                keyword_query += f" LIMIT {match_count * 2}"
                
                # Execute keyword search
                keyword_response = await conn.fetch(keyword_query, *query_params)
                keyword_results = [dict(record) for record in keyword_response]
            
            # 3. Combine results with preference for items appearing in both
            seen_ids = set()
            combined_results = []
            
            # First, add items that appear in both searches (these are the best matches)
            vector_ids = {r.get('id') for r in vector_results if r.get('id')}
            for kr in keyword_results:
                if kr['id'] in vector_ids and kr['id'] not in seen_ids:
                    # Find the vector result to get similarity score
                    for vr in vector_results:
                        if vr.get('id') == kr['id']:
                            # Boost similarity score for items in both results
                            vr['similarity'] = min(1.0, vr.get('similarity', 0) * 1.2)
                            combined_results.append(vr)
                            seen_ids.add(kr['id'])
                            break
            
            # Then add remaining vector results (semantic matches without exact keyword)
            for vr in vector_results:
                if vr.get('id') and vr['id'] not in seen_ids and len(combined_results) < match_count:
                    combined_results.append(vr)
                    seen_ids.add(vr['id'])
            
            # Finally, add pure keyword matches if we still need more results
            for kr in keyword_results:
                if kr['id'] not in seen_ids and len(combined_results) < match_count:
                    # Convert keyword result to match vector result format
                    combined_results.append({
                        'id': kr['id'],
                        'url': kr['url'],
                        'chunk_number': kr['chunk_number'],
                        'content': kr['content'],
                        'summary': kr['summary'],
                        'metadata': kr['metadata'],
                        'source_id': kr['source_id'],
                        'similarity': 0.5  # Default similarity for keyword-only matches
                    })
                    seen_ids.add(kr['id'])
            
            # Use combined results
            results = combined_results[:match_count]
            
        else:
            # Standard vector search only
            results = await search_code_examples_util(
                pool=postgres_pool,
                query=query,
                match_count=match_count,
                filter_metadata=filter_metadata,
                source_id=source_id
            )
        
        # Apply reranking if enabled
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        if use_reranking and ctx.request_context.lifespan_context.reranking_model:
            results = rerank_results(ctx.request_context.lifespan_context.reranking_model, query, results, content_key="content")
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = {
                "url": result.get("url"),
                "code": result.get("content"),
                "summary": result.get("summary"),
                "metadata": result.get("metadata"),
                "source_id": result.get("source_id"),
                "similarity": result.get("similarity")
            }
            # Include rerank score if available
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            formatted_results.append(formatted_result)
        
        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source_id,
            "search_mode": "hybrid" if use_hybrid_search else "vector",
            "reranking_applied": use_reranking and ctx.request_context.lifespan_context.reranking_model is not None,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)