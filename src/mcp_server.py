"""
Core MCP server setup and lifecycle management for Crawl4AI RAG.

This module provides the FastMCP server initialization, context management,
and application lifecycle for the Crawl4AI MCP server.
"""
from mcp.server.fastmcp import FastMCP
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import asyncpg
import asyncio
import os

from crawl4ai import AsyncWebCrawler, BrowserConfig

from utils import create_postgres_pool
from logging_config import log_system_startup, logger

__all__ = [
    "Crawl4AIContext",
    "crawl4ai_lifespan", 
    "mcp",
    "main"
]

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)


@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    postgres_pool: asyncpg.Pool
    reranking_model: Optional[Any] = None  # CrossEncoder when available


@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and PostgreSQL pool
    """
    # Create browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )
    
    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    
    # Initialize PostgreSQL connection pool
    postgres_pool = await create_postgres_pool()
    
    # Initialize cross-encoder model for reranking if enabled
    reranking_model = None
    if os.getenv("USE_RERANKING", "false") == "true":
        try:
            from sentence_transformers import CrossEncoder
            reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"Failed to load reranking model: {e}")
            reranking_model = None
    
    try:
        yield Crawl4AIContext(
            crawler=crawler,
            postgres_pool=postgres_pool,
            reranking_model=reranking_model
        )
    finally:
        # Clean up the crawler and pool
        await crawler.__aexit__(None, None, None)
        await postgres_pool.close()


# Initialize FastMCP server
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="MCP server for RAG and web crawling with Crawl4AI",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8051")
)


async def main():
    """Main entry point for the MCP server."""
    # Log system startup
    log_system_startup("crawl4ai-mcp-server", "0.1.0")
    
    transport = os.getenv("TRANSPORT", "sse")
    logger.info(f"Starting MCP server with {transport} transport")
    
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())