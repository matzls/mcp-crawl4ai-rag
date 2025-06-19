"""
MCP server for web crawling with Crawl4AI.

DEPRECATED: This module has been refactored into multiple components for better maintainability.
The core functionality is now split across:
- mcp_server.py: Core MCP server setup and lifecycle management
- mcp_tools.py: All 5 MCP tool implementations  
- crawl_strategies.py: Crawling strategy implementations
- content_processing.py: Content processing and search enhancements

This file maintains backward compatibility for imports and provides the main entry point.
"""

# Import everything from the new modular structure for backward compatibility
from mcp_server import (
    Crawl4AIContext,
    crawl4ai_lifespan,
    mcp,
    main
)

from mcp_crawl_tools import (
    crawl_single_page,
    smart_crawl_url
)

from mcp_search_tools import (
    get_available_sources, 
    perform_rag_query,
    search_code_examples
)

from crawl_strategies import (
    is_sitemap,
    is_txt,
    parse_sitemap,
    crawl_markdown_file,
    crawl_batch,
    crawl_recursive_internal_links
)

from content_processing import (
    smart_chunk_markdown,
    extract_section_info,
    process_code_example,
    rerank_results
)

# Re-export for backward compatibility
__all__ = [
    # Core server components
    "Crawl4AIContext",
    "crawl4ai_lifespan", 
    "mcp",
    "main",
    
    # MCP tools
    "crawl_single_page",
    "smart_crawl_url", 
    "get_available_sources",
    "perform_rag_query",
    "search_code_examples",
    
    # Crawling strategies
    "is_sitemap",
    "is_txt",
    "parse_sitemap", 
    "crawl_markdown_file",
    "crawl_batch",
    "crawl_recursive_internal_links",
    
    # Content processing
    "smart_chunk_markdown",
    "extract_section_info",
    "process_code_example", 
    "rerank_results"
]

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())