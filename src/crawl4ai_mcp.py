"""
MCP server for web crawling with Crawl4AI.

DEPRECATED: This module has been refactored into multiple components for better maintainability.
The core functionality is now split across:
- mcp_server.py: Core MCP server setup and lifecycle management
- mcp_crawl_tools.py: Crawling MCP tools (2 tools)
- mcp_search_tools.py: Search MCP tools (3 tools)
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

from utils import (
    create_postgres_pool,
    add_documents_to_postgres,
    search_documents,
    extract_code_blocks,
    generate_code_example_summary
)

# For backward compatibility, expose main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())