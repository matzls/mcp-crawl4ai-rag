"""
Utility functions for the Crawl4AI MCP server.

This module maintains backward compatibility by re-exporting functions from
the new modular structure. For new code, import directly from the specific modules.
"""

# Database operations
from database.operations import (
    get_postgres_connection,
    create_postgres_pool,
    add_documents_to_postgres,
    add_code_examples_to_postgres,
    update_source_info,
    search_documents,
    search_code_examples
)

# Embedding operations
from embeddings.generator import (
    create_embeddings_batch,
    create_embedding,
    embedding_to_vector_string,
    generate_contextual_embedding,
    process_chunk_with_context,
    process_documents_with_contextual_embeddings
)

# Content processing
from content.processor import (
    extract_code_blocks,
    generate_code_example_summary,
    extract_source_summary,
    process_code_examples_batch,
    analyze_content_structure
)

# Search operations
from search.engine import (
    rerank_results,
    filter_search_results,
    enhance_search_query,
    process_search_results
)

# Maintain the original API for backward compatibility
__all__ = [
    # Database operations
    "get_postgres_connection",
    "create_postgres_pool", 
    "add_documents_to_postgres",
    "add_code_examples_to_postgres",
    "update_source_info",
    "search_documents",
    "search_code_examples",
    
    # Embedding operations
    "create_embeddings_batch",
    "create_embedding",
    "embedding_to_vector_string",
    "generate_contextual_embedding", 
    "process_chunk_with_context",
    
    # Content processing
    "extract_code_blocks",
    "generate_code_example_summary",
    "extract_source_summary"
]

# Note: This module now serves as a compatibility layer.
# For new development, import directly from the specific modules:
# - from database.operations import create_postgres_pool
# - from embeddings.generator import create_embedding
# - from content.processor import extract_code_blocks
# - from search.engine import rerank_results