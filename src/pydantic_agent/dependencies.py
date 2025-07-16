"""
Dependency injection types for Pydantic AI agents.

This module defines type-safe dependency classes that are injected into
agent tools and system prompts, following Pydantic AI's dependency injection
patterns for configuration and resource management.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CrawlDependencies:
    """
    Dependencies for web crawling operations.
    
    This class encapsulates configuration and settings needed for crawling
    workflows, providing type-safe access to MCP server connection details
    and crawling parameters.
    
    Attributes:
        mcp_server_url: URL of the crawl4ai MCP server (SSE transport)
        default_max_depth: Default recursion depth for recursive crawling
        default_max_concurrent: Default number of concurrent browser sessions
        default_chunk_size: Default size for content chunking
        preferred_strategies: Preferred crawling strategies in order of preference
    """
    mcp_server_url: str = "http://localhost:8051/sse"
    default_max_depth: int = 3
    default_max_concurrent: int = 10
    default_chunk_size: int = 5000
    preferred_strategies: List[str] = field(default_factory=lambda: ["smart", "single", "recursive"])


@dataclass
class RAGDependencies:
    """
    Dependencies for RAG (Retrieval Augmented Generation) operations.
    
    This class provides configuration for semantic search, content retrieval,
    and query processing workflows that interact with the crawled content
    stored in the PostgreSQL vector database.
    
    Attributes:
        mcp_server_url: URL of the crawl4ai MCP server (SSE transport)
        default_match_count: Default number of results to return from searches
        preferred_sources: List of preferred source domains for filtering
        enable_hybrid_search: Whether to prefer hybrid search when available
        enable_code_search: Whether to include code example searches
        confidence_threshold: Minimum confidence score for results (0.0-1.0)
    """
    mcp_server_url: str = "http://localhost:8051/sse"
    default_match_count: int = 5
    preferred_sources: List[str] = field(default_factory=list)
    enable_hybrid_search: bool = True
    enable_code_search: bool = True
    confidence_threshold: float = 0.0


@dataclass
class WorkflowDependencies:
    """
    Dependencies for complex multi-step workflows.
    
    This class combines crawling and RAG dependencies for workflows that
    involve both content acquisition and intelligent querying, providing
    a unified configuration interface.
    
    Attributes:
        crawl_deps: Configuration for crawling operations
        rag_deps: Configuration for RAG operations
        workflow_timeout: Maximum time in seconds for workflow completion
        enable_progress_tracking: Whether to track and report workflow progress
        auto_retry_failed_steps: Whether to automatically retry failed steps
    """
    crawl_deps: CrawlDependencies = field(default_factory=CrawlDependencies)
    rag_deps: RAGDependencies = field(default_factory=RAGDependencies)
    workflow_timeout: int = 300  # 5 minutes
    enable_progress_tracking: bool = True
    auto_retry_failed_steps: bool = True
    
    def __post_init__(self):
        """Ensure both dependencies use the same MCP server URL."""
        if self.crawl_deps.mcp_server_url != self.rag_deps.mcp_server_url:
            # Sync to crawl_deps URL as the primary
            self.rag_deps.mcp_server_url = self.crawl_deps.mcp_server_url
