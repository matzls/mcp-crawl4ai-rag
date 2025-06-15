"""
Pydantic AI Agent for Crawl4AI MCP Integration.

This package provides intelligent agents that connect to the crawl4ai MCP server
as clients, enabling sophisticated web crawling and RAG workflows with structured
outputs and type-safe dependency injection.

Architecture:
    Pydantic AI Agent (Client) ←→ Crawl4AI MCP Server (Server) ←→ PostgreSQL + pgvector

Key Components:
    - Agent: Main Pydantic AI agent classes for different workflows
    - Dependencies: Type-safe dependency injection for configuration
    - Outputs: Structured Pydantic models for validated responses
    - Tools: Intelligent wrappers around existing MCP tools
    - Examples: Usage patterns and workflow demonstrations
"""

from .dependencies import CrawlDependencies, RAGDependencies, WorkflowDependencies
from .outputs import CrawlResult, RAGResult, WorkflowResult
from .agent import (
    create_crawl_agent,
    create_rag_agent,
    create_workflow_agent,
    create_mcp_connection,
    run_agent_with_mcp
)

__all__ = [
    "CrawlDependencies",
    "RAGDependencies",
    "WorkflowDependencies",
    "CrawlResult",
    "RAGResult",
    "WorkflowResult",
    "create_crawl_agent",
    "create_rag_agent",
    "create_workflow_agent",
    "create_mcp_connection",
    "run_agent_with_mcp",
]

__version__ = "0.1.0"
