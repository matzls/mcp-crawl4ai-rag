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
from .unified_agent import (
    create_unified_agent,
    run_unified_agent,
    UnifiedAgentDependencies,
    UnifiedAgentResult,
    setup_logfire_instrumentation
)

__all__ = [
    "CrawlDependencies",
    "RAGDependencies",
    "WorkflowDependencies",
    "CrawlResult",
    "RAGResult",
    "WorkflowResult",
    "create_unified_agent",
    "run_unified_agent",
    "UnifiedAgentDependencies",
    "UnifiedAgentResult",
    "setup_logfire_instrumentation",
]

__version__ = "0.1.0"
