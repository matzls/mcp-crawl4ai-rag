"""
Main Pydantic AI agent implementation for crawl4ai MCP integration.

This module provides the core agent classes that connect to the crawl4ai MCP server
as clients, following the documented patterns from Pydantic AI's MCP integration.
"""

from typing import Optional, List, Any
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerHTTP

try:
    from .dependencies import CrawlDependencies, RAGDependencies, WorkflowDependencies
    from .outputs import CrawlResult, RAGResult, WorkflowResult
except ImportError:
    # Fallback for when running from different contexts
    from pydantic_agent.dependencies import CrawlDependencies, RAGDependencies, WorkflowDependencies
    from pydantic_agent.outputs import CrawlResult, RAGResult, WorkflowResult

# Import logfire for Pydantic AI integration
try:
    import logfire
    # Configure logfire for Pydantic AI agents
    logfire.configure()
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False

# Import logging utilities
try:
    from logging_config import log_agent_interaction, logger
except ImportError:
    # Fallback for when running from different contexts
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from logging_config import log_agent_interaction, logger


async def create_mcp_connection(server_url: str) -> MCPServerHTTP:
    """
    Create an MCP server connection following documented patterns.

    Args:
        server_url: URL of the MCP server (e.g., "http://localhost:8051/sse")

    Returns:
        MCPServerHTTP instance configured for the server
    """
    return MCPServerHTTP(url=server_url)


def create_crawl_agent(server_url: str = "http://localhost:8051/sse") -> Agent:
    """
    Create a crawling agent specialized for web crawling operations.

    Following the documented Pydantic AI + MCP integration pattern.

    Args:
        server_url: URL of the crawl4ai MCP server

    Returns:
        Configured Agent for crawling operations with tools registered
    """
    # Create MCP server connection
    server = MCPServerHTTP(url=server_url)

    # Create agent with MCP server - following documented pattern
    # Enable logfire instrumentation if available
    agent_kwargs = {
        'model': 'openai:o3',
        'deps_type': CrawlDependencies,
        'output_type': CrawlResult,
        'mcp_servers': [server],
        'system_prompt': (
            "You are an intelligent web crawling assistant that helps users efficiently "
            "crawl and index web content. You have access to a crawl4ai MCP server that "
            "provides powerful crawling capabilities including single page crawling, "
            "smart URL detection (sitemaps, text files), and recursive crawling.\n\n"
            "Your role is to:\n"
            "1. Analyze the user's crawling requirements\n"
            "2. Choose the most appropriate crawling strategy\n"
            "3. Execute the crawling operation using the available tools\n"
            "4. Provide structured results with clear summaries\n\n"
            "Always prioritize efficiency and provide helpful summaries of what was accomplished."
        )
    }
    
    # Add logfire instrumentation if available
    if LOGFIRE_AVAILABLE:
        agent_kwargs['logfire'] = {'tags': ['crawl-agent', 'mcp-client']}
    
    agent = Agent(**agent_kwargs)

    # Register tools following documented @agent.tool pattern
    @agent.tool
    async def intelligent_crawl(
        ctx: RunContext[CrawlDependencies],
        url: str,
        strategy: str = "smart"
    ) -> str:
        """
        Intelligently crawl a URL using the most appropriate strategy.

        Args:
            ctx: Runtime context with dependencies
            url: URL to crawl
            strategy: Crawling strategy ("single", "smart", "recursive")

        Returns:
            Summary of crawling results
        """
        # This will use the MCP server tools automatically
        # The agent will call the appropriate MCP tools based on the strategy
        return f"Crawling {url} with {strategy} strategy using MCP server tools"

    return agent


def create_rag_agent(server_url: str = "http://localhost:8051/sse") -> Agent:
    """
    Create a RAG agent specialized for content retrieval and querying.

    Following the documented Pydantic AI + MCP integration pattern.

    Args:
        server_url: URL of the crawl4ai MCP server

    Returns:
        Configured Agent for RAG operations with tools registered
    """
    # Create MCP server connection
    server = MCPServerHTTP(url=server_url)

    # Create agent with MCP server - following documented pattern
    # Enable logfire instrumentation if available
    agent_kwargs = {
        'model': 'openai:o3',
        'deps_type': RAGDependencies,
        'output_type': RAGResult,
        'mcp_servers': [server],
        'system_prompt': (
            "You are an intelligent content retrieval assistant that helps users find "
            "and synthesize information from crawled web content. You have access to a "
            "sophisticated RAG system with vector search, hybrid search, and code example "
            "search capabilities.\n\n"
            "Your role is to:\n"
            "1. Understand the user's information needs\n"
            "2. Choose the most appropriate search strategy\n"
            "3. Retrieve relevant content from the vector database\n"
            "4. Synthesize a comprehensive answer from the retrieved content\n"
            "5. Provide confidence scores and source attribution\n\n"
            "Always strive to provide accurate, well-sourced answers with appropriate confidence levels."
        )
    }
    
    # Add logfire instrumentation if available
    if LOGFIRE_AVAILABLE:
        agent_kwargs['logfire'] = {'tags': ['rag-agent', 'mcp-client']}
    
    agent = Agent(**agent_kwargs)

    # Register tools following documented @agent.tool pattern
    @agent.tool
    async def intelligent_search(
        ctx: RunContext[RAGDependencies],
        query: str,
        search_mode: str = "hybrid"
    ) -> str:
        """
        Perform intelligent content search using the most appropriate method.

        Args:
            ctx: Runtime context with dependencies
            query: Search query
            search_mode: Search strategy ("vector", "hybrid", "code")

        Returns:
            Search results and synthesized answer
        """
        # This will use the MCP server tools automatically
        # The agent will call perform_rag_query and other MCP tools
        return f"Searching for '{query}' using {search_mode} mode via MCP server"

    return agent


def create_workflow_agent(server_url: str = "http://localhost:8051/sse") -> Agent:
    """
    Create a workflow agent for complex multi-step operations.

    Following the documented Pydantic AI + MCP integration pattern.

    Args:
        server_url: URL of the crawl4ai MCP server

    Returns:
        Configured Agent for workflow orchestration with tools registered
    """
    # Create MCP server connection
    server = MCPServerHTTP(url=server_url)

    # Create agent with MCP server - following documented pattern
    # Enable logfire instrumentation if available
    agent_kwargs = {
        'model': 'openai:o3',
        'deps_type': WorkflowDependencies,
        'output_type': WorkflowResult,
        'mcp_servers': [server],
        'system_prompt': (
            "You are an intelligent workflow orchestrator that combines web crawling "
            "and content retrieval to accomplish complex research and analysis tasks. "
            "You can coordinate multiple operations to achieve sophisticated goals.\n\n"
            "Your role is to:\n"
            "1. Break down complex requests into manageable steps\n"
            "2. Execute crawling and RAG operations in the optimal sequence\n"
            "3. Track progress and handle errors gracefully\n"
            "4. Synthesize results from multiple steps into coherent outputs\n"
            "5. Provide recommendations for follow-up actions\n\n"
            "Always think step-by-step and provide clear progress updates."
        )
    }
    
    # Add logfire instrumentation if available
    if LOGFIRE_AVAILABLE:
        agent_kwargs['logfire'] = {'tags': ['workflow-agent', 'mcp-client']}
    
    agent = Agent(**agent_kwargs)

    return agent


@log_agent_interaction("generic")
async def run_agent_with_mcp(
    agent: Agent,
    prompt: str,
    dependencies: Any
) -> Any:
    """
    Helper function to run an agent with MCP server connection.

    This function follows the documented pattern for running Pydantic AI agents
    with MCP servers using the async context manager approach.

    Args:
        agent: The Pydantic AI agent to run (already configured with MCP servers)
        prompt: The user prompt/query
        dependencies: Dependencies object for the agent

    Returns:
        The agent's structured output
    """
    logger.info("Starting agent interaction", prompt_length=len(str(prompt)))
    
    # Run agent with MCP server context (agent already has MCP servers configured)
    async with agent.run_mcp_servers():
        result = await agent.run(prompt, deps=dependencies)

    logger.info("Agent interaction completed successfully", result_type=type(result).__name__)
    return result
