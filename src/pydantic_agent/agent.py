"""
Legacy agent utilities for crawl4ai MCP integration.

This module provides utility functions for MCP server connections and agent execution.
The main agent functionality has been moved to unified_agent.py which provides a single
intelligent orchestrator for all crawling, RAG, and workflow operations.

For new implementations, use the unified agent architecture instead of the legacy
three-agent approach that was removed in TASK-044.
"""

from typing import Any
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP

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


# Legacy agent functions have been removed as part of TASK-044 cleanup.
# The unified agent architecture in unified_agent.py provides all functionality
# through a single intelligent orchestrator that can handle crawling, RAG, and
# workflow operations automatically based on user intent.


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
