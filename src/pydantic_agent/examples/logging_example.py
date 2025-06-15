#!/usr/bin/env python3
"""
Example demonstrating comprehensive logfire logging integration.

This example shows how the logfire logging works across both MCP server tools
and Pydantic AI agent interactions, providing full observability.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from pydantic_agent.agent import create_crawl_agent, create_rag_agent, run_agent_with_mcp
from pydantic_agent.dependencies import CrawlDependencies, RAGDependencies
from logging_config import logger


async def demonstrate_logging_workflow():
    """
    Demonstrate a complete workflow with comprehensive logging.
    
    This will show logging at multiple levels:
    1. Agent creation and configuration
    2. Agent interaction start/completion
    3. MCP tool executions
    4. Database operations
    5. Error handling
    """
    
    print("🚀 Starting comprehensive logging demonstration...")
    
    # Create dependencies
    crawl_deps = CrawlDependencies(
        server_url="http://localhost:8051/sse",
        max_concurrent=3,
        chunk_size=2000
    )
    
    rag_deps = RAGDependencies(
        server_url="http://localhost:8051/sse",
        match_count=5,
        confidence_threshold=0.7
    )
    
    print("📋 Creating agents (check logfire for agent creation logs)...")
    
    # Create agents (will log agent creation)
    crawl_agent = create_crawl_agent()
    rag_agent = create_rag_agent()
    
    print("🌐 Starting crawl workflow (check logfire for detailed execution traces)...")
    
    # Demonstrate crawling workflow with logging
    crawl_prompt = """
    Please crawl the Python documentation homepage to gather information about Python features.
    Use the smart crawling approach to get comprehensive content.
    """
    
    try:
        logger.info("Starting crawl workflow demonstration", 
                   workflow_type="crawl_demo", 
                   target="python.org")
        
        # This will log:
        # - Agent interaction start/completion
        # - MCP tool executions (smart_crawl_url, etc.)
        # - Database operations (storage, embedding generation)
        # - Performance metrics
        crawl_result = await run_agent_with_mcp(crawl_agent, crawl_prompt, crawl_deps)
        
        logger.info("Crawl workflow completed successfully", 
                   result_type=type(crawl_result).__name__,
                   workflow_type="crawl_demo")
        
        print("✅ Crawl workflow completed (check logfire for execution details)")
        
    except Exception as e:
        logger.error("Crawl workflow failed", 
                    error=str(e), 
                    error_type=type(e).__name__,
                    workflow_type="crawl_demo")
        print(f"❌ Crawl workflow failed: {e}")
    
    print("🔍 Starting RAG query workflow (check logfire for search traces)...")
    
    # Demonstrate RAG workflow with logging
    rag_prompt = """
    Search for information about Python's async/await functionality.
    Provide a comprehensive summary with examples and best practices.
    """
    
    try:
        logger.info("Starting RAG workflow demonstration", 
                   workflow_type="rag_demo", 
                   query_topic="async/await")
        
        # This will log:
        # - Agent interaction start/completion  
        # - MCP tool executions (perform_rag_query, get_available_sources)
        # - Database operations (vector search, retrieval)
        # - Search performance metrics
        rag_result = await run_agent_with_mcp(rag_agent, rag_prompt, rag_deps)
        
        logger.info("RAG workflow completed successfully", 
                   result_type=type(rag_result).__name__,
                   workflow_type="rag_demo")
        
        print("✅ RAG workflow completed (check logfire for search details)")
        
    except Exception as e:
        logger.error("RAG workflow failed", 
                    error=str(e), 
                    error_type=type(e).__name__,
                    workflow_type="rag_demo")
        print(f"❌ RAG workflow failed: {e}")
    
    print("🎯 Logging demonstration completed!")
    print("\n📊 Check your logfire dashboard for comprehensive traces:")
    print("   🔗 https://logfire-eu.pydantic.dev/matzls/crawl4ai-agent")
    print("\n🔍 What you'll see in logfire:")
    print("   • Agent interaction traces with timing")
    print("   • MCP tool execution spans with parameters")
    print("   • Database operation metrics")
    print("   • Error handling and recovery")
    print("   • Performance profiling data")
    print("   • Structured metadata for filtering/searching")


async def demonstrate_error_logging():
    """
    Demonstrate error handling and logging capabilities.
    """
    print("\n🚨 Demonstrating error handling and logging...")
    
    # Test with invalid URL
    crawl_agent = create_crawl_agent()
    crawl_deps = CrawlDependencies(
        server_url="http://localhost:8051/sse",
        max_concurrent=1,
        chunk_size=1000
    )
    
    error_prompt = "Please crawl this invalid URL: https://this-domain-does-not-exist-12345.com"
    
    try:
        logger.info("Testing error handling workflow", 
                   test_type="error_demo", 
                   expected_outcome="failure")
        
        result = await run_agent_with_mcp(crawl_agent, error_prompt, crawl_deps)
        print("🤔 Unexpected success - check logfire for details")
        
    except Exception as e:
        logger.error("Expected error occurred in demo", 
                    error=str(e), 
                    error_type=type(e).__name__,
                    test_type="error_demo",
                    expected=True)
        print("✅ Error properly logged (check logfire for error traces)")


async def main():
    """Main demonstration function."""
    print("🔥 Logfire Logging Integration Demonstration")
    print("=" * 50)
    
    print("⚡ Prerequisites:")
    print("   1. MCP server should be running: ./start_mcp_server.sh")
    print("   2. LOGFIRE_TOKEN should be set in .env")
    print("   3. PostgreSQL should be running with crawl4ai_rag database")
    print()
    
    try:
        await demonstrate_logging_workflow()
        await demonstrate_error_logging()
        
    except Exception as e:
        logger.error("Demo failed", error=str(e), error_type=type(e).__name__)
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure the MCP server is running: ./start_mcp_server.sh")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)