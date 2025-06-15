"""
Basic crawling example using Pydantic AI agent.

This example demonstrates the fundamental pattern for using the CrawlAgent
to crawl web content and store it in the vector database.

Prerequisites:
    1. MCP server running on localhost:8051
    2. PostgreSQL database with pgvector extension
    3. OpenAI API key configured
"""

import asyncio
import os
from dotenv import load_dotenv
from pydantic_ai.mcp import MCPServerSSE
from pydantic_ai import Agent

from ..dependencies import CrawlDependencies
from ..outputs import CrawlResult


async def basic_crawl_example():
    """
    Demonstrate basic crawling functionality with the Pydantic AI agent.
    """
    print("🕷️  Basic Crawl Example - Pydantic AI + Crawl4AI MCP")
    print("=" * 60)
    
    # Initialize dependencies with default configuration
    deps = CrawlDependencies(
        mcp_server_url="http://localhost:8051/sse",
        default_max_depth=2,
        default_max_concurrent=5
    )
    
    print(f"📡 Connecting to MCP server: {deps.mcp_server_url}")
    
    try:
        # Create MCP server connection following documented pattern
        server = MCPServerSSE(url=deps.mcp_server_url)
        
        # Configure agent with MCP server
        agent = Agent(
            'openai:gpt-4-turbo',
            deps_type=CrawlDependencies,
            output_type=CrawlResult,
            mcp_servers=[server],
            system_prompt=(
                "You are an intelligent web crawling assistant. "
                "Help users crawl web content efficiently using the available tools."
            )
        )
        
        # Example crawling tasks
        test_urls = [
            "https://docs.python.org/3/tutorial/introduction.html",
            "https://example.com"
        ]
        
        # Run agent with MCP server context
        async with agent.run_mcp_servers():
            for url in test_urls:
                print(f"\n🌐 Crawling: {url}")
                print("-" * 40)
                
                # Run the agent with a natural language prompt
                result = await agent.run(
                    f"Please crawl the URL '{url}' using the smart crawling strategy. "
                    f"Analyze the content and provide a detailed summary of what was found.",
                    deps=deps
                )
                
                # Display structured results
                output = result.output
                print(f"✅ Success: {output.success}")
                print(f"📄 Pages crawled: {output.pages_crawled}")
                print(f"📦 Chunks stored: {output.chunks_stored}")
                print(f"💾 Code examples: {output.code_examples_stored}")
                print(f"⏱️  Processing time: {output.processing_time_seconds:.2f}s")
                print(f"📝 Summary: {output.summary}")
                
                if output.error_details:
                    print(f"❌ Error: {output.error_details}")
                
                print(f"🔗 URLs processed: {', '.join(output.urls_processed[:3])}")
                if len(output.urls_processed) > 3:
                    print(f"    ... and {len(output.urls_processed) - 3} more")
        
        print("\n✨ Basic crawl example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running example: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure MCP server is running: ./start_mcp_server.sh")
        print("2. Check PostgreSQL is running and accessible")
        print("3. Verify OpenAI API key is configured")
        print("4. Confirm network connectivity to target URLs")


async def interactive_crawl_example():
    """
    Interactive example that lets users input URLs to crawl.
    """
    print("\n🎮 Interactive Crawl Mode")
    print("=" * 30)
    print("Enter URLs to crawl (or 'quit' to exit)")
    
    deps = CrawlDependencies()
    server = MCPServerSSE(url=deps.mcp_server_url)
    
    agent = Agent(
        'openai:o3',
        deps_type=CrawlDependencies,
        output_type=CrawlResult,
        mcp_servers=[server],
        system_prompt="You are a helpful web crawling assistant."
    )
    
    async with agent.run_mcp_servers():
        while True:
            try:
                url = input("\n🌐 Enter URL to crawl: ").strip()
                
                if url.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not url:
                    continue
                
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                
                print(f"🕷️  Crawling {url}...")
                
                result = await agent.run(
                    f"Crawl {url} and provide a summary of the content found.",
                    deps=deps
                )
                
                output = result.output
                print(f"✅ Crawled {output.pages_crawled} pages")
                print(f"📦 Stored {output.chunks_stored} content chunks")
                print(f"📝 {output.summary}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Check if MCP server is likely running
    print("🔍 Checking prerequisites...")

    # Basic environment check
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
        print("   Set it with: export OPENAI_API_KEY=your_key_here")
    
    print("🚀 Starting basic crawl example...")
    
    # Run the basic example
    asyncio.run(basic_crawl_example())
    
    # Optionally run interactive mode
    try:
        response = input("\n🎮 Run interactive mode? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            asyncio.run(interactive_crawl_example())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
