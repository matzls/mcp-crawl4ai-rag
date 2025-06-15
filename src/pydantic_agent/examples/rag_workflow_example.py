"""
RAG workflow example using Pydantic AI agent.

This example demonstrates how to use the RAGAgent for intelligent content
retrieval and question answering based on previously crawled content.

Prerequisites:
    1. MCP server running on localhost:8051
    2. PostgreSQL database with crawled content
    3. OpenAI API key configured
"""

import asyncio
import os
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai import Agent

from ..dependencies import RAGDependencies, WorkflowDependencies
from ..outputs import RAGResult, WorkflowResult
from ..agent import RAGAgent, WorkflowAgent


async def basic_rag_example():
    """
    Demonstrate basic RAG functionality with the Pydantic AI agent.
    """
    print("🧠 Basic RAG Example - Pydantic AI + Crawl4AI MCP")
    print("=" * 55)
    
    # Initialize dependencies
    deps = RAGDependencies(
        mcp_server_url="http://localhost:8051/sse",
        default_match_count=5,
        enable_hybrid_search=True,
        enable_code_search=True
    )
    
    print(f"📡 Connecting to MCP server: {deps.mcp_server_url}")
    
    try:
        # Create MCP server connection
        server = MCPServerHTTP(url=deps.mcp_server_url)
        
        # Configure RAG agent
        agent = Agent(
            'openai:o3',
            deps_type=RAGDependencies,
            output_type=RAGResult,
            mcp_servers=[server],
            system_prompt=(
                "You are an intelligent research assistant that helps users find "
                "and synthesize information from crawled web content. Provide "
                "accurate, well-sourced answers with confidence scores."
            )
        )
        
        # Example queries to test different search capabilities
        test_queries = [
            "What is Python's async/await syntax and how does it work?",
            "Show me examples of error handling in Python",
            "How do you create a web server with FastAPI?",
            "What are the best practices for database connections?"
        ]
        
        # Run agent with MCP server context
        async with agent.run_mcp_servers():
            for query in test_queries:
                print(f"\n🔍 Query: {query}")
                print("-" * 50)
                
                # Run the agent with the query
                result = await agent.run(
                    f"Please search for information about: {query}. "
                    f"Provide a comprehensive answer with source attribution.",
                    deps=deps
                )
                
                # Display structured results
                output = result.output
                print(f"✅ Success: {output.success}")
                print(f"🔍 Search mode: {output.search_mode}")
                print(f"📊 Results found: {output.results_found}")
                print(f"📚 Sources used: {', '.join(output.sources_used)}")
                print(f"🎯 Confidence: {output.confidence_score:.2f}" if output.confidence_score else "🎯 Confidence: N/A")
                print(f"⏱️  Processing time: {output.processing_time_seconds:.2f}s")
                print(f"🔄 Reranking applied: {output.reranking_applied}")
                
                print(f"\n💡 Answer:")
                print(f"   {output.answer}")
                
                if output.error_details:
                    print(f"❌ Error: {output.error_details}")
        
        print("\n✨ Basic RAG example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running example: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure MCP server is running: ./start_mcp_server.sh")
        print("2. Check that content has been crawled and stored")
        print("3. Verify PostgreSQL vector database is accessible")
        print("4. Confirm OpenAI API key is configured")


async def research_workflow_example():
    """
    Demonstrate a complete research workflow combining crawling and RAG.
    """
    print("\n🔬 Research Workflow Example")
    print("=" * 35)
    
    # Initialize workflow dependencies
    deps = WorkflowDependencies()
    
    try:
        # Create MCP server connection
        server = MCPServerHTTP(url=deps.crawl_deps.mcp_server_url)
        
        # Configure workflow agent
        agent = Agent(
            'openai:o3',
            deps_type=WorkflowDependencies,
            output_type=WorkflowResult,
            mcp_servers=[server],
            system_prompt=(
                "You are an intelligent research workflow orchestrator. "
                "Break down complex research tasks into steps, execute them "
                "systematically, and provide comprehensive results."
            )
        )
        
        # Example research task
        research_topic = "Python async programming best practices"
        target_urls = [
            "https://docs.python.org/3/library/asyncio.html",
            "https://realpython.com/async-io-python/"
        ]
        
        async with agent.run_mcp_servers():
            print(f"🎯 Research Topic: {research_topic}")
            print(f"🌐 Target URLs: {', '.join(target_urls)}")
            
            # Create a comprehensive research prompt
            prompt = f"""
            Conduct comprehensive research on "{research_topic}".
            
            Please follow this workflow:
            1. First, crawl these URLs for relevant content: {', '.join(target_urls)}
            2. Then, search the crawled content for specific information about async programming
            3. Finally, synthesize the findings into a comprehensive report
            
            Provide detailed progress updates and a final summary with recommendations.
            """
            
            result = await agent.run(prompt, deps=deps)
            
            # Display workflow results
            output = result.output
            print(f"\n📋 Workflow Results:")
            print(f"✅ Success: {output.success}")
            print(f"🏷️  Type: {output.workflow_type}")
            print(f"⏱️  Total time: {output.total_time_seconds:.2f}s")
            print(f"📝 Steps completed: {len(output.steps_completed)}")
            
            # Show workflow steps
            for i, step in enumerate(output.steps_completed, 1):
                print(f"\n   Step {i}: {step.step_name} ({step.step_type})")
                print(f"   ✅ Success: {step.success}")
                print(f"   📄 Output: {step.output_summary}")
                if step.error_details:
                    print(f"   ❌ Error: {step.error_details}")
            
            # Show final result
            print(f"\n🎯 Final Result:")
            if isinstance(output.final_result, str):
                print(f"   {output.final_result}")
            else:
                print(f"   Type: {type(output.final_result).__name__}")
            
            # Show recommendations
            if output.recommendations:
                print(f"\n💡 Recommendations:")
                for rec in output.recommendations:
                    print(f"   • {rec}")
        
        print("\n✨ Research workflow completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running workflow: {e}")


async def interactive_rag_example():
    """
    Interactive RAG example that lets users ask questions.
    """
    print("\n🎮 Interactive RAG Mode")
    print("=" * 25)
    print("Ask questions about crawled content (or 'quit' to exit)")
    
    deps = RAGDependencies()
    server = MCPServerHTTP(url=deps.mcp_server_url)
    
    agent = Agent(
        'openai:o3',
        deps_type=RAGDependencies,
        output_type=RAGResult,
        mcp_servers=[server],
        system_prompt="You are a helpful research assistant."
    )
    
    async with agent.run_mcp_servers():
        while True:
            try:
                query = input("\n❓ Ask a question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                print(f"🔍 Searching for: {query}")
                
                result = await agent.run(
                    f"Please answer this question based on crawled content: {query}",
                    deps=deps
                )
                
                output = result.output
                print(f"\n💡 Answer (confidence: {output.confidence_score:.2f}):")
                print(f"   {output.answer}")
                print(f"📚 Sources: {', '.join(output.sources_used)}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🔍 Checking prerequisites...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
    
    print("🚀 Starting RAG workflow examples...")
    
    # Run basic RAG example
    asyncio.run(basic_rag_example())
    
    # Run research workflow example
    try:
        response = input("\n🔬 Run research workflow example? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            asyncio.run(research_workflow_example())
    except KeyboardInterrupt:
        pass
    
    # Run interactive mode
    try:
        response = input("\n🎮 Run interactive RAG mode? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            asyncio.run(interactive_rag_example())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
