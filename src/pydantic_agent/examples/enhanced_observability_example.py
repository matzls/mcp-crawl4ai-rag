#!/usr/bin/env python3
"""
Enhanced Observability Example for Pydantic AI Agent

This example demonstrates the comprehensive observability features including:
- Detailed agent workflow tracking
- Decision-making process visibility
- Tool orchestration monitoring
- Performance analysis and metrics
- Error context and recovery tracking
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path
project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from pydantic_agent.unified_agent import (
    create_unified_agent, 
    run_unified_agent, 
    UnifiedAgentDependencies
)
from observability import AgentWorkflowTracker, WorkflowStage, DecisionType
from logging_config import logger


async def demonstrate_research_workflow():
    """
    Demonstrate research workflow with comprehensive observability.
    
    This workflow shows:
    - Intent analysis for URL-based queries
    - Tool selection reasoning
    - Crawling and search orchestration
    - Performance metrics and decision tracking
    """
    print("🔬 Demonstrating Research Workflow with Enhanced Observability")
    print("=" * 60)
    
    # Create dependencies
    dependencies = UnifiedAgentDependencies(
        mcp_server_url="http://localhost:8051/sse",
        default_max_depth=2,
        default_max_concurrent=5,
        default_match_count=3,
        enable_hybrid_search=True,
        enable_code_search=True
    )
    
    # Create unified agent
    agent = create_unified_agent(dependencies.mcp_server_url)
    
    # Research query with URL
    research_query = """
    Please research Python's asyncio documentation from https://docs.python.org/3/library/asyncio.html
    and then answer: What are the key concepts for async programming in Python?
    """
    
    print(f"📝 Query: {research_query[:100]}...")
    print("🔍 Check Logfire dashboard for detailed workflow traces")
    print()
    
    try:
        # Execute with enhanced observability
        result = await run_unified_agent(agent, research_query, dependencies)
        
        print("✅ Research workflow completed successfully!")
        print(f"📊 Workflow Strategy: {result.workflow_strategy}")
        print(f"⏱️  Total Execution Time: {result.total_execution_time_seconds:.2f}s")
        print(f"🔧 Steps Executed: {len(result.steps_executed)}")
        print(f"🎯 Confidence Score: {result.confidence_score:.0%}")
        print(f"📚 Sources Accessed: {len(result.sources_accessed)}")
        
        if result.partial_failures:
            print(f"⚠️  Partial Failures: {len(result.partial_failures)}")
            for failure in result.partial_failures[:2]:
                print(f"   • {failure}")
        
        print(f"\n💡 Primary Findings Preview:")
        print(f"   {result.primary_findings[:200]}...")
        
    except Exception as e:
        print(f"❌ Research workflow failed: {e}")
        logger.error("Research workflow demonstration failed", 
                    error=str(e), error_type=type(e).__name__)


async def demonstrate_search_workflow():
    """
    Demonstrate search workflow with decision tracking.
    
    This workflow shows:
    - Question-based intent analysis
    - Source availability checking
    - Search strategy selection
    - Result synthesis and confidence assessment
    """
    print("\n🔍 Demonstrating Search Workflow with Decision Tracking")
    print("=" * 60)
    
    dependencies = UnifiedAgentDependencies(
        mcp_server_url="http://localhost:8051/sse",
        default_match_count=5,
        enable_hybrid_search=True
    )
    
    agent = create_unified_agent(dependencies.mcp_server_url)
    
    # Search query without URLs
    search_query = """
    What are the best practices for error handling in Python async functions?
    Show me some code examples if available.
    """
    
    print(f"📝 Query: {search_query}")
    print("🔍 Check Logfire dashboard for decision-making traces")
    print()
    
    try:
        result = await run_unified_agent(agent, search_query, dependencies)
        
        print("✅ Search workflow completed successfully!")
        print(f"📊 Workflow Strategy: {result.workflow_strategy}")
        print(f"⏱️  Total Execution Time: {result.total_execution_time_seconds:.2f}s")
        print(f"🎯 Confidence Score: {result.confidence_score:.0%}")
        
        if result.supporting_evidence:
            print(f"📋 Supporting Evidence ({len(result.supporting_evidence)} items):")
            for i, evidence in enumerate(result.supporting_evidence[:3], 1):
                print(f"   {i}. {evidence[:100]}...")
        
        if result.recommendations:
            print(f"💡 Recommendations:")
            for rec in result.recommendations[:2]:
                print(f"   • {rec}")
        
    except Exception as e:
        print(f"❌ Search workflow failed: {e}")
        logger.error("Search workflow demonstration failed", 
                    error=str(e), error_type=type(e).__name__)


async def demonstrate_error_handling_observability():
    """
    Demonstrate error handling with comprehensive context tracking.
    
    This shows:
    - Error detection and categorization
    - Recovery decision making
    - Context preservation during failures
    - Performance impact analysis
    """
    print("\n🚨 Demonstrating Error Handling Observability")
    print("=" * 60)
    
    dependencies = UnifiedAgentDependencies(
        mcp_server_url="http://localhost:8051/sse",
        default_max_concurrent=1
    )
    
    agent = create_unified_agent(dependencies.mcp_server_url)
    
    # Query with invalid URL to trigger error
    error_query = """
    Please crawl this invalid website: https://this-domain-definitely-does-not-exist-12345.invalid
    and summarize its content.
    """
    
    print(f"📝 Query: {error_query[:80]}...")
    print("🔍 Check Logfire dashboard for error context and recovery traces")
    print()
    
    try:
        result = await run_unified_agent(agent, error_query, dependencies)
        
        # Even if it "succeeds", check for partial failures
        if result.partial_failures:
            print("⚠️  Workflow completed with partial failures:")
            for failure in result.partial_failures:
                print(f"   • {failure}")
        else:
            print("🤔 Unexpected success - check logs for details")
        
    except Exception as e:
        print(f"✅ Expected error properly handled and logged: {type(e).__name__}")
        print("🔍 Check Logfire for detailed error context and recovery attempts")


async def demonstrate_performance_analysis():
    """
    Demonstrate performance analysis and optimization insights.
    
    This shows:
    - Execution timing analysis
    - Tool performance comparison
    - Bottleneck identification
    - Resource usage tracking
    """
    print("\n⚡ Demonstrating Performance Analysis")
    print("=" * 60)
    
    dependencies = UnifiedAgentDependencies(
        mcp_server_url="http://localhost:8051/sse",
        default_max_concurrent=10,  # High concurrency for performance test
        default_match_count=10      # More results for comprehensive search
    )
    
    agent = create_unified_agent(dependencies.mcp_server_url)
    
    # Complex query that exercises multiple tools
    performance_query = """
    First, check what content sources are available.
    Then search for information about Python async programming patterns.
    Finally, find specific code examples for async/await usage.
    """
    
    print(f"📝 Query: {performance_query[:80]}...")
    print("🔍 Check Logfire dashboard for performance metrics and timing analysis")
    print()
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        result = await run_unified_agent(agent, performance_query, dependencies)
        
        total_time = asyncio.get_event_loop().time() - start_time
        
        print("✅ Performance analysis completed!")
        print(f"⏱️  Total Wall Time: {total_time:.2f}s")
        print(f"⏱️  Agent Execution Time: {result.total_execution_time_seconds:.2f}s")
        print(f"🔧 Steps Executed: {len(result.steps_executed)}")
        print(f"📊 Efficiency Ratio: {(result.total_execution_time_seconds / total_time):.0%}")
        
        # Analyze step performance
        if result.steps_executed:
            print(f"\n📈 Step Performance Analysis:")
            for i, step in enumerate(result.steps_executed[:3], 1):
                print(f"   {i}. {step.step_description[:50]}...")
                print(f"      ⏱️  Duration: {step.tool_execution.execution_time_seconds:.2f}s")
                print(f"      ✅ Success: {step.tool_execution.success}")
        
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")


async def main():
    """Main demonstration function."""
    print("🔥 Enhanced Observability Demonstration for Pydantic AI Agent")
    print("=" * 70)
    
    print("⚡ Prerequisites:")
    print("   1. MCP server running: ./start_mcp_server.sh")
    print("   2. LOGFIRE_TOKEN set in .env file")
    print("   3. PostgreSQL running with crawl4ai_rag database")
    print("   4. OpenAI API key configured")
    print()
    
    print("📊 What you'll see in Logfire dashboard:")
    print("   • Complete agent workflow traces with nested spans")
    print("   • Decision-making process with reasoning and alternatives")
    print("   • Tool selection logic and parameter choices")
    print("   • Performance metrics and timing analysis")
    print("   • Error context and recovery attempts")
    print("   • Conversation context and user intent analysis")
    print()
    
    try:
        # Run all demonstrations
        await demonstrate_research_workflow()
        await demonstrate_search_workflow()
        await demonstrate_error_handling_observability()
        await demonstrate_performance_analysis()
        
        print("\n🎯 Enhanced Observability Demonstration Completed!")
        print("🔗 Check your Logfire dashboard for comprehensive traces:")
        print("   https://logfire-eu.pydantic.dev/matzls/crawl4ai-agent")
        
    except Exception as e:
        logger.error("Enhanced observability demonstration failed", 
                    error=str(e), error_type=type(e).__name__)
        print(f"❌ Demonstration failed: {e}")
        print("💡 Make sure all prerequisites are met")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
