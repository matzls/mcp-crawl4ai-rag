#!/usr/bin/env python3
"""
Unified Agent Example - Single Intelligent Orchestrator

This example demonstrates the new architecture with one intelligent agent
that can orchestrate all 5 MCP tools based on user intent and context.
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

from pydantic_agent.unified_agent import (
    create_unified_agent,
    run_unified_agent,
    UnifiedAgentDependencies
)
from logging_config import log_info, log_error


async def demonstrate_intelligent_orchestration():
    """
    Demonstrate how the unified agent intelligently selects and orchestrates tools.
    """
    print("🤖 Unified Agent Example - Intelligent Tool Orchestration")
    print("=" * 65)
    
    # Create dependencies
    deps = UnifiedAgentDependencies(
        mcp_server_url="http://localhost:8051/sse",
        default_max_depth=2,
        default_max_concurrent=5,
        default_match_count=5,
        enable_hybrid_search=True,
        enable_code_search=True
    )
    
    print(f"📡 Connecting to MCP server: {deps.mcp_server_url}")
    
    try:
        # Create the unified agent
        agent = create_unified_agent(deps.mcp_server_url)
        print("✅ Unified agent created successfully!")
        
        # Test scenarios that demonstrate intelligent tool selection
        test_scenarios = [
            {
                "name": "Research Workflow (URL mentioned)",
                "query": "Research Python async programming from https://docs.python.org/3/library/asyncio.html and provide a comprehensive summary with examples",
                "expected_tools": ["smart_crawl_url", "get_available_sources", "perform_rag_query"]
            },
            {
                "name": "Search Workflow (question about existing content)",
                "query": "What do you know about Python async programming? Explain the key concepts and show me code examples",
                "expected_tools": ["get_available_sources", "perform_rag_query", "search_code_examples"]
            },
            {
                "name": "Discovery Workflow (checking available content)",
                "query": "What content sources do you have available? What topics can you help me with?",
                "expected_tools": ["get_available_sources"]
            },
            {
                "name": "Code-Focused Workflow (specific code request)",
                "query": "Show me specific async function examples and error handling patterns in Python",
                "expected_tools": ["get_available_sources", "search_code_examples"]
            }
        ]
        
        # Execute test scenarios
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{'='*20} Scenario {i}: {scenario['name']} {'='*20}")
            print(f"Query: {scenario['query']}")
            print(f"Expected tools: {', '.join(scenario['expected_tools'])}")
            print()
            
            try:
                log_info(f"Starting scenario {i}", scenario_name=scenario['name'])
                
                # Run the unified agent
                result = await run_unified_agent(agent, scenario['query'], deps)
                
                # Display results
                print("🎯 Agent Response:")
                print("-" * 40)
                print(f"Intent: {result.query_intent}")
                print(f"Strategy: {result.workflow_strategy}")
                print(f"Steps executed: {len(result.steps_executed)}")
                print(f"Execution time: {result.total_execution_time_seconds:.2f}s")
                print(f"Success: {'✅' if result.workflow_success else '❌'}")
                print(f"Confidence: {result.confidence_score:.0%}")
                print()
                
                # Show workflow steps
                print("📋 Workflow Steps:")
                for step in result.steps_executed:
                    status = "✅" if step.tool_execution.success else "❌"
                    print(f"  {step.step_number}. {status} {step.tool_execution.tool_name} - {step.step_description}")
                    print(f"     Time: {step.tool_execution.execution_time_seconds:.2f}s")
                    if step.tool_execution.error_message:
                        print(f"     Error: {step.tool_execution.error_message}")
                print()
                
                # Show primary findings (truncated)
                print("💡 Primary Findings:")
                findings = result.primary_findings
                if len(findings) > 200:
                    findings = findings[:200] + "..."
                print(f"   {findings}")
                print()
                
                # Show sources accessed
                if result.sources_accessed:
                    print(f"🔗 Sources accessed: {', '.join(result.sources_accessed)}")
                
                # Show recommendations
                if result.recommendations:
                    print("💡 Recommendations:")
                    for rec in result.recommendations[:2]:
                        print(f"   • {rec}")
                
                print()
                
                logger.info(f"Scenario {i} completed", 
                           success=result.workflow_success,
                           steps=len(result.steps_executed),
                           confidence=result.confidence_score)
                
            except Exception as e:
                print(f"❌ Scenario {i} failed: {e}")
                log_error(f"Scenario {i} error", error=str(e), scenario=scenario['name'])
                continue
        
        print("\n🎉 Unified agent demonstration completed!")
        print("\n📊 Key Benefits Demonstrated:")
        print("• Single agent handles all scenarios intelligently")
        print("• Automatic tool selection based on user intent")
        print("• Comprehensive workflow orchestration")
        print("• Detailed execution tracking and results synthesis")
        print("• Confidence scoring and recommendations")
        
    except Exception as e:
        print(f"❌ Failed to create unified agent: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure MCP server is running: ./start_mcp_server.sh")
        print("2. Check PostgreSQL is running and accessible")
        print("3. Verify OpenAI API key is configured")


async def demonstrate_interactive_session():
    """
    Demonstrate an interactive session showing conversation context.
    """
    print("\n" + "="*65)
    print("🗣️  Interactive Session Demonstration")
    print("="*65)
    
    deps = UnifiedAgentDependencies()
    agent = create_unified_agent()
    
    # Simulate a conversation flow
    conversation = [
        "What content do you have available?",
        "Research FastAPI documentation from https://fastapi.tiangolo.com",
        "Now show me code examples for creating API endpoints",
        "What are the best practices for error handling in FastAPI?"
    ]
    
    for i, query in enumerate(conversation, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {query}")
        
        try:
            result = await run_unified_agent(agent, query, deps)
            
            print(f"Agent: {result.primary_findings[:150]}...")
            print(f"Strategy: {result.workflow_strategy}")
            print(f"Tools used: {[step.tool_execution.tool_name for step in result.steps_executed]}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n💡 This demonstrates how the agent maintains context and")
    print("   adapts its tool selection based on conversation flow.")


async def main():
    """Main demonstration function."""
    print("🔥 Unified Agent Architecture Demonstration")
    print("This replaces the previous 3-agent approach with intelligent orchestration")
    print()
    
    # Check prerequisites
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found in environment")
        print("   Set it with: export OPENAI_API_KEY=your_key_here")
        return 1
    
    print("💡 Prerequisites:")
    print("   1. MCP server should be running: ./start_mcp_server.sh")
    print("   2. PostgreSQL should be running with crawl4ai_rag database")
    print("   3. OpenAI API key should be configured")
    print()
    
    try:
        # Run demonstrations
        await demonstrate_intelligent_orchestration()
        
        # Optional interactive session demo
        response = input("\n🎮 Run interactive session demo? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            await demonstrate_interactive_session()
        
        return 0
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        log_error("Demo failed", error=str(e), error_type=type(e).__name__)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)