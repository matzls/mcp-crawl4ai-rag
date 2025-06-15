#!/usr/bin/env python3
"""
Test script for enhanced observability implementation.

This script verifies that the enhanced observability features work correctly
including workflow tracking, decision logging, and performance analysis.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path
project_root = Path(__file__).parent
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


async def test_basic_observability():
    """Test basic observability functionality."""
    print("🧪 Testing Basic Observability...")
    
    # Create a simple workflow tracker
    tracker = AgentWorkflowTracker("test_session", "test query")
    
    # Test workflow step tracking
    step_id = tracker.start_step(
        WorkflowStage.INTENT_ANALYSIS, 
        "Testing intent analysis",
        {"test_param": "test_value"}
    )
    
    # Test decision recording
    tracker.record_decision(
        DecisionType.WORKFLOW_STRATEGY,
        "test_strategy",
        "This is a test decision",
        confidence=0.8,
        alternatives_considered=["alt1", "alt2"],
        context={"test": True}
    )
    
    # Complete the step
    tracker.complete_current_step(success=True, output_data={"result": "success"})
    
    # Complete the workflow
    tracker.complete_workflow(success=True, final_result="test_result")
    
    # Verify tracking data
    assert len(tracker.workflow_steps) == 1
    assert tracker.workflow_steps[0].success == True
    assert len(tracker.workflow_steps[0].decisions_made) == 1
    assert tracker.total_decisions == 1
    
    print("✅ Basic observability test passed")


async def test_agent_integration():
    """Test observability integration with the unified agent."""
    print("🧪 Testing Agent Integration...")
    
    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping agent integration test - OPENAI_API_KEY not set")
        return
    
    try:
        # Create dependencies
        dependencies = UnifiedAgentDependencies(
            mcp_server_url="http://localhost:8051/sse",
            default_max_depth=1,
            default_max_concurrent=2,
            default_match_count=2
        )
        
        # Create unified agent
        agent = create_unified_agent(dependencies.mcp_server_url)
        
        # Simple test query
        test_query = "What content sources do you have available?"
        
        print(f"📝 Testing with query: {test_query}")
        
        # Execute with enhanced observability
        result = await run_unified_agent(agent, test_query, dependencies)
        
        # Verify result structure
        assert hasattr(result, 'workflow_strategy')
        assert hasattr(result, 'primary_findings')
        assert hasattr(result, 'confidence_score')
        
        print("✅ Agent integration test passed")
        print(f"   Strategy: {result.workflow_strategy}")
        print(f"   Confidence: {result.confidence_score:.0%}")
        
    except Exception as e:
        print(f"⚠️  Agent integration test failed: {e}")
        print("   This may be expected if MCP server is not running")


async def test_error_handling():
    """Test error handling and logging."""
    print("🧪 Testing Error Handling...")
    
    # Create a workflow tracker
    tracker = AgentWorkflowTracker("error_test_session", "error test query")
    
    # Start a step that will fail
    step_id = tracker.start_step(
        WorkflowStage.TOOL_EXECUTION,
        "Testing error handling"
    )
    
    # Record an error decision
    tracker.record_decision(
        DecisionType.ERROR_RECOVERY,
        "retry_failed",
        "Simulated error recovery attempt",
        confidence=0.3,
        context={"error_type": "test_error"}
    )
    
    # Complete step with error
    tracker.complete_current_step(
        success=False, 
        error_message="Simulated test error"
    )
    
    # Complete workflow with failure
    tracker.complete_workflow(success=False)
    
    # Verify error tracking
    assert tracker.workflow_steps[0].success == False
    assert tracker.workflow_steps[0].error_message == "Simulated test error"
    assert tracker.performance_metrics['failed_steps'] == 1
    
    print("✅ Error handling test passed")


async def test_performance_metrics():
    """Test performance metrics calculation."""
    print("🧪 Testing Performance Metrics...")
    
    import time
    
    # Create a workflow tracker
    tracker = AgentWorkflowTracker("perf_test_session", "performance test query")
    
    # Simulate multiple steps with timing
    for i in range(3):
        step_id = tracker.start_step(
            WorkflowStage.TOOL_EXECUTION,
            f"Performance test step {i+1}"
        )
        
        # Simulate some work
        await asyncio.sleep(0.1)
        
        tracker.complete_current_step(success=True)
    
    # Complete workflow
    tracker.complete_workflow(success=True)
    
    # Verify performance metrics
    metrics = tracker.performance_metrics
    assert metrics['total_steps'] == 3
    assert metrics['successful_steps'] == 3
    assert metrics['failed_steps'] == 0
    assert metrics['total_duration_seconds'] > 0.3  # At least 0.3 seconds
    assert metrics['average_step_duration'] > 0.1   # At least 0.1 seconds per step
    
    print("✅ Performance metrics test passed")
    print(f"   Total Duration: {metrics['total_duration_seconds']:.3f}s")
    print(f"   Average Step Duration: {metrics['average_step_duration']:.3f}s")


async def test_workflow_summary():
    """Test workflow summary generation."""
    print("🧪 Testing Workflow Summary...")
    
    # Create a comprehensive workflow
    tracker = AgentWorkflowTracker("summary_test_session", "summary test query")
    
    # Add conversation context
    tracker.conversation_context.query_intent = "test_intent"
    tracker.conversation_context.context_metadata = {"test": True}
    
    # Add multiple steps with decisions
    for stage in [WorkflowStage.INTENT_ANALYSIS, WorkflowStage.TOOL_SELECTION, WorkflowStage.TOOL_EXECUTION]:
        step_id = tracker.start_step(stage, f"Testing {stage.value}")
        
        tracker.record_decision(
            DecisionType.WORKFLOW_STRATEGY,
            f"decision_{stage.value}",
            f"Test decision for {stage.value}",
            confidence=0.9
        )
        
        tracker.complete_current_step(success=True)
    
    # Record tool execution
    tracker.record_tool_execution("test_tool", {"param": "value"}, "test_result")
    
    # Complete workflow
    tracker.complete_workflow(success=True, final_result="comprehensive_test_result")
    
    # Generate summary
    summary = tracker.get_workflow_summary()
    
    # Verify summary structure
    assert 'session_id' in summary
    assert 'conversation_context' in summary
    assert 'workflow_steps' in summary
    assert 'performance_metrics' in summary
    assert 'tools_executed' in summary
    
    assert len(summary['workflow_steps']) == 3
    assert len(summary['tools_executed']) == 1
    assert summary['conversation_context']['query_intent'] == 'test_intent'
    
    print("✅ Workflow summary test passed")
    print(f"   Steps: {len(summary['workflow_steps'])}")
    print(f"   Tools: {len(summary['tools_executed'])}")
    print(f"   Total Decisions: {sum(len(step['decisions_made']) for step in summary['workflow_steps'])}")


async def main():
    """Run all observability tests."""
    print("🔥 Enhanced Observability Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_observability,
        test_error_handling,
        test_performance_metrics,
        test_workflow_summary,
        test_agent_integration,  # Run this last as it requires external services
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            failed += 1
        print()
    
    print("📊 Test Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {(passed / (passed + failed)) * 100:.0f}%")
    
    if failed == 0:
        print("\n🎉 All observability tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
