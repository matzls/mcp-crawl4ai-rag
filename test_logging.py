#!/usr/bin/env python3
"""
Test script to verify logfire logging setup works correctly.

Run this script to test the logging configuration before running the full server.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from logging_config import (
    configure_logfire, 
    log_info,
    log_error,
    log_system_startup,
    log_crawling_operation,
    log_rag_query
)


async def test_logging_setup():
    """Test the logging configuration."""
    print("🧪 Testing logfire logging setup...")
    
    # Test system startup logging
    log_system_startup("test-service", "0.1.0")
    
    # Test basic logging
    log_info("Basic logging test", test_type="unit_test", success=True)
    
    # Test crawling operation logging
    log_crawling_operation("https://example.com", "test_crawl")
    
    # Test RAG query logging
    log_rag_query("test query", 5, "example.com")
    
    # Test error logging
    try:
        raise ValueError("This is a test error for logging")
    except Exception as e:
        log_error("Test error logging", error=str(e), error_type=type(e).__name__)
    
    # Test structured logging with complex data
    log_info("Complex data logging test", 
             user_data={"prompt": "test", "length": 42},
             metrics={"execution_time": 1.5, "tokens": 100},
             tags=["test", "logging", "verification"])
    
    print("✅ Logging tests completed. Check logfire dashboard for entries.")


async def test_decorators():
    """Test logging decorators."""
    from logging_config import log_mcp_tool_execution, log_agent_interaction
    
    @log_mcp_tool_execution("test_tool")
    async def mock_mcp_tool(url: str) -> dict:
        """Mock MCP tool for testing."""
        await asyncio.sleep(0.1)  # Simulate work
        return {"success": True, "url": url, "chunks_stored": 5}
    
    @log_agent_interaction("test_agent")
    async def mock_agent_run(prompt: str, dependencies=None) -> object:
        """Mock agent run for testing."""
        await asyncio.sleep(0.2)  # Simulate work
        
        class MockResult:
            def __init__(self):
                self.data = MockData()
        
        class MockData:
            def __init__(self):
                self.summary = "Test agent completed successfully"
                self.confidence_score = 0.95
        
        return MockResult()
    
    # Test MCP tool logging
    print("🔧 Testing MCP tool decorator...")
    result = await mock_mcp_tool("https://test.example.com")
    print(f"Tool result: {result}")
    
    # Test agent interaction logging
    print("🤖 Testing agent interaction decorator...")
    agent_result = await mock_agent_run("What is the meaning of life?")
    print(f"Agent result type: {type(agent_result).__name__}")
    
    print("✅ Decorator tests completed.")


async def main():
    """Main test function."""
    # Check if logfire token is configured
    logfire_token = os.getenv('LOGFIRE_TOKEN')
    if not logfire_token:
        print("⚠️  LOGFIRE_TOKEN not found in environment")
        print("   Logging will fall back to console mode")
    else:
        print(f"✅ LOGFIRE_TOKEN configured: {logfire_token[:20]}...")
    
    # Test configuration
    success = configure_logfire()
    if success:
        print("✅ Logfire configured successfully")
    else:
        print("⚠️  Logfire configuration failed, using fallback logging")
    
    # Run tests
    await test_logging_setup()
    await test_decorators()
    
    print("\n🎉 All logging tests completed!")
    print("💡 Tip: Check your logfire dashboard at https://logfire.pydantic.dev")


if __name__ == "__main__":
    asyncio.run(main())