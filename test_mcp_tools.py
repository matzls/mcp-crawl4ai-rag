#!/usr/bin/env python3
"""
Test script for individual MCP tools using proper MCP protocol client.

This script tests each of the 5 MCP tools individually to verify they work
correctly and identify any connection stability issues.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add the src directory to the path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from mcp.client.sse import sse_client
from mcp import ClientSession


async def test_mcp_tool(session: ClientSession, tool_name: str, arguments: dict = None):
    """Test a single MCP tool and return results."""
    try:
        print(f"\n🔧 Testing tool: {tool_name}")
        print(f"📝 Arguments: {arguments or 'None'}")
        
        # Call the tool
        result = await session.call_tool(tool_name, arguments or {})
        
        print(f"✅ Tool '{tool_name}' executed successfully")
        print(f"📄 Result type: {type(result)}")
        
        # Try to extract content
        if hasattr(result, 'content') and result.content:
            content = result.content[0] if isinstance(result.content, list) else result.content
            if hasattr(content, 'text'):
                response_text = content.text[:500] + "..." if len(content.text) > 500 else content.text
                print(f"📋 Response: {response_text}")
            else:
                print(f"📋 Response: {content}")
        else:
            print(f"📋 Response: {result}")
            
        return {"success": True, "result": str(result)[:200], "error": None}
        
    except Exception as e:
        print(f"❌ Tool '{tool_name}' failed: {e}")
        return {"success": False, "result": None, "error": str(e)}


async def test_all_mcp_tools():
    """Test all 5 MCP tools individually."""
    print("🚀 Starting MCP Tools Testing")
    print("=" * 60)
    
    try:
        # Connect to SSE server - use URL directly
        async with sse_client("http://localhost:8051/sse") as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                print("🔗 Initializing MCP session...")
                await session.initialize()
                print("✅ MCP session initialized successfully")
                
                # List available tools first
                print("\n📋 Listing available tools...")
                try:
                    tools = await session.list_tools()
                    print(f"✅ Found {len(tools)} tools:")
                    for tool in tools:
                        print(f"  • {tool.name}: {tool.description}")
                except Exception as e:
                    print(f"❌ Failed to list tools: {e}")
                    return
                
                # Test results storage
                test_results = {}
                
                # Test 1: get_available_sources (no arguments)
                test_results["get_available_sources"] = await test_mcp_tool(
                    session, "get_available_sources"
                )
                
                # Test 2: crawl_single_page (requires URL)
                test_results["crawl_single_page"] = await test_mcp_tool(
                    session, "crawl_single_page", 
                    {"url": "https://httpbin.org/json"}
                )
                
                # Test 3: smart_crawl_url (requires URL, optional parameters)
                test_results["smart_crawl_url"] = await test_mcp_tool(
                    session, "smart_crawl_url", 
                    {"url": "https://httpbin.org/html", "max_depth": 1, "max_concurrent": 2}
                )
                
                # Test 4: perform_rag_query (requires query)
                test_results["perform_rag_query"] = await test_mcp_tool(
                    session, "perform_rag_query", 
                    {"query": "test query", "match_count": 3}
                )
                
                # Test 5: search_code_examples (requires query)
                test_results["search_code_examples"] = await test_mcp_tool(
                    session, "search_code_examples", 
                    {"query": "function", "match_count": 3}
                )
                
                # Generate summary report
                print("\n" + "=" * 60)
                print("📊 MCP TOOLS TEST SUMMARY")
                print("=" * 60)
                
                successful_tools = 0
                failed_tools = 0
                
                for tool_name, result in test_results.items():
                    if result["success"]:
                        print(f"✅ {tool_name}: PASS")
                        successful_tools += 1
                    else:
                        print(f"❌ {tool_name}: FAIL - {result['error']}")
                        failed_tools += 1
                
                print(f"\n📈 Results: {successful_tools}/{len(test_results)} tools passed")
                
                if failed_tools == 0:
                    print("🎉 All MCP tools working correctly!")
                else:
                    print(f"⚠️  {failed_tools} tools have issues that need investigation")
                
                return test_results
                
    except Exception as e:
        print(f"💥 Critical error connecting to MCP server: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure MCP server is running: ./start_mcp_server.sh")
        print("2. Check server is listening on http://localhost:8051/sse")
        print("3. Verify no firewall blocking localhost:8051")
        return None


async def main():
    """Main test execution."""
    try:
        results = await test_all_mcp_tools()
        
        if results:
            # Save results to file for analysis
            with open("mcp_tools_test_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Test results saved to: mcp_tools_test_results.json")
        
        return 0 if results and all(r["success"] for r in results.values()) else 1
        
    except Exception as e:
        print(f"💥 Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)