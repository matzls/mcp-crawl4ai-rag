#!/usr/bin/env python3
"""
Test MCP tools using stdio transport to isolate SSE connection issues.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add the src directory to the path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession


async def test_mcp_stdio():
    """Test MCP connection via stdio transport."""
    print("🚀 Testing MCP via stdio transport")
    print("=" * 50)
    
    # Get the python path from the virtual environment
    venv_python = str(project_root / "crawl_venv" / "bin" / "python")
    server_script = str(project_root / "src" / "crawl4ai_mcp.py")
    
    server_params = StdioServerParameters(
        command=venv_python,
        args=[server_script],
        env={"TRANSPORT": "stdio"}  # Force stdio transport
    )
    
    try:
        print(f"🔧 Starting MCP server: {venv_python} {server_script}")
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                print("🔗 Initializing MCP session...")
                await session.initialize()
                print("✅ MCP session initialized successfully via stdio!")
                
                # Test listing tools
                print("\n📋 Testing list_tools...")
                tools = await session.list_tools()
                print(f"✅ Found {len(tools)} tools:")
                for tool in tools:
                    print(f"  • {tool.name}")
                
                # Test one simple tool
                print("\n🔧 Testing get_available_sources...")
                result = await session.call_tool("get_available_sources", {})
                print(f"✅ Tool executed successfully!")
                print(f"📄 Result type: {type(result)}")
                
                if hasattr(result, 'content') and result.content:
                    content = result.content[0] if isinstance(result.content, list) else result.content
                    if hasattr(content, 'text'):
                        response_text = content.text[:200] + "..." if len(content.text) > 200 else content.text
                        print(f"📋 Response: {response_text}")
                
                return True
                
    except Exception as e:
        print(f"❌ Stdio test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test execution."""
    success = await test_mcp_stdio()
    
    if success:
        print("\n🎉 STDIO transport test successful!")
        print("This confirms the MCP server works but SSE transport has connection issues.")
    else:
        print("\n💥 STDIO transport test failed!")
        print("This indicates a deeper MCP server implementation issue.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)