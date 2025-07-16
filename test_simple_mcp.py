#!/usr/bin/env python3
"""
Simple test to verify basic MCP server functionality.
"""

import asyncio
import sys
import subprocess
import time
from pathlib import Path
import pytest

# Add the src directory to the path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


@pytest.mark.asyncio
async def test_mcp_server_basic():
    """Test basic MCP server startup and response."""
    print("🚀 Testing basic MCP server functionality")
    print("=" * 50)
    
    # Start server in stdio mode to avoid port conflicts
    venv_python = str(project_root / "crawl_venv" / "bin" / "python")
    server_script = str(project_root / "src" / "crawl4ai_mcp.py")
    
    try:
        print(f"🔧 Starting server: TRANSPORT=stdio {venv_python} {server_script}")
        
        # Start the server process
        process = subprocess.Popen(
            [venv_python, server_script],
            env={"TRANSPORT": "stdio"},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if process is running
        if process.poll() is None:
            print("✅ MCP server started successfully (stdio mode)")
            
            # Try to send a basic MCP initialization message
            init_message = """{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test-client", "version": "1.0.0"}}}"""
            
            try:
                # Send initialization
                process.stdin.write(init_message + "\n")
                process.stdin.flush()
                
                # Wait for response (with timeout)
                import select
                import sys
                
                # Read response with timeout
                response = ""
                start_time = time.time()
                while time.time() - start_time < 10:  # 10 second timeout
                    if process.stdout.readable():
                        line = process.stdout.readline()
                        if line:
                            response += line
                            if "result" in line or "error" in line:
                                break
                    time.sleep(0.1)
                
                if response:
                    print(f"✅ Server responded to initialization: {response[:100]}...")
                    success = True
                else:
                    print("❌ No response from server")
                    success = False
                    
            except Exception as e:
                print(f"❌ Communication error: {e}")
                success = False
                
        else:
            print("❌ Server failed to start")
            stderr_output = process.stderr.read()
            print(f"Error output: {stderr_output}")
            success = False
            
        # Cleanup
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
            
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def main():
    """Main test execution."""
    success = await test_mcp_server_basic()
    
    if success:
        print("\n🎉 Basic MCP server test successful!")
        print("The server can start and respond to basic MCP protocol messages.")
    else:
        print("\n💥 Basic MCP server test failed!")
        print("There are fundamental issues with the MCP server implementation.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)