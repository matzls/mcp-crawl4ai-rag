#!/bin/bash

# MCP Server Startup Script
# Optimized for fast startup using existing virtual environment

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Virtual environment path
VENV_PATH="$PROJECT_ROOT/crawl_venv"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at: $VENV_PATH"
    echo "Please run: uv venv crawl_venv && uv pip install -e ."
    exit 1
fi

# Activate virtual environment
echo "🚀 Starting MCP Server..."
source "$VENV_PATH/bin/activate"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    # More robust env loading that handles comments and special characters
    set -a  # automatically export all variables
    source "$PROJECT_ROOT/.env"
    set +a  # stop automatically exporting
fi

# Set default transport (SSE is default for better performance)
TRANSPORT=${TRANSPORT:-sse}

# Change to project directory
cd "$PROJECT_ROOT"

# Start the server
if [ "$TRANSPORT" = "sse" ]; then
    echo "🌐 SSE server starting on http://localhost:${PORT:-8051}/sse"
    echo "📋 Connect MCP Inspector to: http://localhost:${PORT:-8051}/sse"
    TRANSPORT=sse python src/crawl4ai_mcp.py
else
    echo "📡 Stdio server starting"
    echo "📋 Connect MCP Inspector with: npx @modelcontextprotocol/inspector $VENV_PATH/bin/python src/crawl4ai_mcp.py"
    python3 src/crawl4ai_mcp.py
fi
