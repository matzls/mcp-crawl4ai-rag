#!/bin/bash

# MCP Server Stdio Runner
# This script ensures the virtual environment is activated and runs the MCP server in stdio mode

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Virtual environment path
VENV_PATH="$PROJECT_ROOT/crawl_venv"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at: $VENV_PATH" >&2
    echo "Please run: uv venv crawl_venv && uv pip install -e ." >&2
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Load environment variables from .env file if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a  # automatically export all variables
    source "$PROJECT_ROOT/.env"
    set +a  # stop automatically exporting
fi

# Force stdio transport
export TRANSPORT=stdio

# Change to project directory to ensure relative paths work
cd "$PROJECT_ROOT"

# Run the MCP server in stdio mode
exec python src/crawl4ai_mcp.py