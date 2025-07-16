# Quick Start Guide

Get up and running with the Crawl4AI RAG MCP Server in under 10 minutes.

## Prerequisites

- **Python 3.12+** ([download](https://www.python.org/downloads/))
- **uv package manager** ([install](https://docs.astral.sh/uv/))
- **PostgreSQL 17** + **pgvector** ([install guide](https://github.com/pgvector/pgvector))
- **OpenAI API key** ([get key](https://platform.openai.com/api-keys))

## 5-Minute Setup

### 1. Install Dependencies
```bash
# Clone repository
git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
cd mcp-crawl4ai-rag

# Install uv (if not installed)
pip install uv

# Setup environment and dependencies
uv venv crawl_venv
source crawl_venv/bin/activate  # Windows: crawl_venv\Scripts\activate
uv pip install -e .
crawl4ai-setup
```

### 2. Setup Database
```bash
# macOS
brew install postgresql@17 pgvector
brew services start postgresql@17
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"

# Create database and schema
createdb crawl4ai_rag
psql -h localhost -U $(whoami) -d crawl4ai_rag -f crawled_pages.sql
```

### 3. Configure Environment
Create `.env` file:
```bash
# Required
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql://$(whoami):@localhost:5432/crawl4ai_rag

# MCP Server (defaults)
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse

# RAG Strategies (all default false)
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false
USE_RERANKING=false
```

### 4. Start Server
```bash
# Quick startup (recommended)
./start_mcp_server.sh

# Manual startup
python src/crawl4ai_mcp.py
```

### 5. Test Installation
```bash
# Install test dependencies
uv pip install -e ".[dev]"

# Run tests (should complete in <1 second)
pytest tests/ -v

# Test MCP Inspector
npx @modelcontextprotocol/inspector
# Connect to: http://localhost:8051/sse
```

## First Steps

### Test Crawling
1. Open MCP Inspector: `http://localhost:8051/sse`
2. Use `crawl_single_page` tool with: `https://example.com`
3. Verify content stored in database

### Test Search
1. Use `get_available_sources` to see crawled sources
2. Use `perform_rag_query` with query: `"example website"`
3. See semantic search results

### Interactive Testing
```bash
# Interactive CLI
python cli_chat.py

# Database browser
python db_browser.py
```

## Integration with MCP Clients

### Claude Desktop
Add to your MCP settings:
```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "transport": "sse",
      "url": "http://localhost:8051/sse"
    }
  }
}
```

### Windsurf
```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "transport": "sse", 
      "serverUrl": "http://localhost:8051/sse"
    }
  }
}
```

## Troubleshooting

### Database Issues
```bash
# Check PostgreSQL status
brew services list | grep postgresql

# Test connection
psql -d crawl4ai_rag -c "SELECT 1;"

# Verify pgvector
psql -d crawl4ai_rag -c "SELECT vector_dims('[1,2,3]'::vector);"
```

### Server Issues
```bash
# Check environment
source crawl_venv/bin/activate

# Verify dependencies
uv pip list | grep crawl4ai

# Check logs
tail -f server.log
```

### Test Failures
```bash
# Run specific test
pytest tests/unit/test_content_processor.py -v

# Check environment mocking
pytest tests/conftest.py -v
```

## Next Steps

1. **Read ONBOARDING.md** for comprehensive development guide
2. **Study MCP tools** in `src/mcp_crawl_tools.py` and `src/mcp_search_tools.py`
3. **Review architecture** in `src/mcp_server.py`
4. **Explore advanced RAG** by enabling strategies in `.env`
5. **Join development** by picking an area from the onboarding checklist

## Key Commands Reference

```bash
# Development
./start_mcp_server.sh              # Start MCP server
pytest tests/ -v                   # Run tests
ruff format src/ tests/             # Format code
mypy src/                          # Type checking

# Database
python db_browser.py               # Browse database
psql -d crawl4ai_rag              # Connect to database

# Testing
python cli_chat.py                # Interactive agent
npx @modelcontextprotocol/inspector # MCP tool testing
```

---

**🚀 You're ready to go!** The MCP server provides 5 powerful tools for web crawling and semantic search. Connect it to Claude Desktop, Windsurf, or any MCP-compatible AI agent to start using intelligent web crawling in your workflows.