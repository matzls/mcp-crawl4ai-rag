# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a production-ready Model Context Protocol (MCP) server that provides intelligent web crawling and RAG capabilities for AI agents and coding assistants. It integrates Crawl4AI for web scraping with PostgreSQL + pgvector for semantic search and provides 5 MCP tools for external AI agents.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
uv venv crawl_venv && source crawl_venv/bin/activate
uv pip install -e .
crawl4ai-setup

# Database setup (PostgreSQL 17 + pgvector required)
brew install postgresql@17 pgvector
brew services start postgresql@17
createdb crawl4ai_rag
psql -h localhost -U $(whoami) -d crawl4ai_rag -f crawled_pages.sql
```

### Running the Server
```bash
# Quick startup (recommended - 90% faster)
./start_mcp_server.sh

# Manual startup
python src/crawl4ai_mcp.py

# Using uv
uv run src/crawl4ai_mcp.py
```

### Testing Commands
```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Fast unit tests (recommended for development)
pytest tests/unit/test_content_processor.py -v --cov=src/content

# Test specific module with coverage report
pytest tests/unit/ -v --cov=src --cov-report=html

# Quick test validation
pytest tests/unit/test_content_processor.py -q
```

### Code Quality
```bash
# Format and lint code
ruff format src/ tests/ && ruff check src/ tests/

# Type checking
mypy src/

# MCP tool testing with MCP Inspector
npx @modelcontextprotocol/inspector
# Connect to: http://localhost:8051/sse
```

### Interactive Testing
```bash
# Interactive CLI with unified agent
python cli_chat.py

# Database browser (custom tool)
python db_browser.py
```

## Architecture Overview

### Core Components
- **`src/mcp_server.py`** - FastMCP server with SSE/stdio transport and lifespan management
- **`src/mcp_crawl_tools.py`** - Crawling tools (`crawl_single_page`, `smart_crawl_url`)
- **`src/mcp_search_tools.py`** - Search tools (`get_available_sources`, `perform_rag_query`, `search_code_examples`)
- **`src/crawl_strategies.py`** - URL detection and parallel processing strategies
- **`src/content_processing.py`** - Smart chunking and metadata extraction
- **`src/pydantic_agent/unified_agent.py`** - Single orchestrator agent for testing

### Modular Architecture
The project was refactored from a 1,121-line monolith into focused modules. Each file is <500 lines following the codebase standards. The main entry point `src/crawl4ai_mcp.py` imports from all modules for backward compatibility.

### Database Schema
PostgreSQL 17 with pgvector extension. Key tables:
- **`sources`** - Crawled source metadata with AI-generated summaries
- **`crawled_pages`** - Content chunks with vector embeddings for semantic search
- **`code_examples`** - Extracted code examples with AI summaries (if `USE_AGENTIC_RAG=true`)

### MCP Tools Provided
1. **`crawl_single_page(url)`** - Quick single page crawl and storage
2. **`smart_crawl_url(url, max_depth, max_concurrent, chunk_size)`** - Intelligent multi-page crawling with sitemap/txt detection
3. **`get_available_sources()`** - List crawled sources for discovery
4. **`perform_rag_query(query, source, match_count)`** - Semantic search with optional hybrid search and reranking
5. **`search_code_examples(query, source_id, match_count)`** - Code-specific search (when `USE_AGENTIC_RAG=true`)

## Environment Configuration

### Required Environment Variables
```bash
# Core configuration
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql://user:pass@localhost/crawl4ai_rag

# Server configuration
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse

# Optional RAG enhancements
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false
USE_RERANKING=false
```

### RAG Strategy Options
- **USE_CONTEXTUAL_EMBEDDINGS** - Enriches chunks with document context via LLM
- **USE_HYBRID_SEARCH** - Combines vector + keyword search
- **USE_AGENTIC_RAG** - Extracts and summarizes code examples (enables `search_code_examples` tool)
- **USE_RERANKING** - Cross-encoder reranking for better result ordering

## Testing Infrastructure

### Testing Architecture
Production-ready testing with centralized mocking in `tests/conftest.py`:
- **Auto-mocking** of OpenAI API and PostgreSQL operations
- **Zero external dependencies** during unit testing
- **Sub-second execution** (28 tests in <1 second)
- **97% coverage** achieved on content processor module

### Key Testing Files
- **`tests/conftest.py`** - Centralized mock configuration with auto-applied fixtures
- **`tests/unit/test_content_processor.py`** - Content processing tests (97% coverage)
- **`tests/unit/test_mcp_tools.py`** - MCP tool functionality tests
- **`pytest.ini`** - Comprehensive pytest configuration with asyncio support

## Development Patterns

### MCP Tool Development
```python
from mcp import mcp
from mcp.server.fastmcp import Context

@mcp.tool()
async def your_tool(ctx: Context, param: str) -> dict:
    """Tool description for AI agents."""
    # Access shared resources via context
    crawler = ctx.request_context.lifespan_context["crawler"]
    postgres_pool = ctx.request_context.lifespan_context["postgres_pool"]
    
    # Implement tool logic
    result = await your_logic(param, crawler, postgres_pool)
    
    return {"success": True, "result": result}
```

### Testing Patterns
All external dependencies are auto-mocked globally. Tests are fast, isolated, and include happy path, edge cases, and error scenarios.

### File Organization
- Keep modules <500 lines
- Separate concerns (crawling, search, content processing, database operations)
- Use clear imports and `__all__` exports
- Follow async/await patterns throughout

## Key Integration Points

### High-Risk Areas
- **OpenAI API** - Rate limits and embedding generation (production validated: 197 pages/42.5s)
- **PostgreSQL Vector Operations** - Large-scale storage and retrieval (validated: 856 chunks)
- **MCP Protocol Compliance** - SSE/stdio transport stability (all 5 tools verified)

### External Dependencies
- **crawl4ai==0.6.2** - Web crawling with JavaScript rendering
- **mcp>=1.9.4** - Official MCP Python SDK
- **asyncpg==0.30.0** - PostgreSQL async driver
- **openai>=1.86.0** - OpenAI API integration
- **pydantic-ai[logfire]>=0.2.18** - Agent framework with observability

## Common Tasks

### Adding New MCP Tools
1. Create tool function in appropriate module (`mcp_crawl_tools.py` or `mcp_search_tools.py`)
2. Add `@mcp.tool()` decorator and proper docstring
3. Add comprehensive tests with mocking
4. Update `crawl4ai_mcp.py` imports for backward compatibility

### Database Operations
- Use `utils.py` functions for database operations
- All operations are async with proper connection pooling
- Vector operations use pgvector extension
- Always handle PostgreSQL connection errors gracefully

### Testing New Features
1. Write unit tests in `tests/unit/`
2. Use existing mocking patterns from `conftest.py`
3. Ensure tests run in <1 second
4. Aim for >90% coverage on critical modules
5. Test both happy path and error scenarios

## Troubleshooting

### Common Issues
- **Database connection errors** - Verify PostgreSQL is running and `crawl4ai_rag` database exists
- **Test timeouts** - Ensure auto-mocking is working (should run in <1s)
- **MCP tool failures** - Test with MCP Inspector at `http://localhost:8051/sse`
- **Startup issues** - Use `./start_mcp_server.sh` for optimized startup

### Performance Notes
- Testing infrastructure provides 25x speedup (0.6s vs 2+ min) via proper mocking
- Server startup optimized by 90% using existing virtual environment
- Parallel crawling with configurable concurrency limits
- Database operations use connection pooling for efficiency