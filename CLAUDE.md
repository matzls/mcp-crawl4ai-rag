# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Crawl4AI RAG MCP Server** - A Model Context Protocol (MCP) server that provides web crawling and RAG capabilities for AI agents and AI coding assistants. Built with Crawl4AI, PostgreSQL, and OpenAI embeddings.

## Essential Development Commands

### Environment Setup
```bash
# Using Docker (Recommended)
docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag

# Using uv directly
uv venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
uv pip install -e .
crawl4ai-setup
```

### Running the Server
```bash
# Run with SSE transport (default)
uv run src/crawl4ai_mcp.py

# Run with stdio transport
TRANSPORT=stdio uv run src/crawl4ai_mcp.py
```

### Database Setup
```bash
# Set up PostgreSQL database with pgvector extension
# 1. Install PostgreSQL and pgvector:
#    brew install postgresql@16 pgvector  # macOS
#    # OR use Docker: docker run --name crawl4ai-postgres -e POSTGRES_PASSWORD=mypassword -e POSTGRES_DB=crawl4ai_rag -p 5432:5432 -d pgvector/pgvector:pg16
# 2. Create database and run schema:
#    psql -h localhost -U postgres -d crawl4ai_rag -f crawled_pages.sql
```

### Testing and Quality
```bash
# No formal test suite currently exists
# Manual testing via MCP client integration
# Future: Add pytest test suite for core functionality
```

## Architecture Overview

### Core Components

**MCP Server (`src/crawl4ai_mcp.py`)**
- FastMCP server with lifespan management for AsyncWebCrawler and PostgreSQL connection pool
- Five main tools: `crawl_single_page`, `smart_crawl_url`, `get_available_sources`, `perform_rag_query`, `search_code_examples`
- Configurable RAG strategies via environment variables
- Cross-encoder reranking model for improved search results

**Utilities (`src/utils.py`)**
- PostgreSQL connection pool management and database operations
- OpenAI embedding generation with batch processing and retry logic
- Content chunking with smart markdown-aware splitting
- Code block extraction and summarization
- Hybrid search combining vector and keyword search
- Contextual embedding generation for improved retrieval

**Database Schema (`crawled_pages.sql`)**
- `crawled_pages` table: Stores chunked content with embeddings
- `code_examples` table: Stores extracted code examples with summaries  
- `sources` table: Tracks crawled sources with metadata
- PostgreSQL functions for vector similarity search using pgvector

### Data Flow

1. **Content Ingestion**: URLs are crawled using Crawl4AI with intelligent strategy detection (sitemap, text file, or recursive crawling)
2. **Content Processing**: Markdown content is chunked intelligently, respecting code blocks and paragraphs
3. **Embedding Generation**: Text chunks are embedded using OpenAI's text-embedding-3-small model, optionally with contextual enrichment
4. **Storage**: Chunks and embeddings stored in PostgreSQL with metadata for filtering and retrieval
5. **Code Analysis**: Code blocks (≥300 chars) are extracted, summarized, and stored separately when agentic RAG is enabled
6. **Search & Retrieval**: Vector similarity search with optional hybrid search, reranking, and source filtering

### RAG Strategy Configuration

**Environment Variables Control Advanced Features:**
- `USE_CONTEXTUAL_EMBEDDINGS`: Enhances chunk embeddings with document context via LLM
- `USE_HYBRID_SEARCH`: Combines vector search with keyword matching
- `USE_AGENTIC_RAG`: Enables code example extraction and dedicated code search
- `USE_RERANKING`: Applies cross-encoder reranking to improve result relevance

### Key Architectural Decisions

**Why AsyncWebCrawler**: Enables high-performance concurrent crawling with memory-adaptive dispatching
**Why FastMCP**: Provides robust MCP server framework with lifespan management for resource cleanup
**Why PostgreSQL**: Local PostgreSQL with pgvector extension offers scalable vector storage and search with full control
**Why Smart Chunking**: Respects content structure (code blocks, paragraphs) rather than simple character limits
**Why Batch Processing**: Reduces API calls and improves performance for embedding generation

## Configuration

### Required Environment Variables
```bash
# Core Configuration
OPENAI_API_KEY=your_openai_api_key
MODEL_CHOICE=gpt-4o-mini  # For summaries and contextual embeddings

# PostgreSQL Configuration (Option 1: Use DATABASE_URL)
DATABASE_URL=postgresql://username:password@localhost:5432/crawl4ai_rag

# PostgreSQL Configuration (Option 2: Individual components)
# POSTGRES_HOST=localhost
# POSTGRES_PORT=5432
# POSTGRES_DB=crawl4ai_rag
# POSTGRES_USER=postgres
# POSTGRES_PASSWORD=your_postgres_password

# Transport Configuration
TRANSPORT=sse  # or stdio
HOST=0.0.0.0   # for SSE transport
PORT=8051      # for SSE transport

# RAG Strategy Toggles (all default to false)
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false  
USE_RERANKING=false
```

### Integration Patterns

**SSE Transport Configuration:**
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

**Stdio Transport Configuration:**
```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "python",
      "args": ["path/to/crawl4ai-mcp/src/crawl4ai_mcp.py"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_key",
        "DATABASE_URL": "postgresql://username:password@localhost:5432/crawl4ai_rag"
      }
    }
  }
}
```

## Development Guidelines

### Code Organization
- `src/crawl4ai_mcp.py`: Main MCP server with tool definitions
- `src/utils.py`: Database operations, embedding generation, content processing
- `crawled_pages.sql`: Database schema and functions
- `.env.example`: Configuration template
- `Dockerfile`: Container deployment configuration

### Key Patterns

**Error Handling**: All database operations include retry logic with exponential backoff
**Batch Processing**: Embeddings and database operations are batched for performance
**Resource Management**: AsyncWebCrawler and PostgreSQL connection pool lifecycle managed via FastMCP lifespan context
**Parallel Processing**: Code example summarization uses ThreadPoolExecutor for performance

### Testing Strategy
- Manual testing via MCP client integration
- Future: Add pytest suite covering core utilities and MCP tool functionality
- Database operations should be tested against real PostgreSQL instance
- Crawling functionality requires network access for comprehensive testing

### Performance Considerations
- Default batch size of 20 for database operations balances performance and memory
- Concurrent crawling limited by `max_concurrent` parameter (default 10)
- Memory-adaptive dispatcher prevents browser resource exhaustion
- Cross-encoder reranking adds ~100-200ms latency but improves relevance

### Security Notes
- Database credentials stored in environment variables only
- No user authentication - intended for trusted MCP client environments
- Database access through local PostgreSQL connection with configurable permissions
- OpenAI API key required for embedding generation

### Database Inspection Tools
With local PostgreSQL, you can inspect the database using:
- **pgAdmin**: Web-based PostgreSQL administration tool
- **DBeaver**: Free universal database client with excellent PostgreSQL support
- **TablePlus**: Modern native database client (macOS/Windows)
- **Command Line**: Direct `psql` access for SQL queries and administration
- **IDE Extensions**: PostgreSQL plugins for VS Code, IntelliJ, etc.

### Future Development Areas
- Support for additional embedding models (local Ollama integration planned)
- Enhanced chunking strategies (Context 7-inspired approach planned)
- Performance optimizations for faster indexing
- Integration with Archon knowledge engine
- Comprehensive test suite with mocked dependencies