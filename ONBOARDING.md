# Developer Onboarding Guide

## Welcome to Crawl4AI RAG MCP Server

This comprehensive guide will get you up and running as a contributor to the Crawl4AI RAG MCP Server project. This is a production-ready Model Context Protocol (MCP) server that provides intelligent web crawling and RAG capabilities for AI agents and coding assistants.

---

## 1. Project Overview

### Purpose and Main Functionality
The Crawl4AI RAG MCP Server is a **production-ready MCP server** that bridges web crawling capabilities with semantic search through the Model Context Protocol. It enables AI agents and coding assistants to:

- **Crawl websites intelligently** with sitemap detection, recursive link following, and parallel processing
- **Store content semantically** in PostgreSQL with pgvector for vector similarity search
- **Perform RAG queries** with advanced strategies like hybrid search, contextual embeddings, and reranking
- **Extract code examples** specifically for AI coding assistants
- **Integrate seamlessly** with Claude Desktop, Windsurf, and other MCP-compatible clients

### Tech Stack
- **Backend**: Python 3.12+ with asyncio/await patterns throughout
- **Web Crawling**: Crawl4AI 0.6.2 for JavaScript-enabled crawling
- **Database**: PostgreSQL 17 + pgvector extension for vector similarity search
- **MCP Framework**: Official MCP Python SDK (FastMCP) for standardized tool integration
- **AI Integration**: OpenAI API for embeddings and LLM operations
- **Agent Framework**: Pydantic AI with Logfire observability for testing
- **Testing**: Pytest with asyncio support and centralized mocking (28 tests, <1s execution)
- **Package Management**: uv for fast dependency management
- **Development**: Rich CLI interfaces, automated startup scripts

### Architecture Pattern
**Modular MCP Server Architecture** - refactored from monolithic design:
- **MCP Server Core** (`mcp_server.py`) - FastMCP setup and lifecycle management
- **Tool Modules** - Separate modules for crawling tools and search tools
- **Strategy Modules** - Pluggable crawling strategies and content processing
- **Database Layer** - Async PostgreSQL operations with connection pooling
- **Agent Layer** - Testing framework with unified orchestrator agent

### Key Value Proposition
Instead of manual copy-paste documentation workflows, AI agents can **crawl once, search everywhere** - automatically discovering and semantically indexing web content for intelligent retrieval during coding tasks.

---

## 2. Repository Structure

```
mcp-crawl4ai-rag/
├── src/                               # Source code (modular architecture)
│   ├── crawl4ai_mcp.py               # Main entry point (backward compatibility)
│   ├── mcp_server.py                 # 🏗️ Core MCP server setup & lifecycle
│   ├── mcp_crawl_tools.py            # 🔧 Crawling MCP tools (2 tools)
│   ├── mcp_search_tools.py           # 🔍 Search MCP tools (3 tools)
│   ├── crawl_strategies.py           # 📝 URL detection & crawling strategies
│   ├── content_processing.py         # ⚙️ Smart chunking & content processing
│   ├── utils.py                      # 🛠️ Database utilities and helpers
│   ├── logging_config.py             # 📊 Logfire observability setup
│   ├── database/                     # Database operations module
│   │   ├── __init__.py
│   │   └── operations.py             # Async PostgreSQL operations
│   ├── embeddings/                   # Embedding generation module
│   │   ├── __init__.py
│   │   └── generator.py              # OpenAI embedding generation
│   ├── search/                       # Search engine module
│   │   ├── __init__.py
│   │   └── engine.py                 # Vector similarity & hybrid search
│   ├── content/                      # Content processing module
│   │   ├── __init__.py
│   │   └── processor.py              # Chunking and metadata extraction
│   └── pydantic_agent/               # Testing agent framework
│       ├── unified_agent.py          # 🤖 Single orchestrator agent
│       ├── agent.py                  # MCP client utilities
│       ├── dependencies.py           # Type-safe dependency injection
│       ├── outputs.py                # Structured response models
│       ├── tools.py                  # Agent tool implementations
│       └── examples/                 # Usage examples and workflows
├── tests/                            # Testing infrastructure (97% coverage)
│   ├── conftest.py                   # 🧪 Centralized mocking & fixtures
│   ├── unit/                         # Fast unit tests (<1s execution)
│   ├── integration/                  # Integration test placeholders
│   ├── fixtures/                     # Mock data and fixtures
│   └── test_*.py                     # Test modules by component
├── .claude/                          # Claude Code commands (untracked)
├── crawled_pages.sql                 # 🗄️ PostgreSQL schema with pgvector
├── start_mcp_server.sh              # ⚡ Optimized startup script (90% faster)
├── cli_chat.py                       # Interactive CLI interface
├── db_browser.py                     # Rich CLI database browser
├── pyproject.toml                    # Project dependencies & metadata
├── pytest.ini                       # Pytest configuration
├── .env                             # Environment configuration
└── CLAUDE.md                        # Project context for Claude Code
```

### File Organization Patterns
- **<500 lines per file** - Enforced modular architecture
- **Async/await throughout** - All operations are async for performance
- **Clear separation of concerns** - Tools, strategies, processing, database ops
- **Backward compatibility** - Main entry point imports all modules
- **Testing mirrors structure** - Test files match source file organization

---

## 3. Getting Started

### Prerequisites
Before you begin, ensure you have:

1. **Python 3.12+** ([download](https://www.python.org/downloads/))
2. **uv package manager** ([install guide](https://docs.astral.sh/uv/))
3. **PostgreSQL 17** ([download](https://www.postgresql.org/download/))
4. **pgvector extension** ([install guide](https://github.com/pgvector/pgvector))
5. **OpenAI API key** ([get key](https://platform.openai.com/api-keys))

### Environment Setup Commands

```bash
# 1. Clone and navigate to repository
git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
cd mcp-crawl4ai-rag

# 2. Install uv package manager (if not installed)
pip install uv

# 3. Create and activate virtual environment
uv venv crawl_venv
source crawl_venv/bin/activate  # On Windows: crawl_venv\Scripts\activate

# 4. Install dependencies and setup Crawl4AI
uv pip install -e .
crawl4ai-setup
```

### Database Setup

```bash
# macOS - Install PostgreSQL 17 + pgvector
brew install postgresql@17 pgvector
brew services start postgresql@17

# Ubuntu/Debian - Install PostgreSQL 17 + pgvector  
sudo apt-get install postgresql-17 postgresql-17-pgvector

# Add PostgreSQL to PATH (macOS)
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"
# Add this line to ~/.zshrc or ~/.bashrc for persistence

# Create database and run schema
createdb crawl4ai_rag
psql -h localhost -U $(whoami) -d crawl4ai_rag -f crawled_pages.sql

# Verify installation
psql -h localhost -U $(whoami) -d crawl4ai_rag -c "SELECT 1;"
psql -h localhost -U $(whoami) -d crawl4ai_rag -c "SELECT vector_dims('[1,2,3]'::vector);"
```

### Configuration Setup

Create a `.env` file in the project root:

```bash
# MCP Server Configuration
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse

# OpenAI API Configuration  
OPENAI_API_KEY=your_openai_api_key

# Database Configuration
DATABASE_URL=postgresql://$(whoami):@localhost:5432/crawl4ai_rag

# RAG Strategy Options (all default to false)
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false  
USE_AGENTIC_RAG=false
USE_RERANKING=false

# Optional: Logfire Observability
LOGFIRE_TOKEN=your_logfire_token

# LLM Choice for summaries
MODEL_CHOICE=gpt-4
```

### Running the Project

```bash
# Quick startup (recommended - 90% faster)
./start_mcp_server.sh

# Manual startup
source crawl_venv/bin/activate
python src/crawl4ai_mcp.py

# Using uv
uv run src/crawl4ai_mcp.py
```

### Running Tests

```bash
# Install test dependencies
uv pip install -e ".[dev]"

# Run full test suite with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Quick unit tests (recommended for development)
pytest tests/unit/ -v

# Test specific component
pytest tests/unit/test_content_processor.py -v
```

### Building for Production

```bash
# Code quality checks
ruff format src/ tests/ && ruff check src/ tests/
mypy src/

# Run all tests
pytest tests/ -v --cov=src --cov-report=html

# Start production server
TRANSPORT=sse python src/crawl4ai_mcp.py
```

---

## 4. Key Components

### Entry Points

#### Primary Entry Point
**`src/crawl4ai_mcp.py`** - Main MCP server entry point
- **Purpose**: Backward compatibility and single import point
- **Imports**: All modular components for unified access
- **Usage**: `python src/crawl4ai_mcp.py` starts the MCP server
- **Architecture note**: Refactored from 1,121-line monolith to modular imports

#### Startup Script  
**`start_mcp_server.sh`** - Optimized startup script
- **Performance**: 90% faster startup using existing venv
- **Environment**: Automatically loads `.env` and activates virtual environment
- **Transport**: Supports both SSE and stdio transports
- **Validation**: Checks for required virtual environment

### Core Business Logic

#### MCP Server Core
**`src/mcp_server.py`** - FastMCP server setup and lifecycle management
- **FastMCP initialization** with SSE/stdio transport selection
- **Async lifespan management** for database pools and crawler instances
- **Context management** for shared resources across MCP tools
- **Environment loading** with override support

#### MCP Tools (5 Total)
**`src/mcp_crawl_tools.py`** - Crawling tools (2 tools)
- `crawl_single_page(url)` - Quick single page crawl and storage
- `smart_crawl_url(url, max_depth, max_concurrent, chunk_size)` - Intelligent multi-page crawling

**`src/mcp_search_tools.py`** - Search tools (3 tools)  
- `get_available_sources()` - List crawled sources for discovery
- `perform_rag_query(query, source, match_count)` - Semantic search with RAG
- `search_code_examples(query, source_id, match_count)` - Code-specific search (conditional)

#### Strategy Implementations
**`src/crawl_strategies.py`** - URL detection and parallel processing
- **URL type detection**: Sitemap XML, text files, regular web pages
- **Parallel crawling**: Memory-adaptive batch processing
- **Link following**: Recursive internal link discovery
- **Content extraction**: HTML to markdown conversion

**`src/content_processing.py`** - Smart chunking and metadata extraction
- **Intelligent chunking**: Respects code blocks, headers, paragraphs
- **Contextual embeddings**: LLM-enhanced chunk context (optional)
- **Code extraction**: Identifies and summarizes code examples ≥300 chars
- **Reranking**: Cross-encoder result optimization

### Database Models/Schemas

#### PostgreSQL Schema
**`crawled_pages.sql`** - Database schema with pgvector extension
- **`sources` table**: Crawled source metadata with AI-generated summaries
- **`crawled_pages` table**: Content chunks with 1536-dim vector embeddings
- **`code_examples` table**: Extracted code with AI summaries (conditional)
- **Vector indexes**: Optimized for cosine similarity search
- **Unique constraints**: Prevent duplicate chunks

#### Database Operations
**`src/database/operations.py`** - Async PostgreSQL operations
- **Connection pooling**: asyncpg.Pool for efficient connections
- **Vector operations**: Embedding storage and similarity search
- **Batch operations**: Efficient bulk insert and update operations
- **Error handling**: Graceful PostgreSQL error management

### API Endpoints/Routes

#### MCP Protocol Endpoints
The server exposes MCP tools through standardized endpoints:

**SSE Transport** (recommended):
- `http://localhost:8051/sse` - Server-Sent Events endpoint for MCP clients

**Stdio Transport**:
- Direct process communication for MCP tools via stdin/stdout

#### Tool Registration
Tools are automatically registered via `@mcp.tool()` decorators:
```python
from mcp_server import mcp

@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> dict:
    """Crawls a single web page and stores content."""
    # Implementation with shared context access
```

### Configuration Management

#### Environment Configuration
**`.env` file** - Central configuration management
- **Server settings**: Host, port, transport type
- **API keys**: OpenAI API key for embeddings
- **Database**: PostgreSQL connection string
- **RAG strategies**: Feature flags for advanced RAG capabilities
- **Observability**: Logfire token for monitoring

#### Logging Configuration  
**`src/logging_config.py`** - Logfire observability setup
- **Agent tracing**: Automatic spans for all agent operations
- **API monitoring**: OpenAI request/response logging
- **Performance metrics**: Tool execution times and success rates
- **Error tracking**: Exception details with stack traces

### Authentication/Authorization

#### API Key Management
- **OpenAI API**: Required for embedding generation and LLM operations
- **Environment-based**: Stored in `.env` file, never committed
- **Validation**: Automatic validation during server startup

#### Database Security
- **Local PostgreSQL**: No external authentication required for development
- **Connection strings**: Supports password-protected connections
- **SSL support**: Can be configured for production deployments

---

## 5. Development Workflow

### Git Branch Conventions

#### Branch Naming
- **Feature branches**: `feat/description` or `feat/TASK-XXX-description`
- **Bug fixes**: `fix/description` or `fix/TASK-XXX-description`  
- **Refactoring**: `refactor/description` or `refactor/TASK-XXX-description`
- **Documentation**: `docs/description`
- **Testing**: `test/description`

#### Commit Message Format
Following Conventional Commits with task tracking:
```
type(scope): description - TASK-ID

Examples:
feat(mcp): add new search tool for code examples - TASK-046
fix(tests): correct assertion values in content processor - TASK-045  
refactor(agents): remove legacy agent functions - TASK-048
docs: update CLAUDE.md to reflect completed testing - TASK-045
test: implement centralized mocking infrastructure - TASK-045
```

### Creating a New Feature

#### 1. Planning Phase
```bash
# Create feature branch
git checkout -b feat/TASK-XXX-new-feature

# Plan the implementation in CLAUDE.md if complex
# Update todo list for multi-step features
```

#### 2. Implementation Phase
```bash
# Follow TDD cycle when appropriate:
# 1. Write failing tests first (red phase)
# 2. Implement to pass tests (green phase) 
# 3. Refactor for quality (refactor phase)

# For MCP tools - add new tool function:
# - Create function in appropriate tool module
# - Add @mcp.tool() decorator
# - Write comprehensive tests with mocking
# - Update crawl4ai_mcp.py imports for backward compatibility
```

#### 3. Quality Assurance
```bash
# Run code quality checks
ruff format src/ tests/ && ruff check src/ tests/
mypy src/

# Run comprehensive tests
pytest tests/ -v --cov=src --cov-report=term-missing

# Test MCP tools with inspector
npx @modelcontextprotocol/inspector
# Connect to: http://localhost:8051/sse
```

#### 4. Documentation
```bash
# Update relevant documentation:
# - Function docstrings with proper type hints
# - CLAUDE.md for architectural changes
# - README.md for user-facing changes (if needed)
```

### Testing Requirements

#### Test Structure
- **Unit tests**: `tests/unit/` - Fast, isolated, mocked external dependencies
- **Integration tests**: `tests/integration/` - End-to-end workflows
- **Test naming**: `test_*.py` matching source file structure

#### Testing Standards
- **3+ test cases minimum**: Happy path, edge case, error scenario
- **Centralized mocking**: Use `tests/conftest.py` auto-applied fixtures
- **Sub-second execution**: All unit tests must run in <1 second
- **Coverage target**: >90% on critical modules (achieved on content processor)

#### Mock Strategy
```python
# tests/conftest.py provides auto-mocking for:
# - OpenAI API calls (chat + embeddings)
# - PostgreSQL operations
# - Environment variables
# - External HTTP requests

# Tests run with zero external dependencies
```

### Code Style/Linting Rules

#### Formatting & Linting
```bash
# Auto-format code
ruff format src/ tests/

# Check for linting issues
ruff check src/ tests/

# Type checking
mypy src/
```

#### Code Standards
- **File length**: <500 lines per file (enforced)
- **Function length**: <50 lines, single responsibility
- **Type hints**: Complete type annotations required
- **Docstrings**: Every function, method, class documented
- **Async/await**: All I/O operations must be async

### PR Process and Review Guidelines

#### Pull Request Creation
```bash
# Before creating PR:
git add .
git commit -m "feat(scope): description - TASK-XXX"

# Push branch and create PR
git push origin feat/TASK-XXX-new-feature

# PR template should include:
# - Description of changes
# - Testing performed  
# - Documentation updates
# - Breaking changes (if any)
```

#### Review Checklist
- [ ] All tests pass (automated CI)
- [ ] Code coverage maintained >90% on modified files
- [ ] Type checking passes
- [ ] Documentation updated
- [ ] MCP tools tested with inspector
- [ ] No secrets or API keys committed
- [ ] Follows architectural patterns

### CI/CD Pipeline Overview

#### Automated Checks
- **Testing**: Full test suite with coverage reporting
- **Type checking**: mypy validation
- **Code quality**: ruff formatting and linting
- **Security**: No hardcoded secrets detection

#### Manual Testing
- **MCP Inspector**: Verify tool functionality at `http://localhost:8051/sse`
- **Database operations**: Verify PostgreSQL connectivity and vector operations
- **Agent workflows**: Test end-to-end workflows with included agent

---

## 6. Architecture Decisions

### Design Patterns and Reasoning

#### Modular MCP Server Architecture
**Decision**: Refactor from monolithic 1,121-line file to focused modules
**Reasoning**: 
- Maintainability: <500 lines per file makes code reviewable
- Separation of concerns: Clear boundaries between crawling, search, processing
- Testing: Easier to mock and test isolated components
- Extensibility: New tools and strategies can be added without touching core logic

#### FastMCP with Async/Await Throughout
**Decision**: Use official MCP Python SDK with async patterns
**Reasoning**:
- Performance: Non-blocking I/O for database and API operations
- Scalability: Handle multiple concurrent crawling operations
- MCP compliance: Official SDK ensures protocol compatibility
- Resource efficiency: Optimal use of database connection pools

#### PostgreSQL + pgvector for Vector Storage
**Decision**: PostgreSQL 17 with pgvector extension vs. dedicated vector DB
**Reasoning**:
- Unified storage: Single database for metadata and vectors
- ACID transactions: Consistent data integrity
- Rich querying: SQL + vector similarity in single queries
- Production readiness: Enterprise-grade reliability and backup/recovery

### State Management Approach

#### Shared Context via MCP Lifespan
**Pattern**: Resource sharing through FastMCP lifespan context
```python
@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    # Initialize shared resources
    postgres_pool = await create_postgres_pool()
    crawler = AsyncWebCrawler(...)
    
    yield Crawl4AIContext(
        crawler=crawler,
        postgres_pool=postgres_pool,
        reranking_model=reranking_model
    )
```

**Benefits**:
- Resource efficiency: Single database pool shared across all tools
- Lifecycle management: Automatic cleanup on server shutdown
- Type safety: Structured context with dataclass

#### Stateless Tool Design
**Pattern**: Each MCP tool is stateless, accessing shared resources via context
**Benefits**:
- Testability: Easy to mock context dependencies
- Scalability: No tool-specific state to manage
- Reliability: No state corruption between tool calls

### Error Handling Strategy

#### Graceful Degradation
```python
try:
    # Attempt advanced RAG features
    if os.getenv('USE_HYBRID_SEARCH') == 'true':
        results = await hybrid_search(...)
    else:
        results = await vector_search(...)
except Exception as e:
    logger.error(f"Search failed: {e}")
    return {"error": "Search temporarily unavailable", "fallback": True}
```

#### Structured Error Responses
All MCP tools return structured JSON with error details:
```python
return {
    "success": False,
    "error": "Database connection failed",
    "error_type": "DatabaseError",
    "retry_suggested": True
}
```

### Logging and Monitoring Setup

#### Pydantic AI + Logfire Integration
**Decision**: Built-in observability vs. custom logging
**Features**:
- Automatic agent tracing with nested spans
- Raw OpenAI API request/response logging
- Tool execution metrics (timing, success rates)
- Real-time dashboard for debugging

#### Structured Logging
```python
logger.info("Crawling completed", extra={
    "pages_crawled": len(results),
    "chunks_stored": total_chunks,
    "duration_seconds": end_time - start_time,
    "source_id": source_id
})
```

### Security Measures

#### Environment-Based Secrets
- All API keys stored in `.env` file
- No secrets committed to repository
- Automatic validation of required environment variables

#### Input Validation
```python
@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> dict:
    """Crawls a single web page."""
    # URL validation
    if not url.startswith(('http://', 'https://')):
        return {"error": "Invalid URL protocol"}
    
    # Implementation...
```

#### Database Security
- Parameterized queries prevent SQL injection
- Connection pooling with automatic cleanup
- Optional SSL connections for production

### Performance Optimizations

#### Parallel Crawling with Memory Management
```python
async def crawl_batch(urls: List[str], max_concurrent: int = 10):
    """Memory-adaptive parallel crawling."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def crawl_with_limit(url):
        async with semaphore:
            return await crawl_single_url(url)
    
    results = await asyncio.gather(*[crawl_with_limit(url) for url in urls])
    return results
```

#### Database Connection Pooling
- asyncpg.Pool for efficient PostgreSQL connections
- Connection reuse across MCP tool calls
- Automatic connection cleanup and recovery

#### Vector Search Optimization
- pgvector IVFFlat indexes for fast similarity search
- Batch embedding generation to reduce API calls
- Result reranking for improved relevance

---

## 7. Common Tasks

### Adding a New API Endpoint (MCP Tool)

#### 1. Choose the Appropriate Module
- **Crawling tools**: Add to `src/mcp_crawl_tools.py`
- **Search tools**: Add to `src/mcp_search_tools.py`
- **New category**: Create new module following naming convention

#### 2. Implement the Tool Function
```python
# src/mcp_crawl_tools.py
from mcp_server import mcp
from mcp.server.fastmcp import Context

@mcp.tool()
async def your_new_tool(ctx: Context, param1: str, param2: int = 10) -> dict:
    """
    Brief description of what this tool does.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter (default: 10)
        
    Returns:
        Dictionary with tool results and metadata
    """
    # Access shared resources
    crawler = ctx.request_context.lifespan_context["crawler"]
    postgres_pool = ctx.request_context.lifespan_context["postgres_pool"]
    
    try:
        # Implement tool logic
        result = await your_tool_logic(param1, param2, crawler, postgres_pool)
        
        return {
            "success": True,
            "result": result,
            "metadata": {
                "param1": param1,
                "param2": param2,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Tool failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
```

#### 3. Add Tests
```python
# tests/unit/test_mcp_tools.py
@pytest.mark.asyncio
async def test_your_new_tool_success(mock_context):
    """Test successful execution of your new tool."""
    result = await your_new_tool(mock_context, "test_param", 5)
    
    assert result["success"] is True
    assert "result" in result
    assert result["metadata"]["param1"] == "test_param"

@pytest.mark.asyncio  
async def test_your_new_tool_error_handling(mock_context):
    """Test error handling in your new tool."""
    # Mock an error condition
    with patch('your_module.your_tool_logic', side_effect=ValueError("Test error")):
        result = await your_new_tool(mock_context, "test_param", 5)
        
        assert result["success"] is False
        assert "Test error" in result["error"]
```

#### 4. Update Backward Compatibility
```python
# src/crawl4ai_mcp.py - Add import
from mcp_crawl_tools import (
    crawl_single_page,
    smart_crawl_url,
    your_new_tool  # Add this line
)

# Update __all__ list
__all__ = [
    # ... existing exports
    "your_new_tool"
]
```

### Creating a New Database Model

#### 1. Update SQL Schema
```sql
-- crawled_pages.sql
CREATE TABLE your_new_table (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    description TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Add indexes
CREATE INDEX ON your_new_table USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_your_new_table_metadata ON your_new_table USING gin (metadata);
```

#### 2. Add Database Operations
```python
# src/database/operations.py
async def insert_your_model(pool: asyncpg.Pool, name: str, description: str, 
                           embedding: List[float], metadata: dict) -> int:
    """Insert new record and return ID."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO your_new_table (name, description, embedding, metadata)
            VALUES ($1, $2, $3, $4)
            RETURNING id
        """, name, description, embedding, json.dumps(metadata))
        return row['id']

async def search_your_model(pool: asyncpg.Pool, query_embedding: List[float], 
                           limit: int = 10) -> List[dict]:
    """Search records by vector similarity."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, name, description, metadata,
                   embedding <=> $1::vector AS similarity
            FROM your_new_table
            ORDER BY similarity
            LIMIT $2
        """, query_embedding, limit)
        
        return [dict(row) for row in rows]
```

#### 3. Add Model Tests
```python
# tests/unit/test_database_operations.py
@pytest.mark.asyncio
async def test_insert_your_model(mock_db_pool):
    """Test inserting new model record."""
    model_id = await insert_your_model(
        mock_db_pool, 
        "test_name", 
        "test_description",
        [0.1] * 1536,
        {"key": "value"}
    )
    assert isinstance(model_id, int)
```

### Adding a New Test

#### 1. Choose Test Category
- **Unit tests**: `tests/unit/` for isolated component testing
- **Integration tests**: `tests/integration/` for end-to-end workflows
- **MCP tests**: `tests/test_mcp_tools.py` for tool functionality

#### 2. Write Test with Proper Mocking
```python
# tests/unit/test_your_component.py
import pytest
from unittest.mock import patch, MagicMock
from your_module import your_function

@pytest.mark.asyncio
async def test_your_function_success():
    """Test successful execution of your function."""
    # Arrange - use centralized mocks from conftest.py
    input_data = {"key": "value"}
    expected_output = {"result": "success"}
    
    # Act
    result = await your_function(input_data)
    
    # Assert
    assert result == expected_output
    assert "success" in result

@pytest.mark.asyncio
async def test_your_function_edge_case():
    """Test edge case handling."""
    # Test with empty input
    result = await your_function({})
    assert result is not None

@pytest.mark.asyncio  
async def test_your_function_error_handling():
    """Test error scenario."""
    with patch('your_module.external_dependency', side_effect=Exception("Test error")):
        with pytest.raises(Exception):
            await your_function({"key": "value"})
```

#### 3. Run Tests
```bash
# Run specific test file
pytest tests/unit/test_your_component.py -v

# Run with coverage
pytest tests/unit/test_your_component.py -v --cov=src/your_module

# Run all tests
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Debugging Common Issues

#### Database Connection Issues
```bash
# 1. Check PostgreSQL status
brew services list | grep postgresql
# or
sudo systemctl status postgresql

# 2. Verify database exists
psql -l | grep crawl4ai_rag

# 3. Test connection
psql -h localhost -U $(whoami) -d crawl4ai_rag -c "SELECT 1;"

# 4. Check pgvector extension
psql -h localhost -U $(whoami) -d crawl4ai_rag -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

#### MCP Tool Testing Issues  
```bash
# 1. Start MCP server
./start_mcp_server.sh

# 2. Test with MCP Inspector
npx @modelcontextprotocol/inspector
# Connect to: http://localhost:8051/sse

# 3. Check server logs
tail -f server.log

# 4. Test individual tools manually
python -c "
import asyncio
from src.mcp_crawl_tools import crawl_single_page
from tests.conftest import mock_context
asyncio.run(crawl_single_page(mock_context(), 'https://example.com'))
"
```

#### Test Failures
```bash
# 1. Run tests with verbose output
pytest tests/unit/test_failing_module.py -v -s

# 2. Check if mocks are working
pytest tests/unit/test_failing_module.py -v --capture=no

# 3. Verify test isolation
pytest tests/unit/test_failing_module.py::test_specific_function -v

# 4. Check coverage gaps
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Updating Dependencies

#### Using uv Package Manager
```bash
# Update all dependencies
uv pip install --upgrade -e .

# Update specific package
uv pip install --upgrade crawl4ai

# Add new dependency
# 1. Edit pyproject.toml
dependencies = [
    "crawl4ai==0.6.2",
    "new-package>=1.0.0",  # Add here
]

# 2. Install updated dependencies
uv pip install -e .

# 3. Update lock file
uv pip freeze > requirements.lock
```

#### Testing After Updates
```bash
# 1. Run full test suite
pytest tests/ -v --cov=src

# 2. Test MCP server startup
./start_mcp_server.sh

# 3. Verify MCP tools work
npx @modelcontextprotocol/inspector

# 4. Test agent workflows
python cli_chat.py
```

---

## 8. Potential Gotchas

### Non-obvious Configurations

#### Environment Variable Loading Order
**Issue**: Environment variables may not load as expected
**Solution**: The project loads `.env` with `override=True` to ensure consistency
```python
# src/mcp_server.py
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path, override=True)  # Force override
```

#### PostgreSQL Path Issues (macOS)
**Issue**: PostgreSQL 17 not in PATH after brew install
**Solution**: Add to shell profile
```bash
# Add to ~/.zshrc or ~/.bashrc
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"

# Reload shell
source ~/.zshrc
```

#### Virtual Environment Detection
**Issue**: `start_mcp_server.sh` fails if virtual environment missing
**Solution**: Script validates environment before starting
```bash
# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found"
    echo "Please run: uv venv crawl_venv && uv pip install -e ."
    exit 1
fi
```

### Required Environment Variables

#### Critical Variables
These environment variables are **required** for basic functionality:
- `OPENAI_API_KEY` - Required for embeddings and LLM operations
- `DATABASE_URL` - PostgreSQL connection string

#### Optional but Important
- `USE_AGENTIC_RAG=true` - Enables `search_code_examples` tool
- `LOGFIRE_TOKEN` - Enables observability dashboard
- `TRANSPORT=sse` - SSE transport (recommended) vs stdio

#### Validation Pattern
```python
# All critical environment variables are validated at startup
required_vars = ['OPENAI_API_KEY']
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    raise ValueError(f"Missing required environment variables: {missing}")
```

### External Service Dependencies

#### OpenAI API Rate Limits
**Issue**: Rate limiting during large crawling operations
**Mitigation**: 
- Batch embedding generation when possible
- Implement exponential backoff (already in place)
- Monitor usage via Logfire dashboard

#### PostgreSQL Connection Limits
**Issue**: Connection pool exhaustion during concurrent operations
**Solution**: Connection pooling configured with limits
```python
postgres_pool = await asyncpg.create_pool(
    dsn=database_url,
    min_size=5,
    max_size=20,
    command_timeout=60
)
```

#### Crawl4AI JavaScript Rendering
**Issue**: Some websites require JavaScript to render content
**Solution**: Crawl4AI uses browser automation by default
```python
# Browser configuration for JavaScript-heavy sites
config = CrawlerRunConfig(
    cache_mode=CacheMode.BYPASS,
    js_code="window.scrollTo(0, document.body.scrollHeight);",  # Trigger lazy loading
    wait_for="css:main"  # Wait for specific elements
)
```

### Known Issues and Workarounds

#### Test Execution Speed
**Issue**: Tests were previously slow (>2 minutes)
**✅ Fixed**: Centralized mocking reduces execution to <1 second
**Implementation**: `tests/conftest.py` auto-applies mocks globally

#### Server Startup Time
**Issue**: Server startup was slow due to dependency installation
**✅ Fixed**: `start_mcp_server.sh` uses existing virtual environment (90% faster)

#### Vector Embedding Format Compatibility
**Issue**: PostgreSQL pgvector format conversion errors
**✅ Fixed**: Proper vector format conversion in database operations
```python
# Correct vector format for pgvector
embedding_vector = f"[{','.join(map(str, embedding))}]"
```

#### MCP Tool Function Naming Conflicts
**Issue**: Function names conflicted with module imports
**✅ Fixed**: Clear separation between tool functions and helper functions

### Performance Bottlenecks

#### Large Document Processing
**Issue**: Memory usage spikes with large documents
**Mitigation**: Chunking strategy respects memory limits
```python
def smart_chunk_markdown(content: str, chunk_size: int = 5000):
    """Memory-efficient chunking that respects boundaries."""
    # Chunking logic that prevents memory spikes
```

#### Concurrent Crawling Limits
**Issue**: Too many concurrent requests can overwhelm target servers
**Solution**: Configurable concurrency with sensible defaults
```python
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, 
                         max_concurrent: int = 10, chunk_size: int = 5000):
    # Semaphore controls concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
```

#### Vector Search Performance
**Issue**: Slow similarity search on large datasets
**Solution**: Optimized pgvector indexes
```sql
-- Optimized for cosine similarity
CREATE INDEX ON crawled_pages USING ivfflat (embedding vector_cosine_ops);
```

### Areas of Technical Debt

#### Backward Compatibility Layer
**Current**: `crawl4ai_mcp.py` imports all modules for compatibility
**Future**: May be removed in favor of direct module imports
**Impact**: Existing integrations would need import updates

#### Environment Configuration
**Current**: Mix of `.env` file and environment variables
**Future**: Could be unified into single configuration system
**Workaround**: Use `.env` file for consistency

#### Test Coverage Gaps
**Current**: 97% coverage on content processor, lower on some modules
**Goal**: >90% coverage across all critical modules
**Action**: Expand test coverage for database and search modules

---

## 9. Documentation and Resources

### Existing Documentation

#### In-Repository Documentation
- **`README.md`** - Comprehensive user guide with installation, configuration, and usage
- **`CLAUDE.md`** - Development context and commands for Claude Code integration
- **`ONBOARDING.md`** - This comprehensive developer guide
- **`QUICKSTART.md`** - Essential setup steps for rapid development start

#### Code Documentation
- **Function docstrings** - Every function has type hints and usage examples
- **Module docstrings** - Each module explains its purpose and key exports
- **Inline comments** - Complex logic includes `# Reason:` explanations

#### Database Documentation
- **`crawled_pages.sql`** - Complete schema with comments explaining each table
- **Vector operations** - Function definitions with parameter documentation
- **Index strategies** - Comments explaining performance optimizations

### API Documentation

#### MCP Tools Documentation
Each tool function includes comprehensive docstrings:
```python
@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> dict:
    """
    Crawls a single web page and stores its content in the database.
    
    This tool is ideal for quickly retrieving content from a specific URL
    without following links. It converts HTML to clean markdown, chunks 
    content intelligently, and stores with vector embeddings.
    
    Args:
        url (str): The URL to crawl. Must start with http:// or https://
        
    Returns:
        dict: Results containing:
            - success (bool): Whether crawling succeeded
            - chunks_stored (int): Number of content chunks stored
            - source_info (dict): Source metadata and summary
            - word_count (int): Total words processed
            - content_length (int): Total character count
    
    Example:
        >>> result = await crawl_single_page(ctx, "https://docs.python.org")
        >>> print(f"Stored {result['chunks_stored']} chunks")
    """
```

#### Database Schema Documentation
```sql
-- Sources table: Stores metadata about crawled domains/websites
CREATE TABLE sources (
    source_id TEXT PRIMARY KEY,        -- Domain name (e.g., 'docs.python.org')
    summary TEXT,                      -- AI-generated summary of the source
    total_word_count INTEGER,          -- Total words across all pages from this source
    created_at TIMESTAMP,              -- When first crawled
    updated_at TIMESTAMP               -- Last update time
);
```

### Testing Documentation

#### Test Architecture Guide
- **`tests/conftest.py`** - Documents centralized mocking strategy
- **Test categories** - Unit vs integration test guidelines  
- **Coverage targets** - Minimum coverage requirements per module
- **Mock patterns** - Examples of proper test mocking

#### Performance Testing
```python
# Example performance test pattern
@pytest.mark.asyncio
async def test_crawling_performance():
    """Test that crawling operations complete within time limits."""
    start_time = time.time()
    
    result = await crawl_single_page(mock_context, "https://example.com")
    
    duration = time.time() - start_time
    assert duration < 5.0  # Should complete within 5 seconds
    assert result["success"] is True
```

### Deployment Documentation

#### Production Deployment Guide
The README.md includes production deployment patterns:
- SSL configuration for PostgreSQL
- Environment variable security
- Monitoring and observability setup
- Performance tuning recommendations

#### MCP Client Integration
Detailed integration examples for major MCP clients:
- Claude Desktop configuration
- Windsurf setup instructions  
- Generic MCP client patterns
- Troubleshooting connection issues

### Team Conventions

#### Code Style Guide
- **File organization**: <500 lines per file, clear module boundaries
- **Naming conventions**: Snake_case for Python, descriptive function names
- **Type hints**: Required for all function parameters and return values
- **Documentation**: Docstring required for all public functions

#### Git Workflow
- **Branch naming**: `feat/`, `fix/`, `refactor/`, `docs/`, `test/` prefixes
- **Commit messages**: Conventional commits with task tracking
- **PR requirements**: Tests pass, documentation updated, type checking passes

#### Development Environment
- **Python version**: 3.12+ required for latest async features
- **Package manager**: uv for fast dependency management
- **Editor setup**: VS Code recommended with Python and Pylance extensions
- **Database tools**: pgAdmin or DBeaver for database inspection

---

## 10. Next Steps - Onboarding Checklist

### Environment Setup ✅
- [ ] **Install Python 3.12+** and verify version: `python --version`
- [ ] **Install uv package manager**: `pip install uv`
- [ ] **Clone repository**: `git clone <repo-url> && cd mcp-crawl4ai-rag`
- [ ] **Create virtual environment**: `uv venv crawl_venv`
- [ ] **Activate environment**: `source crawl_venv/bin/activate`
- [ ] **Install dependencies**: `uv pip install -e .`
- [ ] **Setup Crawl4AI**: `crawl4ai-setup`

### Database Setup ✅  
- [ ] **Install PostgreSQL 17**: `brew install postgresql@17 pgvector` (macOS)
- [ ] **Start PostgreSQL**: `brew services start postgresql@17`
- [ ] **Add to PATH**: `export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"`
- [ ] **Create database**: `createdb crawl4ai_rag`
- [ ] **Run schema**: `psql -h localhost -U $(whoami) -d crawl4ai_rag -f crawled_pages.sql`
- [ ] **Verify setup**: `psql -d crawl4ai_rag -c "SELECT 1;"`

### Configuration Setup ✅
- [ ] **Create .env file** with required variables (see template above)
- [ ] **Set OPENAI_API_KEY** in .env file
- [ ] **Configure DATABASE_URL** in .env file
- [ ] **Choose RAG strategies** (start with defaults: all false)
- [ ] **Optional: Add LOGFIRE_TOKEN** for observability

### First Run ✅
- [ ] **Start MCP server**: `./start_mcp_server.sh`
- [ ] **Verify server starts** without errors (should show SSE endpoint)
- [ ] **Test MCP Inspector**: `npx @modelcontextprotocol/inspector`
- [ ] **Connect to**: `http://localhost:8051/sse`
- [ ] **Test basic tool**: Use `crawl_single_page` with `https://example.com`

### Development Validation ✅
- [ ] **Install test dependencies**: `uv pip install -e ".[dev]"`
- [ ] **Run test suite**: `pytest tests/ -v`
- [ ] **Verify <1s execution**: Tests should run very quickly
- [ ] **Check code quality**: `ruff format src/ tests/ && ruff check src/ tests/`
- [ ] **Run type checking**: `mypy src/`

### Understand Main User Flow ✅
- [ ] **Study MCP tools**: Review 5 tools in `mcp_crawl_tools.py` and `mcp_search_tools.py`
- [ ] **Test crawling workflow**: Crawl a simple site, verify storage in database
- [ ] **Test search workflow**: Use `get_available_sources` then `perform_rag_query`
- [ ] **Review agent testing**: Run `python cli_chat.py` for interactive testing
- [ ] **Check database**: Use `python db_browser.py` to see stored data

### Make a Test Change ✅
- [ ] **Create feature branch**: `git checkout -b test/onboarding-validation`
- [ ] **Make small change**: Add a comment or improve a docstring
- [ ] **Run tests**: Ensure all tests still pass
- [ ] **Commit change**: `git commit -m "test: onboarding validation change"`
- [ ] **Understand git workflow**: Review commit message format and branch naming

### Identify Contributing Area ✅
Choose one area to start contributing:

#### MCP Server Development
- [ ] **Learn MCP patterns**: Study existing tools in `mcp_crawl_tools.py`
- [ ] **Review tool testing**: Examine `tests/test_mcp_tools.py`
- [ ] **Understand context sharing**: Review lifespan management in `mcp_server.py`

#### Database & RAG Development  
- [ ] **Study vector operations**: Review `src/database/operations.py`
- [ ] **Understand chunking strategy**: Examine `content_processing.py`
- [ ] **Learn search algorithms**: Review `src/search/engine.py`

#### Testing & Quality
- [ ] **Study mocking strategy**: Review `tests/conftest.py`
- [ ] **Understand coverage goals**: Check current coverage reports
- [ ] **Learn test patterns**: Study existing unit tests

#### Documentation & DevX
- [ ] **Review documentation gaps**: Check for missing docstrings
- [ ] **Study user workflow**: Understand pain points in setup process
- [ ] **Examine CLI tools**: Review `cli_chat.py` and `db_browser.py`

### Advanced Setup (Optional) ✅
- [ ] **Setup Logfire observability**: Add token and test dashboard
- [ ] **Configure advanced RAG**: Enable `USE_HYBRID_SEARCH=true` and test
- [ ] **Setup development tools**: Configure VS Code with Python extensions
- [ ] **Review monitoring**: Study Logfire dashboard and error tracking

---

## Welcome to the Team! 🚀

You now have a comprehensive understanding of the Crawl4AI RAG MCP Server project. This system represents a production-ready foundation for integrating web crawling and semantic search into AI agent workflows.

### Key Takeaways
- **Modular architecture** enables focused development and testing
- **Production-ready** with comprehensive testing and observability  
- **MCP-first design** ensures compatibility with major AI coding assistants
- **Performance-optimized** with sub-second tests and efficient crawling
- **Extensible** through clear patterns for adding tools and strategies

### Getting Help
- **Architecture questions**: Review `CLAUDE.md` and this onboarding guide
- **Setup issues**: Check troubleshooting section and known issues
- **Development patterns**: Study existing code examples and tests
- **MCP integration**: Use MCP Inspector for tool testing and debugging

The codebase follows clear patterns and conventions that make it approachable for developers at all levels. Focus on understanding the MCP tool patterns first, then dive deeper into the areas that align with your interests and expertise.

Happy coding! 🧠⚡