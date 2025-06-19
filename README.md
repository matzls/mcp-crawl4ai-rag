<h1 align="center">Crawl4AI RAG MCP Server</h1>

<p align="center">
  <em>Web Crawling and RAG Capabilities for AI Agents and AI Coding Assistants</em>
</p>

A production-ready [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that provides **5 powerful tools** for web crawling and RAG operations. Built with [Crawl4AI](https://crawl4ai.com) and PostgreSQL + pgvector, this server enables any MCP-compatible AI agent or coding assistant to **scrape anything** and then **use that knowledge anywhere** for RAG.

> **✅ Production Ready (January 2025)**: All 5 MCP tools verified working, PostgreSQL integration complete, comprehensive testing implemented. System tested with 197 pages crawled in 42.5 seconds with 856 content chunks stored successfully.

**Primary Use Case**: Connect this MCP server to Claude Desktop, Windsurf, or any MCP-compatible AI agent to provide intelligent web crawling and RAG capabilities. The goal is to integrate this into [Archon](https://github.com/coleam00/Archon) as a knowledge engine for AI coding assistants.

## Overview

This MCP server provides **5 core tools** that external AI agents and coding assistants can use:

**Core Value Proposition**: Instead of manually copying and pasting documentation, AI agents can crawl websites, store content in a vector database (PostgreSQL with pgvector), and perform semantic search over the crawled content - all through standardized MCP tool calls.

**Included for Testing**: A Pydantic AI agent is included for testing and demonstration purposes, but the **MCP server is the primary product** designed for integration with external MCP clients.

The server includes several advanced RAG strategies that can be enabled to enhance retrieval quality:
- **Contextual Embeddings** for enriched semantic understanding
- **Hybrid Search** combining vector and keyword search
- **Agentic RAG** for specialized code example extraction
- **Reranking** for improved result relevance using cross-encoder models

See the [Configuration section](#configuration) below for details on how to enable and configure these strategies.

## Vision

The Crawl4AI RAG MCP server is just the beginning. Here's where we're headed:

1. **Integration with Archon**: Building this system directly into [Archon](https://github.com/coleam00/Archon) to create a comprehensive knowledge engine for AI coding assistants to build better AI agents.

2. **Multiple Embedding Models**: Expanding beyond OpenAI to support a variety of embedding models, including the ability to run everything locally with Ollama for complete control and privacy.

3. **Advanced RAG Strategies**: Implementing sophisticated retrieval techniques like contextual retrieval, late chunking, and others to move beyond basic "naive lookups" and significantly enhance the power and precision of the RAG system, especially as it integrates with Archon.

4. **Enhanced Chunking Strategy**: Implementing a Context 7-inspired chunking approach that focuses on examples and creates distinct, semantically meaningful sections for each chunk, improving retrieval precision.

5. **Performance Optimization**: Increasing crawling and indexing speed to make it more realistic to "quickly" index new documentation to then leverage it within the same prompt in an AI coding assistant.

## Features

### MCP Server Capabilities
- **Smart URL Detection**: Automatically detects and handles different URL types (regular webpages, sitemaps, text files)
- **Recursive Crawling**: Follows internal links to discover content
- **Parallel Processing**: Efficiently crawls multiple pages simultaneously
- **Content Chunking**: Intelligently splits content by headers and size for better processing
- **Vector Search**: Performs RAG over crawled content, optionally filtering by data source for precision
- **Source Retrieval**: Retrieve sources available for filtering to guide the RAG process

### Pydantic AI Agent Capabilities
- **Intelligent Workflow Orchestration**: Combines multiple MCP tools for complex multi-step operations
- **Type-Safe Configuration**: Structured dependency injection with Pydantic models
- **Structured Outputs**: Validated responses with comprehensive metadata
- **Unified Agent Architecture**: Single intelligent orchestrator that automatically selects optimal tools based on user intent
- **MCP Client Integration**: Seamless connection to MCP server using documented patterns
- **Standard Logfire Observability**: Built-in tracing and monitoring using Pydantic AI's native Logfire integration

## Tools

The server provides 5 powerful tools for web crawling and intelligent content retrieval:

### Core Tools (Always Available)

#### 1. `crawl_single_page(url: str)`
Crawls a single web page and stores its content in the PostgreSQL database. This tool is ideal for quickly retrieving content from a specific URL without following links.

- **Use case**: Quick content acquisition from specific pages
- **Process**: Converts HTML to clean markdown, chunks content intelligently (respects code blocks and paragraphs)
- **Storage**: Stores chunks in PostgreSQL with vector embeddings for semantic search
- **Code extraction**: Extracts and processes code examples if `USE_AGENTIC_RAG=true`
- **Returns**: Chunk count, code examples stored, content length, word count, source info

#### 2. `smart_crawl_url(url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000)`
Intelligently detects URL type and applies the appropriate crawling strategy. This is the most powerful crawling tool that adapts to different content sources.

- **URL Detection**:
  - **Sitemaps** (*.xml): Extracts all URLs and crawls them in parallel
  - **Text files** (*.txt): Direct content retrieval (useful for llms.txt files)
  - **Web pages**: Recursive crawling of internal links up to specified depth
- **Parallel processing**: Configurable concurrency with memory-adaptive dispatcher
- **Smart chunking**: Respects code blocks, paragraphs, and sentence boundaries
- **Returns**: Crawl type, pages crawled, chunks stored, code examples, source updates

#### 3. `get_available_sources()`
Lists all crawled sources (domains) with summaries and metadata. Essential for discovering what content is available before performing targeted searches.

- **Use case**: Discovery and source filtering preparation
- **Information provided**: Source IDs, AI-generated summaries, word counts, timestamps
- **Best practice**: Always use this before calling RAG query tools with source filtering
- **Returns**: Complete source catalog with statistics

#### 4. `perform_rag_query(query: str, source: str = None, match_count: int = 5)`
Performs semantic search across stored content using vector similarity. The core RAG tool for finding relevant information.

- **Search capabilities**:
  - **Vector search**: Semantic similarity using OpenAI embeddings (text-embedding-3-small)
  - **Hybrid search**: Combines vector + keyword search if `USE_HYBRID_SEARCH=true`
  - **Source filtering**: Optional filtering by specific domain/source
  - **Reranking**: Cross-encoder reranking if `USE_RERANKING=true`
- **Advanced features**: Smart result combination, similarity scoring, metadata preservation
- **Returns**: Ranked content chunks with similarity scores, search mode info, reranking status

### Conditional Tools

#### 5. `search_code_examples(query: str, source_id: str = None, match_count: int = 5)`
**Available only when `USE_AGENTIC_RAG=true`**

Specialized search for code examples with AI-generated summaries. This tool provides targeted code snippet retrieval specifically designed for AI coding assistants.

- **Code extraction**: Identifies code blocks ≥300 characters with surrounding context
- **AI summaries**: Each code example gets an intelligent summary via LLM
- **Dual search**: Searches both code content and summaries for comprehensive coverage
- **Same advanced features**: Supports hybrid search and reranking like regular RAG
- **Use case**: Finding specific implementations, patterns, or usage examples
- **Returns**: Code blocks with summaries, source info, similarity scores

### Tool Workflow

The typical workflow combines multiple tools:

1. **Crawl content**: Use `crawl_single_page` or `smart_crawl_url` to index websites
2. **Discover sources**: Use `get_available_sources` to see what's available
3. **Search content**: Use `perform_rag_query` for general content or `search_code_examples` for code-specific queries
4. **Iterate**: Refine searches with source filtering based on discovered sources

## Integration with MCP Clients

This MCP server is designed to be used by external AI agents and coding assistants through the Model Context Protocol. Here's how to connect:

### SSE Configuration (Recommended)

For Claude Desktop, Windsurf, or any SSE-compatible MCP client:

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

> **Note for Windsurf users**: Use `serverUrl` instead of `url`:
> ```json
> {
>   "mcpServers": {
>     "crawl4ai-rag": {
>       "transport": "sse", 
>       "serverUrl": "http://localhost:8051/sse"
>     }
>   }
> }
> ```

### Stdio Configuration  

For applications that prefer stdio transport:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "python",
      "args": ["path/to/crawl4ai-mcp/src/crawl4ai_mcp.py"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "DATABASE_URL": "postgresql://mg@localhost:5432/crawl4ai_rag"
      }
    }
  }
}
```

### Using with uv

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "uv",
      "args": ["run", "src/crawl4ai_mcp.py"],
      "cwd": "/path/to/mcp-crawl4ai-rag",
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key", 
        "DATABASE_URL": "postgresql://mg@localhost:5432/crawl4ai_rag"
      }
    }
  }
}
```

### Available MCP Tools

Once connected, your AI agent will have access to these 5 tools:

1. **`crawl_single_page`** - Quick single page crawl
2. **`smart_crawl_url`** - Intelligent multi-page crawling  
3. **`get_available_sources`** - List crawled sources
4. **`perform_rag_query`** - Semantic search over content
5. **`search_code_examples`** - Code-specific search (if enabled)

## Testing with Included Agent

For testing and demonstration purposes, this project includes a Pydantic AI agent that connects to the MCP server as a client.

> **Note**: The agent is provided for testing only. The **MCP server is the primary product** designed for external integrations.

### Quick Testing Example

```python
from pydantic_agent.unified_agent import create_unified_agent

# Test the MCP server using the included agent
agent = create_unified_agent("http://localhost:8051/sse")

async with agent.run_mcp_servers():
    # Test crawling
    result = await agent.run("Crawl https://docs.python.org/3/tutorial/")
    
    # Test search  
    result = await agent.run("Find information about Python functions")
```

### Testing Results

✅ **MCP Server Performance Verified**:
- **197 pages crawled** in 42.5 seconds
- **856 content chunks** stored successfully in PostgreSQL
- All 5 MCP tools working correctly
- Both SSE and stdio transports operational

## Prerequisites

- [Python 3.12+](https://www.python.org/downloads/) with [uv package manager](https://docs.astral.sh/uv/)
- [PostgreSQL 17](https://www.postgresql.org/download/) with [pgvector extension](https://github.com/pgvector/pgvector) (local database for RAG)
- [OpenAI API key](https://platform.openai.com/api-keys) (for generating embeddings)

## Installation

### Using uv (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. Install uv if you don't have it:
   ```bash
   pip install uv
   ```

3. Create and activate a virtual environment:
   ```bash
   uv venv crawl_venv
   crawl_venv\Scripts\activate
   # on Mac/Linux: source crawl_venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   uv pip install -e .
   crawl4ai-setup
   ```

   This will install all required dependencies including:
   - `crawl4ai==0.6.2` - Web crawling capabilities
   - `mcp>=1.9.4` - Model Context Protocol framework
   - `pydantic-ai[logfire]>=0.2.18` - Intelligent agent framework with built-in Logfire observability
   - `asyncpg==0.30.0` - PostgreSQL async driver
   - `openai>=1.86.0` - OpenAI API integration
   - `sentence-transformers>=4.1.0` - Embedding models

5. Create a `.env` file based on the configuration section below

## Project Structure

```
mcp-crawl4ai-rag/
├── src/
│   ├── crawl4ai_mcp.py          # Main MCP server implementation (official MCP SDK)
│   ├── utils.py                 # Database and RAG utilities
│   ├── logging_config.py        # Logfire observability configuration
│   └── pydantic_agent/          # Unified agent layer
│       ├── __init__.py          # Agent exports
│       ├── unified_agent.py     # Single orchestrator agent (OpenAI o3)
│       ├── agent.py             # MCP connection utilities
│       ├── dependencies.py     # Type-safe dependency injection models
│       ├── outputs.py           # Structured output validation models
│       ├── tools.py             # Agent tool implementations
│       └── examples/            # Usage examples and workflows
│           ├── unified_agent_example.py     # Main example
│           ├── basic_crawl_example.py
│           └── rag_workflow_example.py
├── tests/                       # Testing infrastructure
│   ├── __init__.py             # Test package initialization
│   ├── test_mcp_tools.py       # Comprehensive MCP tool testing
│   └── test_logging.py         # Logging verification tests
├── crawled_pages.sql            # PostgreSQL schema
├── pyproject.toml               # Project dependencies + dev dependencies
├── pytest.ini                  # Pytest configuration
├── .env                         # Environment configuration
├── cli_chat.py                  # Interactive CLI interface
├── db_browser.py                # Custom database browser with rich CLI interface
└── start_mcp_server.sh          # Optimized startup script
```

## Database Setup

Before running the server, you need to set up PostgreSQL 17 with the pgvector extension:

### PostgreSQL 17 Setup

1. **Install PostgreSQL 17 and pgvector:**
   ```bash
   # macOS
   brew install postgresql@17 pgvector

   # Ubuntu/Debian
   sudo apt-get install postgresql-17 postgresql-17-pgvector
   ```

2. **Start PostgreSQL and configure PATH:**
   ```bash
   # macOS - Start service
   brew services start postgresql@17

   # Add to PATH (add to your shell profile: ~/.zshrc or ~/.bashrc)
   export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"
   ```

3. **Create the database and run schema:**
   ```bash
   # Create database
   createdb crawl4ai_rag

   # Run schema to create tables and functions
   psql -h localhost -U $(whoami) -d crawl4ai_rag -f crawled_pages.sql
   ```

### Verify Installation

Test your PostgreSQL setup:
```bash
# Test connection
psql -h localhost -U $(whoami) -d crawl4ai_rag -c "SELECT 1;"

# Test pgvector extension
psql -h localhost -U $(whoami) -d crawl4ai_rag -c "SELECT vector_dims('[1,2,3]'::vector);"
```

## Configuration

Create a `.env` file in the project root with the following variables:

```
# MCP Server Configuration
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key

# Logfire Observability (Optional)
LOGFIRE_TOKEN=your_logfire_token

# LLM for summaries and contextual embeddings  
MODEL_CHOICE=gpt-o3

# RAG Strategies (set to "true" or "false", default to "false")
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false
USE_RERANKING=false

# PostgreSQL Configuration (Option 1: Use DATABASE_URL)
DATABASE_URL=postgresql://mg:@localhost:5432/crawl4ai_rag

# PostgreSQL Configuration (Option 2: Individual components)
# POSTGRES_HOST=localhost
# POSTGRES_PORT=5432
# POSTGRES_DB=crawl4ai_rag
# POSTGRES_USER=mg
# POSTGRES_PASSWORD=
```

### Observability & Monitoring

The system includes comprehensive observability through **Pydantic AI's built-in Logfire integration**:

#### Features
- **Automatic Agent Tracing**: All agent runs are automatically traced with nested spans
- **HTTP Request Monitoring**: Raw OpenAI API requests/responses visible for debugging
- **MCP Tool Performance**: Execution times, success rates, and result summaries
- **Error Tracking**: Exception details with stack traces and context
- **Real-time Dashboard**: Live monitoring during agent execution

#### Setup
1. **Get Logfire Token**: Sign up at [logfire.pydantic.dev](https://logfire.pydantic.dev)
2. **Add to Environment**: Set `LOGFIRE_TOKEN=your_token` in your `.env` file
3. **Automatic Configuration**: Logfire is automatically configured when agents start
4. **View Dashboard**: https://logfire-eu.pydantic.dev/matzls/crawl4ai-agent

#### What You'll See
- Complete agent workflow traces with timing information
- Tool selection and execution details
- OpenAI API calls with token usage and response times
- Error context and recovery attempts
- Performance metrics and bottleneck identification

### RAG Strategy Options

The Crawl4AI RAG MCP server supports four powerful RAG strategies that can be enabled independently:

#### 1. **USE_CONTEXTUAL_EMBEDDINGS**
When enabled, this strategy enhances each chunk's embedding with additional context from the entire document. The system passes both the full document and the specific chunk to an LLM (configured via `MODEL_CHOICE`) to generate enriched context that gets embedded alongside the chunk content.

- **When to use**: Enable this when you need high-precision retrieval where context matters, such as technical documentation where terms might have different meanings in different sections.
- **Trade-offs**: Slower indexing due to LLM calls for each chunk, but significantly better retrieval accuracy.
- **Cost**: Additional LLM API calls during indexing.

#### 2. **USE_HYBRID_SEARCH**
Combines traditional keyword search with semantic vector search to provide more comprehensive results. The system performs both searches in parallel and intelligently merges results, prioritizing documents that appear in both result sets.

- **When to use**: Enable this when users might search using specific technical terms, function names, or when exact keyword matches are important alongside semantic understanding.
- **Trade-offs**: Slightly slower search queries but more robust results, especially for technical content.
- **Cost**: No additional API costs, just computational overhead.

#### 3. **USE_AGENTIC_RAG**
Enables specialized code example extraction and storage. When crawling documentation, the system identifies code blocks (≥300 characters), extracts them with surrounding context, generates summaries, and stores them in a separate vector database table specifically designed for code search.

- **When to use**: Essential for AI coding assistants that need to find specific code examples, implementation patterns, or usage examples from documentation.
- **Trade-offs**: Significantly slower crawling due to code extraction and summarization, requires more storage space.
- **Cost**: Additional LLM API calls for summarizing each code example.
- **Benefits**: Provides a dedicated `search_code_examples` tool that AI agents can use to find specific code implementations.

#### 4. **USE_RERANKING**
Applies cross-encoder reranking to search results after initial retrieval. Uses a lightweight cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to score each result against the original query, then reorders results by relevance.

- **When to use**: Enable this when search precision is critical and you need the most relevant results at the top. Particularly useful for complex queries where semantic similarity alone might not capture query intent.
- **Trade-offs**: Adds ~100-200ms to search queries depending on result count, but significantly improves result ordering.
- **Cost**: No additional API costs - uses a local model that runs on CPU.
- **Benefits**: Better result relevance, especially for complex queries. Works with both regular RAG search and code example search.

### Recommended Configurations

**For general documentation RAG:**
```
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=true
```

**For AI coding assistant with code examples:**
```
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
```

**For fast, basic RAG:**
```
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=false
```

## Database Browsing and Management

The project includes multiple ways to browse and interact with your PostgreSQL database directly, allowing you to inspect crawled content, search data, and monitor the system.

### Method 1: Custom Database Browser (Recommended)

A custom Python script provides an interactive CLI interface specifically designed for the Crawl4AI RAG database:

```bash
# Install required dependency
pip install asyncpg

# Run the interactive database browser
python db_browser.py
```

**Features:**
- 📊 **Database Overview** - Statistics for all tables (sources, crawled_pages, code_examples)
- 📚 **Source Browser** - Lists all crawled sources with metadata and chunk counts
- 📄 **Recent Content Browser** - Shows recently crawled pages with previews
- 🔍 **Content Search** - Search through crawled content by keywords
- 💻 **Code Examples Browser** - Browse extracted code examples (if `USE_AGENTIC_RAG=true`)
- 🔧 **Custom SQL Query** - Execute any SQL query you want
- 🎨 **Rich Formatting** - Beautiful CLI interface with tables and panels

### Method 2: Command Line (psql) - Quick Access

**Basic connection:**
```bash
# Connect to your database
psql -h localhost -U mg -d crawl4ai_rag

# Or using the DATABASE_URL from your .env
psql postgresql://mg@localhost:5432/crawl4ai_rag
```

**Quick data exploration commands:**
```bash
# Check what content you have
psql postgresql://mg@localhost:5432/crawl4ai_rag -c "
SELECT source_id, summary, total_word_count, created_at::date
FROM sources
ORDER BY created_at DESC;"

# Count crawled pages by source
psql postgresql://mg@localhost:5432/crawl4ai_rag -c "
SELECT source_id, COUNT(*) as chunk_count,
       MIN(created_at) as first_crawled,
       MAX(created_at) as last_crawled
FROM crawled_pages
GROUP BY source_id
ORDER BY chunk_count DESC;"

# Recent crawled content
psql postgresql://mg@localhost:5432/crawl4ai_rag -c "
SELECT url, chunk_number, LEFT(content, 100) as preview, source_id, created_at::date
FROM crawled_pages
ORDER BY created_at DESC
LIMIT 10;"

# Search for specific content (text search)
psql postgresql://mg@localhost:5432/crawl4ai_rag -c "
SELECT url, chunk_number, LEFT(content, 150) as preview
FROM crawled_pages
WHERE content ILIKE '%python%'
LIMIT 5;"

# Database size information
psql postgresql://mg@localhost:5432/crawl4ai_rag -c "
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;"
```

### Method 3: GUI Database Tools

**pgAdmin (Web-based, Free):**
```bash
# Install pgAdmin
brew install --cask pgadmin4

# Setup connection:
# - Name: Crawl4AI RAG
# - Host: localhost, Port: 5432
# - Database: crawl4ai_rag, Username: mg
```

**DBeaver (Cross-platform, Free):**
```bash
# Install DBeaver Community Edition
brew install --cask dbeaver-community

# Connection: localhost:5432/crawl4ai_rag (user: mg)
```

**TablePlus (macOS, Free tier available):**
```bash
# Install TablePlus
brew install --cask tableplus

# Free tier: 2 connections, 2 tabs (sufficient for single database)
```

### Method 4: Database Schema Exploration

**Useful psql commands once connected:**
```sql
-- List all tables
\dt

-- Describe table structure
\d sources
\d crawled_pages
\d code_examples

-- View table sizes
\dt+

-- List all indexes
\di

-- Show database info
\l
```

**Key tables in your database:**
- **`sources`** - Crawled source metadata (domains, summaries, word counts)
- **`crawled_pages`** - Content chunks with vector embeddings for semantic search
- **`code_examples`** - Extracted code examples with AI-generated summaries (if `USE_AGENTIC_RAG=true`)

## Running the Server

### Quick Start (Recommended)

```bash
# Fast startup with optimized script (90% faster)
./start_mcp_server.sh
```

### Manual Startup

```bash
# Activate environment and run
source crawl_venv/bin/activate
python src/crawl4ai_mcp.py
```

### Alternative with uv

```bash
uv run src/crawl4ai_mcp.py
```

The server will start and listen on the configured host and port (default: http://localhost:8051/sse).

> **✅ Ready for Integration**: Your MCP server is now running and ready to accept connections from Claude Desktop, Windsurf, or any MCP-compatible client using the configuration shown in the [Integration section](#integration-with-mcp-clients) above.

## Troubleshooting

### Common Issues and Solutions

#### Database Connection Errors
```bash
# Verify PostgreSQL is running
brew services list | grep postgresql

# Test database connection
psql -d crawl4ai_rag -c "SELECT 1;"
```

#### Server Startup Issues
- ✅ **Fixed**: Startup optimization implemented (90% faster)
- Use `./start_mcp_server.sh` for optimized startup
- Verify virtual environment exists: `ls crawl_venv/`

#### Vector Embedding Errors
- ✅ **Fixed**: PostgreSQL vector format compatibility resolved
- All embedding operations now work correctly

#### MCP Tool Execution Errors
- ✅ **Fixed**: Function naming conflicts resolved
- All tools return proper JSON responses

### Testing Your Setup

```bash
# Install testing dependencies
uv pip install -e ".[dev]"

# Run comprehensive test suite
pytest tests/ -v --cov=src --cov-report=html

# Test MCP tools individually  
pytest tests/test_mcp_tools.py -v

# Test logging configuration
pytest tests/test_logging.py -v

# Test MCP Inspector connection
npx @modelcontextprotocol/inspector
# Connect to: http://localhost:8051/sse

# Test basic crawling
# Use crawl_single_page tool with: https://example.com
```

## Building Your Own MCP Server

This implementation provides a foundation for building more complex MCP servers with web crawling and RAG capabilities.

### Extending the MCP Server

To build your own MCP server based on this implementation:

1. **Add Custom Tools**: Create methods with the `@mcp.tool()` decorator from the official MCP Python SDK
2. **Custom Lifespan Management**: Extend the lifespan function to add your own dependencies (database connections, API clients, etc.)
3. **Utility Functions**: Modify `utils.py` for domain-specific helper functions
4. **Specialized Crawlers**: Extend crawling capabilities for specific content types or websites
5. **Database Schema**: Customize the PostgreSQL schema in `crawled_pages.sql` for your use case

### MCP Server Development Patterns

```python
from mcp.server.fastmcp import FastMCP
from mcp import mcp

# Create your custom MCP server
app = FastMCP("your-server-name")

@mcp.tool()
async def your_custom_tool(ctx: mcp.Context, param: str) -> dict:
    """Your custom tool description."""
    # Access shared resources via context
    db_pool = ctx.request_context.lifespan_context["postgres_pool"]
    
    # Implement your logic
    result = await your_custom_logic(param, db_pool)
    
    return {"success": True, "result": result}
```

### Key Design Principles

When extending this MCP server:

1. **Follow MCP Standards**: Use the official MCP Python SDK patterns for tool registration and context management
2. **Database-First Design**: PostgreSQL + pgvector provides the foundation for semantic search capabilities  
3. **Async Everything**: All operations are async for optimal performance
4. **Structured Responses**: Always return structured JSON responses from MCP tools
5. **Error Handling**: Implement graceful error handling with informative error messages
6. **Resource Management**: Use lifespan context for shared resources like database pools

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Areas

- **MCP Server Tools**: Add new crawling capabilities, data processing tools, or content extraction methods
- **Database Integration**: Enhance PostgreSQL schema, vector operations, or search algorithms  
- **RAG Strategies**: Implement advanced retrieval techniques, chunking strategies, or embedding models
- **Performance Optimization**: Improve crawling speed, database queries, or memory usage
- **MCP Client Integration**: Better support for Claude Desktop, Windsurf, and other MCP clients
- **Testing**: Expand test coverage for MCP tools, database operations, and integration patterns

## License

This project is licensed under the MIT License - see the LICENSE file for details.