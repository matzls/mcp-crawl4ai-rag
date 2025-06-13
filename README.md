<h1 align="center">Crawl4AI RAG MCP Server</h1>

<p align="center">
  <em>Web Crawling and RAG Capabilities for AI Agents and AI Coding Assistants</em>
</p>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [Crawl4AI](https://crawl4ai.com) and PostgreSQL for providing AI agents and AI coding assistants with advanced web crawling and RAG capabilities.

With this MCP server, you can <b>scrape anything</b> and then <b>use that knowledge anywhere</b> for RAG.

The primary goal is to bring this MCP server into [Archon](https://github.com/coleam00/Archon) as I evolve it to be more of a knowledge engine for AI coding assistants to build AI agents. This first version of the Crawl4AI/RAG MCP server will be improved upon greatly soon, especially making it more configurable so you can use different embedding models and run everything locally with Ollama.

## Overview

This MCP server provides tools that enable AI agents to crawl websites, store content in a vector database (PostgreSQL with pgvector), and perform RAG over the crawled content. It follows the best practices for building MCP servers based on the [Mem0 MCP server template](https://github.com/coleam00/mcp-mem0/) I provided on my channel previously.

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

- **Smart URL Detection**: Automatically detects and handles different URL types (regular webpages, sitemaps, text files)
- **Recursive Crawling**: Follows internal links to discover content
- **Parallel Processing**: Efficiently crawls multiple pages simultaneously
- **Content Chunking**: Intelligently splits content by headers and size for better processing
- **Vector Search**: Performs RAG over crawled content, optionally filtering by data source for precision
- **Source Retrieval**: Retrieve sources available for filtering to guide the RAG process

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
   uv venv
   .venv\Scripts\activate
   # on Mac/Linux: source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   uv pip install -e .
   crawl4ai-setup
   ```

5. Create a `.env` file based on the configuration section below

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

# LLM for summaries and contextual embeddings
MODEL_CHOICE=gpt-4o-mini

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

## Running the Server

### Using Python with uv

```bash
uv run src/crawl4ai_mcp.py
```

The server will start and listen on the configured host and port.

## Integration with MCP Clients

### SSE Configuration

Once you have the server running with SSE transport, you can connect to it using this configuration:

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

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
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
>
> **Note**: Make sure PostgreSQL 17 is running and accessible at `localhost:5432` before starting the server.

### Stdio Configuration

Add this server to your MCP configuration for Claude Desktop, Windsurf, or any other MCP client:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "python",
      "args": ["path/to/crawl4ai-mcp/src/crawl4ai_mcp.py"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "DATABASE_URL": "postgresql://mg:@localhost:5432/crawl4ai_rag"
      }
    }
  }
}
```

### Alternative: Direct Python Execution

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
        "DATABASE_URL": "postgresql://mg:@localhost:5432/crawl4ai_rag"
      }
    }
  }
}
```

## Building Your Own Server

This implementation provides a foundation for building more complex MCP servers with web crawling capabilities. To build your own:

1. Add your own tools by creating methods with the `@mcp.tool()` decorator
2. Create your own lifespan function to add your own dependencies
3. Modify the `utils.py` file for any helper functions you need
4. Extend the crawling capabilities by adding more specialized crawlers