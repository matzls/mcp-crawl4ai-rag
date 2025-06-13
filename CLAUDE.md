# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# AI Assistant Memory - Project Context & Instructions

## 🎯 Role Definition & Primary Objective
You are an expert Code Assist Agent. Your primary objective is to help me develop and maintain my software project by providing accurate, reliable, and well-structured coding assistance, while **prioritizing documentation-as-you-go to maintain perfect synchronization between code and all project documentation.**

## 🚀 Quick Reference Guide

**Emergency Context Loading:**
- Read: Project Architecture + Current Tasks (below)
- Git: Check `git_status()` before starting
- Output: Follow format in section 📤

**Quality Gates (Non-negotiable):**
- Tests: 3+ cases (happy/edge/failure)
- Code: <500 lines, complete type hints
- Docs: Atomic updates to this CLAUDE.md file
- Git: Commit after every completed task

**Common Git Workflow:**
- Review: `git_status()` + `git_diff_unstaged()`
- Stage: `git_add(files=["CLAUDE.md", "src/", "tests/"])`
- Commit: `git_commit(message="feat(scope): description - TASK-ID")`

## 🚨 CRITICAL DOCUMENTATION REQUIREMENTS 🚨
**Documentation is not optional - it's a core part of every task.**

### Living Documentation Protocol
**ALWAYS start every coding session by:**
1. **READ** current Project Architecture section below
2. **CHECK** Current Tasks section for priorities  
3. **UPDATE** this CLAUDE.md file **AS YOU WORK** (not after)

### Documentation Standards
- **Atomic Updates:** Document each change as you make it
- **Context Preservation:** Explain WHY, not just WHAT
- **Future Self Test:** Write as if you'll forget everything in 6 months
- **Undocumented code or features are considered incomplete**

## 🏗️ Project Context & Standards

### Primary Tech Stack
- **Language:** Python 3.12+
- **Package Manager:** UV (`uv venv`, `uv pip install`)
- **MCP Framework:** FastMCP for Model Context Protocol server implementation
- **Web Crawling:** Crawl4AI with AsyncWebCrawler
- **Database:** PostgreSQL with pgvector extension for vector storage
- **AI/ML:** OpenAI API for embeddings and LLM processing
- **Agent Framework:** Pydantic AI for intelligent agent development with MCP integration
- **Key Dependencies:** crawl4ai, mcp, asyncpg, openai, sentence-transformers, pydantic-ai[logfire]

<project_architecture>
## Project Architecture

### Overview
This is a Model Context Protocol (MCP) server that provides web crawling and RAG (Retrieval Augmented Generation) capabilities for AI agents and coding assistants. It integrates Crawl4AI for intelligent web scraping with PostgreSQL vector database for semantic search and retrieval.

### Key Components

#### Core MCP Server (`src/crawl4ai_mcp.py`)
- **FastMCP Server**: Main server implementing MCP protocol with SSE/stdio transport
- **Lifespan Management**: Handles AsyncWebCrawler and PostgreSQL pool initialization
- **MCP Tools**: Implements 5 core tools for crawling and searching content

#### Crawl4AI Integration (Web Content Acquisition Layer)
- **Role**: Purely handles web crawling and content extraction
- **Capabilities**: 
  - Converts HTML to clean markdown using headless browser (Chrome/Chromium)
  - Handles JavaScript-heavy sites with dynamic content rendering
  - Extracts structured data (links, headers, code blocks)
  - Manages parallel processing with memory-adaptive dispatcher
  - Provides three crawling strategies: single page, batch, and recursive
- **Output**: Clean markdown content that feeds into custom RAG processing
- **Note**: Crawl4AI does NOT handle embedding, chunking, or database operations

#### Utility Layer (`src/utils.py`) - Custom RAG Processing
- **Database Operations**: PostgreSQL connection handling and vector operations
- **Embedding Generation**: OpenAI API integration for text embeddings (text-embedding-3-small)
- **Content Processing**: Smart chunking (respects code blocks), contextual embeddings, and code extraction

#### Database Schema (`crawled_pages.sql`)
- **Vector Search**: PostgreSQL with pgvector for semantic similarity search
- **Hybrid Search**: Combines vector and keyword search capabilities
- **Source Management**: Tracks crawled sources with summaries and metadata

### Architecture Decisions

#### MCP Protocol Implementation
- Uses FastMCP framework for clean tool-based architecture
- Implements both SSE (HTTP) and stdio transports for flexibility
- Context management pattern for sharing crawler and database connections

#### Advanced RAG Strategies (Configurable)
- **Contextual Embeddings**: Enhances chunks with document context for better retrieval
- **Hybrid Search**: Combines semantic vector search with keyword matching
- **Agentic RAG**: Specialized code example extraction and storage
- **Reranking**: Cross-encoder models for result relevance improvement

#### Crawling Intelligence
- **Smart URL Detection**: Automatically handles sitemaps, text files, and webpages
- **Recursive Crawling**: Follows internal links with configurable depth
- **Parallel Processing**: Memory-adaptive dispatcher for concurrent crawling

### Data Flow

#### Complete Processing Pipeline
1. **URL Input** → Smart detection (sitemap/txt/webpage)
2. **Crawl4AI Processing** → Web crawling and HTML-to-markdown conversion
3. **Custom Content Processing** → Smart chunking (respects code blocks), metadata extraction
4. **Custom Embedding** → OpenAI API (text-embedding-3-small) with optional contextual enhancement
5. **Custom Storage** → PostgreSQL with vector indices via custom database operations
6. **Custom Retrieval** → Vector similarity search + optional hybrid/reranking

#### Clear Separation of Responsibilities
- **Crawl4AI**: Web content acquisition (HTML → clean markdown)
- **Custom Code**: All RAG intelligence (chunking, embedding, storage, retrieval)

### Current System Status (Updated 2025-01-13)

#### Operational Services
- **PostgreSQL@17**: Running with pgvector extension enabled
- **Database**: `crawl4ai_rag` initialized with complete schema (sources, crawled_pages, code_examples tables)
- **MCP Server**: Running on http://localhost:8051/sse
- **SSE Endpoint**: Responding with proper event streams
- **Virtual Environment**: `crawl_venv` with all dependencies installed

#### Pydantic AI Agent Status - PRODUCTION READY
- **Agent Factory Functions**: Fully implemented and tested
  - `create_crawl_agent()` - Web crawling operations
  - `create_rag_agent()` - Content retrieval and semantic search
  - `create_workflow_agent()` - Multi-step workflow orchestration
- **Dependencies System**: Type-safe configuration with CrawlDependencies, RAGDependencies, WorkflowDependencies
- **Structured Outputs**: Complete Pydantic models (CrawlResult, RAGResult, WorkflowResult)
- **MCP Integration**: Proper MCPServerSSE client pattern with agent.run_mcp_servers() context manager
- **Code Quality**: Import errors fixed, example files corrected

#### Recent Key Fixes (All Resolved)
- ✅ **Vector Embedding Format**: PostgreSQL vector compatibility fixed (TASK-017)
- ✅ **Function Naming Conflicts**: MCP tool execution reliability improved (TASK-018)
- ✅ **Startup Performance**: 90% faster server initialization (TASK-015, TASK-016)
- ✅ **Pydantic AI Integration**: Complete agent framework integration (TASK-020)
- ✅ **Testing Results**: Production-ready performance verified (TASK-021)

### External Dependencies

#### Critical Services
- **OpenAI API**: Embeddings (text-embedding-3-small) and LLM processing
- **PostgreSQL**: Primary database with pgvector extension for vector operations

#### Optional Enhancements
- **Cross-encoder Models**: For reranking (runs locally, CPU-based)
</project_architecture>

<current_tasks>
## Current Tasks & Priorities

### Active Tasks
<!-- Tasks currently being worked on -->
- [x] Streamline CLAUDE.md file by removing redundant sections and condensing verbose content (2025-01-13) - TASK-022

### Completed Tasks
<!-- Recently completed tasks with completion dates -->
- [x] Migrate from Supabase to PostgreSQL (2025-01-08) - TASK-MIGRATE-001
- [x] Setup project structure and MCP server implementation (2025-01-08)
- [x] Document project architecture in CLAUDE.md (2025-01-08) - TASK-001
- [x] Add fork maintenance strategy and commands (2025-01-08) - TASK-008
- [x] Add comprehensive end-to-end testing plan and MCP Inspector setup (2025-01-08) - TASK-009
- [x] Update documentation to reflect PostgreSQL 17 configuration (2025-01-08) - TASK-010
- [x] Consolidate virtual environments to use only crawl_venv (2025-01-13) - TASK-014
- [x] Optimize MCP server startup to use existing virtual environment (2025-01-13) - TASK-015
- [x] Clean up project structure and finalize production-ready setup (2025-01-13) - TASK-016
- [x] Fix PostgreSQL vector embedding format issue (2025-01-13) - TASK-017
- [x] Fix function naming conflict causing NoneType callable error (2025-01-13) - TASK-018
- [x] Update documentation with all fixes and improvements (2025-01-13) - TASK-019
- [x] Add Pydantic AI agent integration with MCP server (2025-01-13) - TASK-020
- [x] Complete end-to-end testing and verification of system (2025-01-13) - TASK-021

### Backlog
<!-- Future tasks and improvements -->
- [ ] Add support for multiple embedding models (Ollama integration) - TASK-023
- [ ] Implement Context 7-inspired chunking strategy - TASK-024
- [ ] Performance optimization for crawling speed - TASK-025
- [ ] Integration with Archon knowledge engine - TASK-026
- [ ] Enhanced configuration management for RAG strategies - TASK-027

### Task Guidelines
- Each task should have a unique ID (TASK-001, TASK-002, etc.)
- Include brief description and acceptance criteria
- Mark completion date when finished
- Reference task ID in commit messages
</current_tasks>

## 📏 Development Standards

### Code Quality
- **File size**: <500 lines per file, split when approaching limit
- **Functions**: <50 lines, methods <30 lines, max 3 levels nesting
- **Type hints**: Complete for all functions, methods, attributes
- **Error handling**: Specific exception types, log with context
- **Testing**: 3+ cases per feature (happy/edge/failure)

### Import Organization
```python
# Standard library
import os
from typing import List, Optional

# Third-party
import pydantic
from fastapi import FastAPI

# Local application
from .models import User
from ..core import config
```

### Documentation Requirements
- **Google-style docstrings** for all public functions, classes, methods
- **Inline comments** explaining "why" for complex logic
- **Update CLAUDE.md** with architectural changes
- **TODO comments** sparingly with task IDs

## 🧪 Testing & Quality

### Test Structure
- **Mirror app structure** in `/tests` directory
- **AAA pattern**: Arrange-Act-Assert
- **Mock external dependencies** (APIs, databases)
- **Coverage targets**: >80% production, >60% prototype

### Key Test Scenarios
- **MCP Tools**: All 5 tools (crawl_single_page, smart_crawl_url, get_available_sources, perform_rag_query, search_code_examples)
- **Database Operations**: Vector storage and retrieval
- **Error Handling**: Invalid inputs, API failures, connection issues
- **RAG Strategies**: Contextual embeddings, hybrid search, reranking

## 🛠️ Essential Development Commands

### Environment Setup
```bash
# Initial setup
uv venv crawl_venv && source crawl_venv/bin/activate
uv pip install -e .
crawl4ai-setup

# Database setup (PostgreSQL 17 + pgvector)
brew install postgresql@17 pgvector
brew services start postgresql@17
createdb crawl4ai_rag
psql -h localhost -U $(whoami) -d crawl4ai_rag -f crawled_pages.sql
```

### Running the MCP Server
```bash
# Start server (SSE transport by default)
./start_mcp_server.sh

# For stdio transport
TRANSPORT=stdio ./start_mcp_server.sh

# Connect MCP Inspector
npx @modelcontextprotocol/inspector
# Then connect to: http://localhost:8051/sse
```

### Testing & Debugging
```bash
# Test database connection
python -c "from src.utils import create_postgres_pool; import asyncio; asyncio.run(create_postgres_pool())"

# Test MCP server
curl -X POST "http://localhost:8051/tools/get_available_sources"

# Monitor database
psql -h localhost -U $(whoami) -d crawl4ai_rag -c "SELECT COUNT(*) FROM crawled_pages;"
```

## 🧠 AI Behavior Rules

### Context Management
- **Never assume missing context** - ask for clarification when unclear
- **Only use approved libraries** or discuss new dependencies first
- **Never delete existing code** without explicit instruction

### Git Integration Requirements
- **Commit after every completed task** with Git MCP tools
- **Include task IDs in commit messages** for traceability
- **Update CLAUDE.md AS YOU WORK** (not after)

### MCP Tools Implementation Patterns
- **All tools are async** - use `async def` with `@mcp.tool()` decorator
- **Context Access** - get lifespan context via `ctx.request_context.lifespan_context`
- **Error Handling** - return JSON with success/error fields, never raise exceptions
- **Type Safety** - include complete type hints for all parameters

#### Available MCP Tools
1. **`crawl_single_page(url: str)`** - Quick single page crawl and storage
2. **`smart_crawl_url(url: str, max_depth: int = 3, max_concurrent: int = 10)`** - Intelligent crawling with URL type detection
3. **`get_available_sources()`** - List all crawled sources for filtering
4. **`perform_rag_query(query: str, source: str = None, match_count: int = 5)`** - Semantic search with optional source filtering
5. **`search_code_examples(query: str, source_id: str = None, match_count: int = 5)`** - Code-specific search (requires USE_AGENTIC_RAG=true)

### Project-Specific Considerations
- **Environment Configuration**: RAG strategies are toggle-based via environment variables
- **Database Schema**: Always run migrations on crawled_pages.sql when schema changes
- **Embedding Models**: Currently hardcoded to text-embedding-3-small
- **Memory Management**: Use memory-adaptive dispatcher for large crawl operations

## 📤 Required Output Format

**ALWAYS include in responses:**

### 1. Brief Summary of Changes
Concise overview of what was implemented or changed.

### 2. Code Implementation
```python
# Complete, formatted code with docstrings and type hints
```

### 3. Test Files Created/Modified
```python
# tests/test_feature.py - Complete Pytest code
```

### 4. Documentation Updates
```markdown
## CLAUDE.MD Updates
### Current Tasks Section
- [x] Task completed (YYYY-MM-DD) with notes

### Project Architecture Section  
- Updated with new architectural decision and rationale
```

### 5. Git Workflow Completion
- Changes committed with task ID: [✓/✗/NA]
- Repository status clean: [✓/✗/NA]

### 6. Final Verification Checklist
- `CLAUDE.MD` updated: [✓/✗/NA]
- Tests written/updated: [✓/✗/NA]
- All tests passing: [✓/✗/NA]
- Code formatted: [✓/✗/NA]
- Docstrings and type hints: [✓/✗/NA]

---

**Remember: Every code change MUST be accompanied by corresponding documentation updates in this CLAUDE.md file. If you cannot show which sections were updated, the task is incomplete.**