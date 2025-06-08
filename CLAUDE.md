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

## 🚀 Project Initialization Checklist

When starting a new project with this CLAUDE.md:

### 1. Update Project Context
- [ ] Fill in Primary Tech Stack section with project-specific dependencies
- [ ] Complete Project Architecture section with initial architecture decisions
- [ ] Add initial tasks to Current Tasks section
- [ ] Update Essential Development Commands with project-specific tools

### 2. Customize Standards
- [ ] Adjust coverage targets based on project phase
- [ ] Define project-specific error handling patterns
- [ ] Set up project-specific security requirements

### 3. Initialize Repository
- [ ] Create README.md with project-specific setup instructions
- [ ] Set up initial project structure
- [ ] Configure development environment

## 🏗️ Project Context & Standards

### Primary Tech Stack
- **Language:** Python 3.12+
- **Package Manager:** UV (`uv venv`, `uv pip install`)
- **MCP Framework:** FastMCP for Model Context Protocol server implementation
- **Web Crawling:** Crawl4AI with AsyncWebCrawler
- **Database:** PostgreSQL with pgvector extension for vector storage
- **AI/ML:** OpenAI API for embeddings and LLM processing
- **Key Dependencies:** crawl4ai, mcp, asyncpg, openai, sentence-transformers

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

### External Dependencies

#### Critical Services
- **OpenAI API**: Embeddings (text-embedding-3-small) and LLM processing
- **PostgreSQL**: Primary database with pgvector extension for vector operations

#### Optional Enhancements
- **Cross-encoder Models**: For reranking (runs locally, CPU-based)
- **Docker**: For containerized deployment

### Deployment Strategy

#### Local Development
```bash
uv venv && source .venv/bin/activate
uv pip install -e .
crawl4ai-setup
```

#### Docker Deployment
```bash
docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
```

#### MCP Client Integration
- **SSE Transport**: HTTP server on configurable port (default 8051)
- **Stdio Transport**: Direct process communication for desktop clients
- **Configuration**: Environment-based with .env file support
</project_architecture>

<current_tasks>
## Current Tasks & Priorities

### Active Tasks
<!-- Tasks currently being worked on -->

### Completed Tasks
<!-- Recently completed tasks with completion dates -->
- [x] Migrate from Supabase to PostgreSQL (2025-01-08) - TASK-MIGRATE-001
- [x] Setup project structure and MCP server implementation (2025-01-08)
- [x] Document project architecture in CLAUDE.md (2025-01-08) - TASK-001

### Backlog
<!-- Future tasks and improvements -->
- [ ] Add support for multiple embedding models (Ollama integration)
- [ ] Implement Context 7-inspired chunking strategy
- [ ] Performance optimization for crawling speed
- [ ] Integration with Archon knowledge engine
- [ ] Enhanced configuration management for RAG strategies

### Task Guidelines
- Each task should have a unique ID (TASK-001, TASK-002, etc.)
- Include brief description and acceptance criteria
- Mark completion date when finished
- Reference task ID in commit messages
</current_tasks>

## 📏 Code Quality & Structure Standards

### File Organization
- **Maximum 500 lines per file** - split when approaching limit
- **Single responsibility** per function/class
- **Functions <50 lines, methods <30 lines**
- **Nesting maximum 3 levels**

### Type Safety & Validation
- **Complete type hints** for all functions, methods, attributes
- **Avoid `Any`** except in documented circumstances  
- **Pydantic models** at all system boundaries
- **Custom exception classes** for domain-specific errors

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

### Error Handling Principles
- **Use specific exception types** instead of generic exceptions
- **Log all exceptions** with context information
- **Try/except/finally** for all external service calls
- **Graceful degradation** for non-critical failures
- **Context managers** (`with` statements) for resource management

## 🧪 Testing Requirements

### Test Structure
- **Mirror main app structure** in `/tests` directory
- **AAA pattern:** Arrange-Act-Assert for test clarity
- **Fixtures** for test setup and reusable components
- **Mock external dependencies** (APIs, databases)

### Required Test Coverage
**For each new feature, include minimum:**
1. **Happy path test** - correct behavior with valid inputs
2. **Edge case test** - boundary conditions  
3. **Failure case test** - graceful error handling with invalid inputs

### Coverage Targets
- **Production phase:** >80% coverage
- **Prototype phase:** >60% coverage
- **Focus on critical paths** and business logic over boilerplate

## 🔧 Environment & Dependencies

### Virtual Environment
- **One environment per project** using `uv venv -p python3.11 .venv`
- **Activate:** `source .venv/bin/activate`
- **Install:** `uv pip install <package>`
- **Track in pyproject.toml** with appropriate version constraints

### Version Management
- **Use ^version** for libraries with semver
- **Use ==version** for critical infrastructure
- **Pin exact versions** in requirements.txt only
- **Store secrets in .env** files (excluded in .gitignore)

## 📦 Git Workflow & Version Control

### Commit Strategy: Early & Often
**ALWAYS commit after completing any task:**
- One commit per completed task with task ID in message
- Use conventional commit format: `type(scope): description - TASK-ID`
- Commit immediately after task completion - don't accumulate changes

### Available Git MCP Commands
```python
# Status and review
git_status(repo_path="/path/to/repo")
git_diff_unstaged(repo_path="/path/to/repo")

# Stage and commit
git_add(repo_path="/path/to/repo", files=["specific_files"])
git_commit(repo_path="/path/to/repo", message="feat: description - TASK-034")

# Branch management
git_create_branch(repo_path="/path/to/repo", branch_name="feature/task-name")
git_checkout(repo_path="/path/to/repo", branch_name="target-branch")
```

### Post-Task Workflow
1. **Review:** `git_status()` and `git_diff_unstaged()`
2. **Stage:** Documentation first, then code: `git_add(files=["CLAUDE.md", "src/", "tests/"])`
3. **Commit:** `git_commit(message="feat(scope): description - TASK-ID")`

### Commit Types
- `feat:` New feature | `fix:` Bug fix | `docs:` Documentation only
- `refactor:` Code restructuring | `test:` Tests | `chore:` Maintenance

### Git Integration with Documentation Protocol
- Check `git_status` before starting work
- Update this CLAUDE.md file AS YOU CODE (not after)
- Commit with proper message including task ID
- Verify all documentation reflects current code state

## 🛡️ Security Guidelines

### Secret Management
- **Environment variables only** - never hardcode secrets
- **Use python-dotenv** for loading .env files
- **Rotate secrets regularly**
- **Different secrets** for dev/staging/production

### Input Validation & Safety
- **Validate all inputs** at system boundaries using Pydantic
- **Sanitize data** before storage or processing
- **Use parameterized queries** to prevent SQL injection
- **Apply rate limiting** for authentication attempts

## 📝 Documentation Standards

### Code Documentation
- **Google-style docstrings** for all public functions, classes, methods
- **Include:** Args, Returns, Raises, Examples sections
- **Inline comments** explaining "why" for complex logic
- **Document assumptions** and edge cases

### Comment Guidelines
- **"Reason:"** comments for non-obvious implementation choices
- **TODO comments** sparingly with associated task IDs
- **Step-by-step explanations** for complex algorithms

## 🔄 Development Workflow

### Code Review Process
- **Self-review** code before requesting review
- **Focused PRs** (<500 lines when possible)
- **Address security and performance** issues
- **Verify test coverage** for new functionality

### Pre-commit Requirements
- **Ruff formatting** applied to all code (`make format`)
- **Ruff linting** passes without errors (`make ruff`)
- **Type checking** with mypy passes (`make mypy`)
- **Security checks** with bandit pass (`make bandit`)
- **All tests pass** before commit (`make test`)

## 🧠 AI Behavior Rules

### Context Management
- **Never assume missing context** - ask for clarification when unclear
- **Confirm file paths** and module names before referencing
- **Only use approved libraries** or discuss new dependencies first
- **Never delete existing code** without explicit instruction

### Git Integration Requirements
- **Commit after every completed task** with Git MCP tools
- **Include task IDs in commit messages** for traceability
- **Never leave uncommitted changes** without explicit instruction

### Task Prioritization
- **Instructions in Current Tasks take precedence** for specific tasks
- **Point out conflicts** between general guidelines and task requirements
- **Be proactive** in updating this CLAUDE.md file after significant changes

### MCP Tools Implementation Patterns

#### Tool Development Guidelines
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

#### Tool Usage Patterns
```python
# Always check sources before filtering
sources = await get_available_sources()
# Then use specific source for targeted search
results = await perform_rag_query("python async", source="docs.python.org")

# Use smart_crawl_url for comprehensive site indexing
await smart_crawl_url("https://example.com/sitemap.xml", max_depth=2)
# Then search the newly indexed content
```

### Development Process
**For complex tasks, outline steps:**
1. Analyze the problem and break into subtasks
2. Plan documentation updates alongside implementation  
3. Implement systematically with tests
4. Verify solution and update documentation
5. Confirm all standards compliance

### Project-Specific Considerations
- **Environment Configuration** - RAG strategies are toggle-based via environment variables
- **Database Schema** - Always run migrations on crawled_pages.sql when schema changes
- **Embedding Models** - Currently hardcoded to text-embedding-3-small, future plans for model flexibility
- **Memory Management** - Use memory-adaptive dispatcher for large crawl operations
- **Error Recovery** - Implement exponential backoff for API calls and database operations

## 📤 Required Output Format

**ALWAYS include in responses:**

### 1. Brief Summary of Changes
Concise overview of what was implemented or changed.

### 2. Code Implementation
```python
# Complete, Black-formatted code with docstrings and type hints
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
- Code formatted with Black: [✓/✗/NA]
- Docstrings and type hints: [✓/✗/NA]

## 🎯 Development Phase Awareness

### Current Phase Assessment
Check Current Tasks section to determine current phase:

- **POC Phase:** Focus on core functionality validation, basic error handling
- **Prototype Phase:** Add stability, basic type hints, essential tests  
- **Production Phase:** Full type hints, comprehensive testing, complete documentation

### Quality Bar Adaptation
Adjust quality standards based on current phase while maintaining documentation synchronization throughout all phases.

## 🛠️ Essential Development Commands

### Environment Setup
```bash
# Initial setup - create virtual environment
uv venv

# Activate virtual environment (Linux/Mac)
source .venv/bin/activate
# OR Windows:
# .venv\Scripts\activate

# Install project dependencies
uv pip install -e .

# Setup Crawl4AI (required for web crawling)
crawl4ai-setup
```

### Database Setup
```bash
# Install PostgreSQL with pgvector (macOS)
brew install postgresql@16 pgvector

# Create database and run schema
psql -h localhost -U postgres -d crawl4ai_rag -f crawled_pages.sql

# OR using Docker for PostgreSQL
docker run --name crawl4ai-postgres \
  -e POSTGRES_PASSWORD=mypassword \
  -e POSTGRES_DB=crawl4ai_rag \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

### Running the MCP Server
```bash
# Run server directly (stdio transport)
uv run src/crawl4ai_mcp.py

# Run server with SSE transport (HTTP)
TRANSPORT=sse uv run src/crawl4ai_mcp.py

# Run with Docker
docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
```

### Configuration Management
```bash
# Create environment configuration
cp .env.example .env
# Edit .env with your settings:
# - OPENAI_API_KEY
# - DATABASE_URL or individual PostgreSQL settings
# - RAG strategy toggles (USE_CONTEXTUAL_EMBEDDINGS, etc.)

# Test configuration
python -c "from src.utils import create_postgres_pool; import asyncio; asyncio.run(create_postgres_pool())"
```

### Development & Testing
```bash
# Test database connection
python -c "from src.utils import get_postgres_connection; import asyncio; asyncio.run(get_postgres_connection())"

# Test single page crawl (manual testing)
curl -X POST "http://localhost:8051/tools/crawl_single_page" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# View available sources
curl -X POST "http://localhost:8051/tools/get_available_sources"

# Perform RAG query
curl -X POST "http://localhost:8051/tools/perform_rag_query" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "match_count": 5}'
```

### Debugging & Monitoring
```bash
# Check crawler health
python -c "from crawl4ai import AsyncWebCrawler; import asyncio; asyncio.run(AsyncWebCrawler().__aenter__())"

# Monitor database tables
psql -h localhost -U postgres -d crawl4ai_rag -c "SELECT COUNT(*) FROM crawled_pages;"
psql -h localhost -U postgres -d crawl4ai_rag -c "SELECT source_id, total_word_count FROM sources;"

# Check environment variables
env | grep -E "(OPENAI|DATABASE|POSTGRES|USE_)"
```

---

**Remember: Every code change MUST be accompanied by corresponding documentation updates in this CLAUDE.md file. If you cannot show which sections were updated, the task is incomplete.**