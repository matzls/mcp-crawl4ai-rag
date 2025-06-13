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

### Recent Fixes and Improvements

#### Database Layer Fixes (TASK-017)
- **Vector Embedding Format**: Fixed PostgreSQL vector type compatibility by converting Python lists to proper string format
- **Embedding Conversion**: Added `embedding_to_vector_string()` helper function for consistent format conversion
- **Database Operations**: All embedding insertions and queries now use correct PostgreSQL vector format

#### Function Architecture Fixes (TASK-018)
- **Naming Conflict Resolution**: Fixed recursive function call issue in `search_code_examples` tool
- **Import Aliasing**: Renamed utility function import to `search_code_examples_util` to avoid conflicts
- **MCP Tool Reliability**: All tools now return proper JSON responses without NoneType errors

#### Startup Optimization (TASK-015, TASK-016)
- **Virtual Environment Reuse**: Eliminated virtual environment creation overhead (90% faster startup)
- **Browser Initialization**: Fixed Crawl4AI browser setup issues for reliable startup
- **Clean Project Structure**: Removed testing artifacts and streamlined production setup

### External Dependencies

#### Critical Services
- **OpenAI API**: Embeddings (text-embedding-3-small) and LLM processing
- **PostgreSQL**: Primary database with pgvector extension for vector operations

#### Optional Enhancements
- **Cross-encoder Models**: For reranking (runs locally, CPU-based)

### Deployment Strategy

#### Local Development (Optimized)
```bash
# One-time setup
uv venv crawl_venv && source crawl_venv/bin/activate
uv pip install -e .
crawl4ai-setup

# Fast startup (90% faster)
./start_mcp_server.sh
```

#### Production Deployment (Optimized)
```bash
# Set environment variables and run with optimized startup
export OPENAI_API_KEY=your_key
export DATABASE_URL=postgresql://mg:@localhost:5432/crawl4ai_rag
export TRANSPORT=sse

# Use startup script for optimized performance
./start_mcp_server.sh

# Or manual startup
source crawl_venv/bin/activate && python src/crawl4ai_mcp.py
```

#### MCP Client Integration
- **SSE Transport**: HTTP server on configurable port (default 8051)
- **Stdio Transport**: Direct process communication for desktop clients
- **Configuration**: Environment-based with .env file support

#### Fork Maintenance Strategy
- **Automated Upstream Sync**: GitHub Actions workflow (`.github/workflows/sync-upstream.yml`) syncs upstream changes daily at 2 AM UTC
- **Upstream Repository**: `https://github.com/coleam00/mcp-crawl4ai-rag.git`
- **Branch Management**: Automatically merges upstream changes into main and feature/custom-database branches
- **Conflict Resolution**: Creates GitHub issue with label `sync-conflict` when merge conflicts occur
- **Manual Trigger**: Workflow can be triggered manually via GitHub Actions `workflow_dispatch`
- **Workflow Features**:
  - Compares SHA hashes to detect upstream changes
  - Uses `github-actions[bot]` for automated commits
  - Graceful handling of missing feature branches
  - Full fetch depth for complete history access
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
- [x] Add fork maintenance strategy and commands (2025-01-08) - TASK-008
- [x] Add comprehensive end-to-end testing plan and MCP Inspector setup (2025-01-08) - TASK-009
- [x] Update documentation to reflect PostgreSQL 17 configuration (2025-01-08) - TASK-010
- [x] Consolidate virtual environments to use only crawl_venv (2025-01-13) - TASK-014
- [x] Optimize MCP server startup to use existing virtual environment (2025-01-13) - TASK-015
- [x] Clean up project structure and finalize production-ready setup (2025-01-13) - TASK-016
- [x] Fix PostgreSQL vector embedding format issue (2025-01-13) - TASK-017
- [x] Fix function naming conflict causing NoneType callable error (2025-01-13) - TASK-018
- [x] Update documentation with all fixes and improvements (2025-01-13) - TASK-019

### Backlog
<!-- Future tasks and improvements -->
- [ ] Add support for multiple embedding models (Ollama integration) - TASK-009
- [ ] Implement Context 7-inspired chunking strategy - TASK-010
- [ ] Performance optimization for crawling speed - TASK-011
- [ ] Integration with Archon knowledge engine - TASK-012
- [ ] Enhanced configuration management for RAG strategies - TASK-013

### Task Guidelines
- Each task should have a unique ID (TASK-001, TASK-002, etc.)
- Include brief description and acceptance criteria
- Mark completion date when finished
- Reference task ID in commit messages

## 🧪 End-to-End Testing Plan

### MCP Inspector Setup and Usage

#### Prerequisites for Testing
1. **MCP Inspector Installation**: Use `npx @modelcontextprotocol/inspector` (no installation required)
2. **Database Setup**: PostgreSQL with pgvector extension running and accessible
3. **Environment Configuration**: Complete .env file with all required variables
4. **Test URLs**: Curated list of test websites for different scenarios
5. **Server Status**: Verify server starts without errors and accepts connections

#### Known Issues Resolved
- ✅ **Vector Embedding Format**: PostgreSQL vector compatibility fixed
- ✅ **Function Naming Conflicts**: MCP tool execution reliability improved
- ✅ **Startup Performance**: 90% faster server initialization
- ✅ **Browser Initialization**: Crawl4AI setup issues resolved

#### MCP Inspector Usage for Our Server

**Method 1: Startup Script (Recommended)**
```bash
# Start server (SSE transport by default)
./start_mcp_server.sh

# Connect MCP Inspector to: http://localhost:8051/sse
npx @modelcontextprotocol/inspector
```

**Method 2: Manual Startup (Advanced)**
```bash
# Activate environment and start server
source crawl_venv/bin/activate
python src/crawl4ai_mcp.py  # Defaults to SSE on port 8051

# Connect MCP Inspector to: http://localhost:8051/sse
npx @modelcontextprotocol/inspector
```

**Method 3: Stdio Transport (Alternative)**
```bash
# For stdio transport
TRANSPORT=stdio ./start_mcp_server.sh

# Connect MCP Inspector directly
npx @modelcontextprotocol/inspector /Users/mg/Desktop/mcp-workspace/local-mcp-servers/mcp-crawl4ai-rag/crawl_venv/bin/python src/crawl4ai_mcp.py
```

### Core Testing Scenarios

#### Test Data URLs
- **Simple Webpage**: `https://example.com` (basic HTML content)
- **Documentation Site**: `https://docs.python.org/3/tutorial/` (rich content with code examples)
- **Sitemap**: `https://docs.python.org/sitemap.xml` (XML sitemap testing)
- **Text File**: `https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/README.md` (direct text content)
- **JavaScript-Heavy Site**: `https://react.dev/learn` (dynamic content testing)

#### Environment Configurations for Testing
Create multiple .env configurations to test different RAG strategies:

**Basic Configuration (.env.basic)**
```
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false
USE_RERANKING=false
```

**Full Features Configuration (.env.full)**
```
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
```

**Hybrid Only Configuration (.env.hybrid)**
```
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=true
```

### Detailed Test Cases

#### Test Suite 1: Tool Availability and Schema Validation
**Objective**: Verify all tools are properly exposed and have correct schemas

**Test Steps**:
1. Connect MCP Inspector to server
2. Navigate to "Tools" tab
3. Verify all 5 tools are listed:
   - `crawl_single_page`
   - `smart_crawl_url`
   - `get_available_sources`
   - `perform_rag_query`
   - `search_code_examples` (only if USE_AGENTIC_RAG=true)

**Success Criteria**:
- All expected tools appear in the tools list
- Each tool shows proper parameter schema
- Tool descriptions are clear and accurate
- Parameter types and requirements are correctly specified

#### Test Suite 2: crawl_single_page Tool Testing
**Objective**: Test single page crawling functionality

**Test Case 2.1: Basic Webpage Crawling**
- **Input**: `url: "https://example.com"`
- **Expected**: Successful crawl with chunks stored in PostgreSQL
- **Verify**: Response includes chunk count, content length, source_id

**Test Case 2.2: Documentation Page with Code**
- **Input**: `url: "https://docs.python.org/3/tutorial/introduction.html"`
- **Expected**: Rich content extraction with proper chunking
- **Verify**: Code blocks preserved, headers extracted in metadata

**Test Case 2.3: Invalid URL Handling**
- **Input**: `url: "https://nonexistent-domain-12345.com"`
- **Expected**: Graceful error handling
- **Verify**: Error message returned, no database corruption

**Test Case 2.4: Code Example Extraction (USE_AGENTIC_RAG=true)**
- **Input**: `url: "https://docs.python.org/3/tutorial/controlflow.html"`
- **Expected**: Code examples extracted and summarized
- **Verify**: code_examples_stored > 0 in response

#### Test Suite 3: smart_crawl_url Tool Testing
**Objective**: Test intelligent URL detection and crawling strategies

**Test Case 3.1: Sitemap Crawling**
- **Input**: `url: "https://docs.python.org/sitemap.xml", max_depth: 2, max_concurrent: 5`
- **Expected**: Multiple URLs extracted and crawled in parallel
- **Verify**: crawl_type: "sitemap", pages_crawled > 1

**Test Case 3.2: Text File Direct Retrieval**
- **Input**: `url: "https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/README.md"`
- **Expected**: Direct content retrieval without browser rendering
- **Verify**: crawl_type: "text_file", content properly stored

**Test Case 3.3: Recursive Webpage Crawling**
- **Input**: `url: "https://docs.python.org/3/tutorial/", max_depth: 2, max_concurrent: 3`
- **Expected**: Internal links followed up to specified depth
- **Verify**: crawl_type: "webpage", multiple related pages crawled

**Test Case 3.4: Concurrent Crawling Limits**
- **Input**: `url: "https://docs.python.org/sitemap.xml", max_concurrent: 1`
- **Expected**: Crawling respects concurrency limits
- **Verify**: No overwhelming of target server, proper rate limiting

#### Test Suite 4: get_available_sources Tool Testing
**Objective**: Verify source discovery and metadata

**Test Case 4.1: Empty Database**
- **Precondition**: Clean database with no crawled content
- **Expected**: Empty sources list returned
- **Verify**: sources: [] in response

**Test Case 4.2: Multiple Sources**
- **Precondition**: Crawl content from 2-3 different domains
- **Expected**: All sources listed with metadata
- **Verify**: Each source has source_id, summary, word_count, timestamps

**Test Case 4.3: Source Summary Quality**
- **Precondition**: Crawl a well-known documentation site
- **Expected**: AI-generated summaries are relevant and informative
- **Verify**: Summary accurately describes the source content

#### Test Suite 5: perform_rag_query Tool Testing
**Objective**: Test semantic search and retrieval functionality

**Test Case 5.1: Basic Semantic Search**
- **Precondition**: Crawl Python documentation
- **Input**: `query: "how to define a function", match_count: 5`
- **Expected**: Relevant content chunks about Python functions
- **Verify**: Results contain function definition examples and explanations

**Test Case 5.2: Source-Filtered Search**
- **Precondition**: Crawl content from multiple sources
- **Input**: `query: "variables", source: "docs.python.org", match_count: 3`
- **Expected**: Results only from specified source
- **Verify**: All results have source_id matching filter

**Test Case 5.3: Hybrid Search (USE_HYBRID_SEARCH=true)**
- **Input**: `query: "def function", match_count: 5`
- **Expected**: Results combine semantic and keyword matching
- **Verify**: Response indicates hybrid search was used

**Test Case 5.4: Reranking (USE_RERANKING=true)**
- **Input**: `query: "error handling in Python", match_count: 10`
- **Expected**: Results reordered by relevance using cross-encoder
- **Verify**: Response includes rerank_score for each result

**Test Case 5.5: Empty Query Handling**
- **Input**: `query: "", match_count: 5`
- **Expected**: Graceful handling of empty query
- **Verify**: Appropriate error message or empty results

#### Test Suite 6: search_code_examples Tool Testing (USE_AGENTIC_RAG=true)
**Objective**: Test specialized code example search

**Test Case 6.1: Code-Specific Search**
- **Precondition**: Crawl documentation with USE_AGENTIC_RAG=true
- **Input**: `query: "for loop example", match_count: 3`
- **Expected**: Code examples with for loops and their summaries
- **Verify**: Results contain actual code blocks and AI-generated summaries

**Test Case 6.2: Source-Filtered Code Search**
- **Input**: `query: "class definition", source_id: "docs.python.org", match_count: 5`
- **Expected**: Code examples only from specified source
- **Verify**: All results match source filter

**Test Case 6.3: Tool Availability Check**
- **Precondition**: USE_AGENTIC_RAG=false
- **Expected**: search_code_examples tool not available
- **Verify**: Tool does not appear in MCP Inspector tools list

#### Test Suite 7: RAG Strategy Configuration Testing
**Objective**: Verify different RAG strategies work correctly

**Test Case 7.1: Contextual Embeddings (USE_CONTEXTUAL_EMBEDDINGS=true)**
- **Setup**: Enable contextual embeddings, crawl content
- **Input**: Perform RAG query on crawled content
- **Expected**: Enhanced retrieval quality due to contextual information
- **Verify**: Metadata indicates contextual_embedding: true

**Test Case 7.2: Strategy Combination Testing**
- **Setup**: Enable all RAG strategies (full configuration)
- **Input**: Complex query requiring multiple strategies
- **Expected**: All strategies work together without conflicts
- **Verify**: Response indicates all enabled strategies were used

#### Test Suite 8: Error Handling and Edge Cases
**Objective**: Test system robustness and error recovery

**Test Case 8.1: Database Connection Failure**
- **Setup**: Temporarily stop PostgreSQL service
- **Expected**: Graceful error handling, informative error messages
- **Verify**: Server doesn't crash, proper error responses

**Test Case 8.2: OpenAI API Failure**
- **Setup**: Use invalid OPENAI_API_KEY
- **Expected**: Embedding operations fail gracefully
- **Verify**: Appropriate error messages, fallback behavior

**Test Case 8.3: Large Content Handling**
- **Input**: Crawl very large webpage (>100KB content)
- **Expected**: Proper chunking, no memory issues
- **Verify**: Content split into appropriate chunks, all stored successfully

**Test Case 8.4: Concurrent Tool Calls**
- **Setup**: Make multiple simultaneous tool calls
- **Expected**: Proper handling of concurrent requests
- **Verify**: No race conditions, all requests complete successfully

### Testing Execution Workflow

#### Phase 1: Environment Setup Verification
1. **Database Connectivity**: Test PostgreSQL connection and schema
2. **Environment Variables**: Verify all required variables are set
3. **MCP Server Startup**: Confirm server starts without errors
4. **MCP Inspector Connection**: Establish connection via SSE or stdio

#### Phase 2: Basic Functionality Testing
1. Execute Test Suites 1-2 (tool availability and basic crawling)
2. Verify database operations work correctly
3. Test error handling with invalid inputs

#### Phase 3: Advanced Feature Testing
1. Execute Test Suites 3-6 (advanced crawling and RAG functionality)
2. Test different RAG strategy configurations
3. Verify performance with larger datasets

#### Phase 4: Integration and Stress Testing
1. Execute Test Suites 7-8 (configuration and error handling)
2. Test concurrent operations and edge cases
3. Verify system stability under load

### Success Criteria Summary

**Critical Success Criteria (Must Pass)**:
- ✅ All 5 tools properly exposed and functional (fixed naming conflicts)
- ✅ Basic crawling and storage operations work (fixed vector format)
- ✅ RAG queries return relevant results (database compatibility resolved)
- ✅ Database operations complete without errors (embedding format fixed)
- ✅ Error handling prevents system crashes (MCP tool reliability improved)
- ✅ Server startup is fast and reliable (optimization completed)

**Quality Success Criteria (Should Pass)**:
- RAG strategies enhance retrieval quality as expected
- Performance meets acceptable thresholds
- Code example extraction works when enabled
- Source filtering operates correctly

**Excellence Success Criteria (Nice to Have)**:
- Advanced RAG strategies show measurable improvement
- System handles edge cases gracefully
- Concurrent operations perform well
- Error messages are helpful and actionable

### Troubleshooting Common Issues

#### Database Connection Issues
**Problem**: `asyncpg.exceptions.ConnectionDoesNotExistError`
**Solution**:
1. Verify PostgreSQL is running: `brew services list | grep postgresql`
2. Check database exists: `psql -d crawl4ai_rag -c "\dt"`
3. Verify DATABASE_URL in .env file

#### Vector Embedding Errors
**Problem**: `invalid input for query argument` with list/vector type mismatch
**Solution**: ✅ **FIXED** - Vector embeddings now properly converted to PostgreSQL format

#### MCP Tool Execution Errors
**Problem**: `'NoneType' object is not callable` in MCP responses
**Solution**: ✅ **FIXED** - Function naming conflicts resolved, all tools return proper JSON

#### Server Startup Issues
**Problem**: Slow startup or browser initialization failures
**Solution**: ✅ **FIXED** - Optimized to use existing virtual environment, 90% faster startup

#### MCP Inspector Connection Issues
**Problem**: Cannot connect to SSE endpoint
**Solution**:
1. Verify server is running: Check for "Uvicorn running on http://0.0.0.0:8051"
2. Use correct URL: `http://localhost:8051/sse`
3. Try alternative: Use stdio transport with `TRANSPORT=stdio`

### Quick Reference: Recent Fixes

#### What Was Fixed (January 13, 2025)
1. **PostgreSQL Vector Format** (TASK-017)
   - Issue: Python list embeddings incompatible with PostgreSQL vector type
   - Fix: Added `embedding_to_vector_string()` conversion function
   - Impact: All database operations now work correctly

2. **Function Naming Conflict** (TASK-018)
   - Issue: MCP tool `search_code_examples` recursively calling itself
   - Fix: Renamed utility import to `search_code_examples_util`
   - Impact: All MCP tools return proper JSON responses

3. **Startup Optimization** (TASK-015, TASK-016)
   - Issue: Slow server startup due to virtual environment creation
   - Fix: Reuse existing virtual environment, streamlined startup script
   - Impact: 90% faster server initialization

#### Verification Commands
```bash
# Test server startup
./start_mcp_server.sh

# Test database connection
psql -d crawl4ai_rag -c "SELECT COUNT(*) FROM crawled_pages;"

# Test MCP Inspector connection
npx @modelcontextprotocol/inspector
# Connect to: http://localhost:8051/sse
```
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
- **One environment per project** using `uv venv -p python3.12 crawl_venv`
- **Activate:** `source crawl_venv/bin/activate`
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
uv venv crawl_venv

# Activate virtual environment (Linux/Mac)
source crawl_venv/bin/activate
# OR Windows:
# crawl_venv\Scripts\activate

# Install project dependencies
uv pip install -e .

# Setup Crawl4AI (required for web crawling)
crawl4ai-setup
```

### Database Setup
```bash
# Install PostgreSQL 17 with pgvector (macOS)
brew install postgresql@17 pgvector

# Start PostgreSQL service
brew services start postgresql@17

# Add PostgreSQL 17 to PATH (add to your shell profile)
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"

# Create database and run schema
createdb crawl4ai_rag
psql -h localhost -U $(whoami) -d crawl4ai_rag -f crawled_pages.sql
```

### Running the MCP Server

**Method 1: Startup Script (Recommended)**
```bash
# Start server (SSE transport by default)
./start_mcp_server.sh

# For stdio transport
TRANSPORT=stdio ./start_mcp_server.sh

# With custom port
PORT=8052 ./start_mcp_server.sh
```

**Method 2: Direct Execution**
```bash
# Activate virtual environment
source crawl_venv/bin/activate

# Start server (SSE by default)
python src/crawl4ai_mcp.py

# Or with specific transport
TRANSPORT=stdio python src/crawl4ai_mcp.py
```

### Configuration Management

**Environment Configuration:**
```bash
# Create environment configuration
cp .env.example .env
# Edit .env with your settings:
# - OPENAI_API_KEY
# - DATABASE_URL or individual PostgreSQL settings
# - RAG strategy toggles (USE_CONTEXTUAL_EMBEDDINGS, etc.)

# Test configuration
source crawl_venv/bin/activate
python -c "from src.utils import create_postgres_pool; import asyncio; asyncio.run(create_postgres_pool())"
```

**Quick Start Summary:**
```bash
# 1. Start the server
./start_mcp_server.sh

# 2. Connect MCP Inspector
npx @modelcontextprotocol/inspector
# Then connect to: http://localhost:8051/sse

# 3. Test basic functionality
# Use the tools in MCP Inspector to crawl and search content
```

### Development & Testing

**Prerequisites: Activate Virtual Environment**
```bash
# Always activate the virtual environment first for optimal performance
source crawl_venv/bin/activate
```

**Database and Environment Testing:**
```bash
# Test database connection
python -c "import asyncpg; import asyncio; asyncio.run(asyncpg.connect('postgresql://mg:@localhost:5432/crawl4ai_rag').close())"

# Test configuration loading
python -c "from src.utils import create_postgres_pool; import asyncio; asyncio.run(create_postgres_pool())"
```

**MCP Server Testing (SSE Transport):**
```bash
# Start server in background
TRANSPORT=sse python src/crawl4ai_mcp.py &
SERVER_PID=$!

# Test single page crawl
curl -X POST "http://localhost:8051/tools/crawl_single_page" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# View available sources
curl -X POST "http://localhost:8051/tools/get_available_sources"

# Perform RAG query
curl -X POST "http://localhost:8051/tools/perform_rag_query" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "match_count": 5}'

# Stop server
kill $SERVER_PID
```

### Debugging & Monitoring
```bash
# Check crawler health
python -c "from crawl4ai import AsyncWebCrawler; import asyncio; asyncio.run(AsyncWebCrawler().__aenter__())"

# Monitor database tables
psql -h localhost -U $(whoami) -d crawl4ai_rag -c "SELECT COUNT(*) FROM crawled_pages;"
psql -h localhost -U $(whoami) -d crawl4ai_rag -c "SELECT source_id, total_word_count FROM sources;"

# Check environment variables
env | grep -E "(OPENAI|DATABASE|POSTGRES|USE_)"
```

### Fork Maintenance Commands
```bash
# Manual upstream sync (alternative to GitHub Actions)
git remote add upstream https://github.com/coleam00/mcp-crawl4ai-rag.git
git fetch upstream
git checkout main
git merge upstream/main --no-edit
git push origin main

# Sync feature branch manually
git checkout feature/custom-database
git merge main
# Resolve any conflicts manually
git commit
git push origin feature/custom-database

# Trigger automated workflow manually
gh workflow run sync-upstream.yml

# Check workflow status
gh run list --workflow=sync-upstream.yml
```

---

**Remember: Every code change MUST be accompanied by corresponding documentation updates in this CLAUDE.md file. If you cannot show which sections were updated, the task is incomplete.**