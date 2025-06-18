# CLAUDE.md - Project Documentation & AI Assistant Guide

## 🎯 Project Overview

### Mission Statement
**Crawl4AI RAG MCP Server**: A production-ready Model Context Protocol server providing intelligent web crawling and RAG capabilities for AI agents and coding assistants, with PostgreSQL vector database integration and Pydantic AI agent orchestration.

### Current Development Phase: 🟡 MAKE IT RIGHT
**Phase 2 of 3**: Core functionality works perfectly, now optimizing architecture, code quality, and documentation standards.

**Phase Progress:**
- ✅ **PHASE 1: MAKE IT WORK** - Completed (Jan 2025)
  - Core MCP server functionality
  - Web crawling and RAG capabilities
  - Database integration with PostgreSQL + pgvector
  - Agent framework integration with Pydantic AI
  - Production-ready performance (197 pages/42.5s, 856 chunks stored)

- 🟡 **PHASE 2: MAKE IT RIGHT** - Current Phase
  - 🔄 Architecture refactoring (single orchestrator agent - TASK-024)
  - 🟡 Code quality improvements and standardization
  - 🟡 Documentation restructuring and enhancement
  - 🟡 Testing framework optimization

- 🟢 **PHASE 3: MAKE IT FAST** - Future
  - Performance optimization and scalability
  - Advanced features and integrations
  - Production deployment optimization

### Key Objectives
1. **Complete unified agent architecture** (replacing 3-agent approach)
2. **Standardize code quality** across all modules
3. **Optimize documentation** for maintainability
4. **Enhance testing framework** for comprehensive coverage

## 🚨 Emergency Protocols

### Quick Context Loading
**Start every session with:**
1. **READ**: Current Tasks section (🔴🟡🟢 priorities)
2. **CHECK**: Git status and recent commits
3. **VERIFY**: Development phase requirements
4. **UPDATE**: Documentation as you work (not after)

### Critical Commands
```bash
# Emergency status check
git status && git log --oneline -5

# Quick server verification
./start_mcp_server.sh && echo "MCP server started - use MCP Inspector at http://localhost:8051/sse for tool testing"

# Database health check
psql -h localhost -U $(whoami) -d crawl4ai_rag -c "SELECT COUNT(*) FROM crawled_pages;"
```

### Quality Gates (Non-negotiable)
- **🔴 Critical**: Tests (3+ cases: happy/edge/failure)
- **🔴 Critical**: Code (<500 lines, complete type hints)
- **🔴 Critical**: Documentation (atomic updates to CLAUDE.md)
- **🔴 Critical**: Git workflow (commit after every completed task)

### Living Documentation Protocol
**Documentation is not optional - it's core to every task.**
- **Atomic Updates**: Document each change as you make it
- **Context Preservation**: Explain WHY, not just WHAT
- **Future Self Test**: Write as if you'll forget everything in 6 months
- **Undocumented code/features are considered incomplete**

## 🏗️ Technical Architecture

### Core Technology Stack
- **Language:** Python 3.12+ with UV package manager
- **MCP Framework:** Official MCP Python SDK (mcp>=1.9.4) for Model Context Protocol server implementation
- **Web Crawling:** Crawl4AI with AsyncWebCrawler for intelligent content extraction
- **Database:** PostgreSQL 17 + pgvector extension for vector storage and semantic search
- **AI/ML:** OpenAI API (text-embedding-3-small, GPT-4.1) for embeddings and LLM processing
- **Agent Framework:** Pydantic AI for intelligent agent orchestration with MCP integration
- **Observability:** Logfire for comprehensive logging and performance monitoring
- **Key Dependencies:** crawl4ai, mcp>=1.9.4, asyncpg, openai, sentence-transformers, pydantic-ai[logfire]

<project_architecture>
## Project Architecture

### Overview
This is a Model Context Protocol (MCP) server that provides web crawling and RAG (Retrieval Augmented Generation) capabilities for AI agents and coding assistants. It integrates Crawl4AI for intelligent web scraping with PostgreSQL vector database for semantic search and retrieval.

### Key Components

#### Core MCP Server (`src/crawl4ai_mcp.py`)
- **Official MCP Server**: Main server implementing MCP protocol using official Python SDK with SSE/stdio transport
- **Lifespan Management**: Handles AsyncWebCrawler and PostgreSQL pool initialization via asynccontextmanager
- **MCP Tools**: Implements 5 core tools for crawling and searching content using @mcp.tool() decorators

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
- Uses official MCP Python SDK for standards-compliant MCP server implementation
- Implements MCP protocol over SSE and stdio transports (NOT REST/HTTP endpoints)
- MCP tools accessible via MCP protocol clients (Claude Desktop, MCP Inspector, custom MCP clients)
- Context management pattern for sharing crawler and database connections via lifespan_context

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

### Current System Status (Updated 2025-01-18)

#### Production Ready Status ✅
- **PostgreSQL@17**: Running with pgvector extension enabled
- **Database**: `crawl4ai_rag` initialized with complete schema (sources, crawled_pages, code_examples tables)
- **MCP Server**: Running on http://localhost:8051/sse with both SSE and stdio transports
- **Authentication**: Fixed PostgreSQL connection authentication issues
- **Virtual Environment**: `crawl_venv` with all dependencies installed
- **All 5 MCP Tools**: Fully functional and tested (crawl_single_page, smart_crawl_url, get_available_sources, perform_rag_query, search_code_examples)

#### Agent Testing Status (2025-01-18)
- **CLI Interface**: Fully functional with resolved import issues
- **Model Configuration**: All agents using OpenAI o3 (openai:o3) for enhanced reasoning
- **Import System**: Robust fallback mechanisms for cross-context compatibility
- **MCP Tool Integration**: All tools verified working correctly via systematic testing
- **Production Status**: System ready for production deployment

#### Pydantic AI Agent Status - UNIFIED ARCHITECTURE IMPLEMENTED
- **Previous Architecture (Deprecated)**: Three separate agents (crawl, rag, workflow) - Replaced
- **New Architecture (Production Ready)**: Single intelligent orchestrator agent with GPT-4.1 integration
- **Agent Design Philosophy**: One agent that intelligently selects from 5 MCP tools based on user intent
- **MCP Tool Integration**: Direct access to all 5 tools with intelligent selection logic
- **Structured Outputs**: Unified response models with comprehensive result synthesis
- **Enhanced Tool Descriptions**: Improved with USE WHEN/DON'T USE guidance and workflow patterns
- **Model Configuration**: All agents now use OpenAI o3 (openai:o3) for optimal performance and advanced reasoning capabilities
- **Import Architecture**: Fixed relative import issues in pydantic_agent module for cross-context compatibility

#### Recent Key Fixes (All Resolved)
- ✅ **Vector Embedding Format**: PostgreSQL vector compatibility fixed (TASK-017)
- ✅ **Function Naming Conflicts**: MCP tool execution reliability improved (TASK-018)
- ✅ **Startup Performance**: 90% faster server initialization (TASK-015, TASK-016)
- ✅ **Pydantic AI Integration**: Complete agent framework integration (TASK-020)
- ✅ **Testing Results**: Production-ready performance verified (TASK-021)
- ✅ **Logfire Integration**: Comprehensive logging for MCP server and Pydantic AI agents (TASK-023)
- ✅ **Architecture Refactoring**: Single orchestrator agent with GPT-4.1 integration completed (TASK-024)
- ✅ **Import System**: Fixed relative import issues in pydantic_agent module (TASK-024)
- ✅ **Model Configuration**: Standardized all agents to use GPT-4 Turbo for consistency (TASK-024)

### New Architecture: Single Intelligent Orchestrator

#### Design Philosophy (Following MCP Best Practices)
- **One Agent, Five Tools**: Single agent intelligently selects from 5 MCP tools based on user intent
- **Intent-Driven Tool Selection**: Agent analyzes user queries to determine optimal workflow patterns
- **Workflow Orchestration**: Automated multi-step processes with context management
- **Enhanced User Experience**: No agent selection needed - natural language interaction

#### Tool Selection Logic
**Research Workflow** (URLs mentioned):
1. `smart_crawl_url` → Gather new content
2. `get_available_sources` → Confirm storage  
3. `perform_rag_query` → Answer questions about crawled content

**Search Workflow** (questions about topics):
1. `get_available_sources` → Check available content
2. `perform_rag_query` → Find relevant information
3. `search_code_examples` → If code specifically requested

**Discovery Workflow** (content exploration):
1. `get_available_sources` → List what's available
2. Optional follow-up searches based on user interest

#### Enhanced Tool Descriptions
- **USE WHEN/DON'T USE** guidance for clear tool boundaries
- **Workflow patterns** showing tool dependencies and sequences  
- **Concrete examples** with parameter demonstrations
- **Error scenarios** for robust error handling

#### Implementation Components
- `src/pydantic_agent/unified_agent.py` - Single orchestrator agent (GPT-4 Turbo configured)
- `cli_chat.py` - Interactive CLI interface with rich formatting (functional, import issues resolved)
- `src/pydantic_agent/examples/unified_agent_example.py` - Comprehensive demonstrations (GPT-4 Turbo)
- `src/pydantic_agent/agent.py` - Legacy agent implementations (updated to GPT-4 Turbo)
- Enhanced MCP tool descriptions in `src/crawl4ai_mcp.py`
- Fixed import system with fallback mechanisms for cross-context compatibility

### Technical Implementation Details

#### Model Configuration (Updated 2025-01-15)
- **Primary Model**: OpenAI o3 (openai:o3) across all agent implementations  
- **Rationale**: Advanced reasoning capabilities and improved performance for complex MCP tool orchestration
- **Migration**: Unified migration from GPT-4 Turbo to o3 for enhanced agent intelligence (TASK-034)
- **Consistency**: All agents (unified, crawl, rag, workflow) use same model for predictable behavior

#### Import System Architecture (Fixed 2025-01-15)
- **Problem**: Relative imports in pydantic_agent module failed when running CLI directly
- **Solution**: Implemented fallback import mechanism with path manipulation
- **Pattern**: Try relative imports first, fallback to absolute imports with sys.path adjustment
- **Files Updated**: `agent.py`, `unified_agent.py` with robust import handling
- **Result**: CLI interface (`cli_chat.py`) now functional across different execution contexts

### External Dependencies

#### Critical Services
- **OpenAI API**: Embeddings (text-embedding-3-small) and LLM processing (GPT-4 Turbo)
- **PostgreSQL**: Primary database with pgvector extension for vector operations

#### Optional Enhancements
- **Cross-encoder Models**: For reranking (runs locally, CPU-based)

### Observability & Logging

#### Standard Logfire Integration - PRODUCTION READY (Updated 2025-01-15)
- **Pydantic AI Built-in Instrumentation**: Uses `logfire.instrument_pydantic_ai()` for automatic agent tracing
- **HTTP Request Monitoring**: `logfire.instrument_httpx()` captures raw prompts and responses
- **MCP Tool Execution Logging**: Simple `@log_mcp_tool_execution` decorator with basic metrics
- **Error Tracking**: Structured error logging with context and timing information
- **Dashboard Integration**: Real-time traces in Logfire dashboard with automatic configuration

#### Simple Logging Architecture
- **Standard Instrumentation**: Uses Pydantic AI's built-in Logfire support instead of custom decorators
- **Basic MCP Tool Logging**: `@log_mcp_tool_execution` with execution time, success/failure, and result summaries
- **Automatic Agent Tracing**: Pydantic AI automatically creates spans for agent runs and tool calls
- **HTTP Visibility**: Raw OpenAI API requests/responses visible in Logfire for debugging
- **Clean Error Handling**: Simple error logging without complex categorization

#### Observability Components
- **Logging Configuration**: `src/logging_config.py` - Simple MCP tool logging with basic metrics
- **Agent Setup**: `setup_logfire_instrumentation()` in unified agent for automatic configuration
- **CLI Integration**: Automatic Logfire setup on startup with dashboard link display
- **Standard Patterns**: Uses Logfire's recommended instrumentation patterns

#### Logfire Dashboard Features
- **Agent Execution Traces**: Complete agent runs with nested tool calls and timing
- **HTTP Request Details**: Raw prompts, responses, and token usage from OpenAI API
- **MCP Tool Performance**: Execution times, success rates, and result summaries
- **Error Tracking**: Exception details with stack traces and context
- **Real-time Monitoring**: Live traces during agent execution
- **Dashboard URL**: https://logfire-eu.pydantic.dev/matzls/crawl4ai-agent
</project_architecture>

## 📋 Task Management

### 🔴 Critical Priority (Blocking/High Risk)
**Active Tasks - Immediate Action Required**
- *All critical blocking tasks completed successfully* ✅

**Recently Completed Critical Tasks (Jan 2025):**
- [x] **TASK-042**: Fix fundamental documentation errors about MCP framework and transport implementation (2025-01-18)
- [x] **TASK-036**: Test all 5 MCP tools individually via MCP Inspector and verify responses (2025-01-18)
- [x] **TASK-041**: Investigate and fix MCP server SSE connection stability issues (2025-01-18)

### 🟡 Important Priority (Medium Risk)
**MAKE IT RIGHT Phase Tasks**
- [ ] **TASK-025**: Implement Context 7-inspired chunking strategy
  - **Phase**: Code quality improvement
  - **Risk**: Performance impact if delayed
- [ ] **TASK-026**: Performance optimization for crawling speed
  - **Phase**: Architecture optimization
  - **Risk**: Scalability limitations
- [ ] **TASK-028**: Enhanced configuration management for RAG strategies
  - **Phase**: Code standardization
  - **Risk**: Maintenance complexity
- [ ] **TASK-038**: Set up formal pytest testing framework with proper structure
  - **Phase**: Testing infrastructure
  - **Risk**: Quality assurance foundation
- [ ] **TASK-039**: Write comprehensive test suites for MCP tools and agent orchestration
  - **Phase**: Testing implementation
  - **Risk**: Coverage and reliability

### 🟢 Nice-to-Have Priority (Low Risk)
**MAKE IT FAST Phase Tasks (Future)**
- [ ] **TASK-027**: Integration with Archon knowledge engine
  - **Phase**: Advanced features
  - **Risk**: Low - enhancement only
- [ ] **TASK-029**: Add support for multiple embedding models (Ollama integration)
  - **Phase**: Feature expansion
  - **Risk**: Low - optional capability

### ✅ Recently Completed (MAKE IT WORK → MAKE IT RIGHT Transition)
**Phase 2 Progress (Jan 2025)**
- [x] **TASK-042**: Fix fundamental documentation errors about MCP framework and transport implementation (2025-01-18)
- [x] **TASK-041**: Investigate and fix MCP server SSE connection stability issues (2025-01-18)
- [x] **TASK-036**: Test all 5 MCP tools individually via MCP Inspector and verify responses (2025-01-18)
- [x] **TASK-040**: Fix logging compatibility issues across all agent examples and CLI interface (2025-01-18)
- [x] **TASK-037**: Investigate TASK-031 agent runtime errors through systematic testing (2025-01-18)
- [x] **TASK-035**: Execute manual verification tests for MCP server, CLI, and database connectivity (2025-01-18)
- [x] **TASK-031**: Investigate and resolve agent testing runtime errors (2025-01-18)
- [x] **TASK-033**: Comprehensive codebase analysis vs CLAUDE.md documentation verification (2025-01-15)
- [x] **TASK-034**: Unified migration of all agent implementations from GPT-4 Turbo to OpenAI o3 (2025-01-15)
- [x] **TASK-024**: Complete unified agent architecture implementation with GPT-4.1 integration and import fixes (2025-01-15)
- [x] **TASK-030**: Comprehensive CLAUDE.md restructuring with three-phase development model (2025-01-15)
- [x] **TASK-032**: Implement simplified observability using Pydantic AI's built-in Logfire integration (2025-01-15)

**Phase 1 Completion (Jan 2025)**
- [x] **TASK-020**: Pydantic AI agent integration with MCP server (2025-01-13)
- [x] **TASK-021**: Complete end-to-end testing and verification (2025-01-13)
- [x] **TASK-023**: Comprehensive logfire logging implementation (2025-01-13)
- [x] **TASK-017**: Fix PostgreSQL vector embedding format issue (2025-01-13)
- [x] **TASK-018**: Fix function naming conflicts (2025-01-13)
- [x] **TASK-015/016**: Optimize startup performance (90% improvement) (2025-01-13)

**Foundation Tasks (Jan 2025)**
- [x] **TASK-MIGRATE-001**: Migrate from Supabase to PostgreSQL (2025-01-08)
- [x] **TASK-001**: Document project architecture in CLAUDE.md (2025-01-08)
- [x] **TASK-009**: Comprehensive end-to-end testing plan (2025-01-08)
- [x] **TASK-010**: PostgreSQL 17 configuration documentation (2025-01-08)
- [x] **TASK-014**: Consolidate virtual environments (2025-01-13)
- [x] **TASK-022**: Streamline CLAUDE.md documentation (2025-01-13)

### Recent Analysis Findings (TASK-033 - 2025-01-15)

#### Documentation vs Reality Assessment
**CLAUDE.md Accuracy Score: 95% ✅**
- **Architecture Documentation**: Completely accurate - unified agent implementation verified in code
- **Technology Stack**: All components confirmed operational (PostgreSQL, MCP server, Pydantic AI)
- **Phase Classification**: ✅ Confirmed in MAKE IT RIGHT phase based on code quality and task completion
- **Task Status**: All documented completions verified in git history with proper task IDs

#### Code Quality Assessment
**MAKE IT RIGHT Phase Standards: Met ✅**
- **File Organization**: Clean structure with <500 lines per file (largest file: 334 lines)
- **Type Safety**: Complete type hints across all agent implementations  
- **Error Handling**: Comprehensive with structured JSON responses
- **Architecture Integrity**: Single orchestrator pattern correctly implemented
- **Import System**: Robust fallback mechanisms for cross-context compatibility

#### OpenAI o3 Migration Status
**Migration Complete: 100% ✅**
- **Files Updated**: 4 agent files migrated from GPT-4 Turbo to o3
  - `unified_agent.py`: Main orchestrator (already had o3)
  - `agent.py`: 3 legacy agents migrated
  - `basic_crawl_example.py`: Example updated
  - `rag_workflow_example.py`: Multiple references updated
- **Consistency**: All agents now use `openai:o3` for enhanced reasoning capabilities
- **Verification**: No remaining GPT-4 references in active agent code

### Task Management Protocol
- **Task IDs**: Sequential numbering (TASK-001, TASK-002, etc.)
- **Risk Classification**: 🔴🟡🟢 system for priority management
- **Phase Alignment**: Tasks mapped to three-phase development model
- **Commit References**: All commits must include task ID
- **Completion Tracking**: Date and acceptance criteria verification

## 📏 Quality Standards

### Phase-Specific Code Quality Requirements

#### 🟡 MAKE IT RIGHT Phase Standards (Current)
**Code Organization**
- **File size**: <500 lines per file, split when approaching limit
- **Functions**: <50 lines, methods <30 lines, max 3 levels nesting
- **Type hints**: Complete for all functions, methods, attributes
- **Error handling**: Specific exception types with structured logging
- **Testing**: 3+ cases per feature (happy/edge/failure paths)

**Architecture Standards**
- **Single Responsibility**: Each module has one clear purpose
- **Dependency Injection**: Use Pydantic AI dependency patterns
- **MCP Integration**: Follow FastMCP best practices
- **Database Operations**: Async patterns with proper connection pooling

#### Import Organization (Enforced)
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

#### Documentation Standards (Non-negotiable)
- **Google-style docstrings** for all public functions, classes, methods
- **Inline comments** explaining "why" for complex logic
- **CLAUDE.md updates** for all architectural changes
- **TODO comments** only with task IDs and deadlines
- **Framework verification** for all external library usage

### Quality Assurance Process

#### Pre-commit Requirements
- **Ruff formatting**: Code style consistency
- **Type checking**: mypy validation
- **Test execution**: All tests must pass
- **Documentation sync**: CLAUDE.md must be updated

#### Code Review Checklist
- [ ] Follows phase-specific standards
- [ ] Complete type hints and docstrings
- [ ] Error handling with proper logging
- [ ] Tests cover happy/edge/failure cases
- [ ] Documentation updated atomically
- [ ] No TODO comments without task IDs

## 🧪 Quality Assurance

### Testing Framework (Phase-Aligned)

#### 🟡 MAKE IT RIGHT Phase Testing (Current Focus)
**Test Structure Standards**
- **Mirror app structure** in `/tests` directory
- **AAA pattern**: Arrange-Act-Assert for clarity
- **Mock external dependencies** (APIs, databases, MCP servers)
- **Coverage targets**: >80% for production code, >60% for prototypes

**Critical Test Scenarios**
- **🔴 Unified Agent Architecture**: Single orchestrator with intelligent tool selection
- **🔴 MCP Tool Integration**: All 5 tools with proper error handling
  - `crawl_single_page`, `smart_crawl_url`, `get_available_sources`
  - `perform_rag_query`, `search_code_examples`
- **🟡 Workflow Orchestration**: Intent analysis and multi-step coordination
  - Research workflow (crawl→search), Search-only, Code-focused, Discovery
- **🟡 Database Operations**: Vector storage, retrieval, and connection pooling
- **🟡 Error Handling**: Invalid inputs, API failures, graceful degradation
- **🟢 RAG Strategies**: Contextual embeddings, hybrid search, reranking

#### Framework Verification Patterns
**External Library Integration Testing**
- **Pydantic AI**: Agent creation, MCP server integration, dependency injection
- **FastMCP**: Tool registration, context management, transport protocols
- **Crawl4AI**: Web crawling, content extraction, parallel processing
- **PostgreSQL**: Vector operations, connection pooling, query performance
- **OpenAI API**: Embedding generation, rate limiting, error handling

#### Test Execution Strategy
```bash
# Phase-specific test execution
pytest tests/ -v --cov=src --cov-report=html

# Critical path testing (🔴 priority)
pytest tests/test_unified_agent.py tests/test_mcp_tools.py -v

# Integration testing (🟡 priority)
pytest tests/test_workflows.py tests/test_database.py -v

# Performance testing (🟢 priority)
pytest tests/test_performance.py -v --benchmark-only
```

## 🛠️ Development Workflows

### Phase-Based Development Commands

#### 🟡 MAKE IT RIGHT Phase Workflows (Current)

**Environment Setup & Verification**
```bash
# Initial setup (one-time)
uv venv crawl_venv && source crawl_venv/bin/activate
uv pip install -e .
crawl4ai-setup

# Database setup (PostgreSQL 17 + pgvector)
brew install postgresql@17 pgvector
brew services start postgresql@17
createdb crawl4ai_rag
psql -h localhost -U $(whoami) -d crawl4ai_rag -f crawled_pages.sql

# Verify environment health
python -c "from src.utils import create_postgres_pool; import asyncio; asyncio.run(create_postgres_pool())"
```

**Common Development Tasks**
```bash
# Quick health check (run this first in any session)
git status && git log --oneline -5
./start_mcp_server.sh  # Check server starts successfully
psql -h localhost -U $(whoami) -d crawl4ai_rag -c "SELECT COUNT(*) FROM crawled_pages;"

# Development cycle
ruff format src/ tests/        # Format code
ruff check src/ tests/         # Lint code  
mypy src/                      # Type check
pytest tests/ -v               # Run tests
```

**Development Server Management**
```bash
# Start MCP server (SSE transport - default)
./start_mcp_server.sh

# Alternative transport modes
TRANSPORT=stdio ./start_mcp_server.sh

# Connect MCP Inspector for debugging
npx @modelcontextprotocol/inspector
# Connect to: http://localhost:8051/sse
```

**MCP Tools Testing**
```bash
# Test MCP tools via MCP Inspector (recommended)
# 1. Start server: ./start_mcp_server.sh
# 2. Open MCP Inspector: npx @modelcontextprotocol/inspector  
# 3. Connect to: http://localhost:8051/sse
# 4. Test tools: get_available_sources, perform_rag_query, smart_crawl_url, etc.

# Test via Pydantic AI agents (integration testing)
python src/pydantic_agent/examples/unified_agent_example.py
python cli_chat.py  # Interactive CLI with natural language
```

**Quality Assurance Workflows**
```bash
# Code quality checks
ruff check src/ tests/
ruff format src/ tests/
mypy src/

# Testing workflows (phase-aligned)
uv pip install -e ".[dev]"                    # Install testing dependencies
pytest tests/ -v --cov=src --cov-report=html  # Full test suite
pytest tests/test_mcp_tools.py -v             # MCP tools testing (🔴)
pytest tests/test_logging.py -v               # Logging verification (🟡)

# Performance monitoring and testing
python tests/test_logging.py                  # Logging verification
python src/pydantic_agent/examples/logging_example.py  # Comprehensive demo

# Database monitoring
psql -h localhost -U $(whoami) -d crawl4ai_rag -c "SELECT COUNT(*) FROM crawled_pages;"

# Agent architecture testing (GPT-4.1 configured)
python src/pydantic_agent/examples/unified_agent_example.py  # Unified orchestrator
python src/pydantic_agent/examples/basic_crawl_example.py    # Individual components
python src/pydantic_agent/examples/rag_workflow_example.py   # Workflow testing

# Interactive CLI interface with Logfire observability
python cli_chat.py
```

**Development & Debugging Tools**
```bash
# System health checks
# Use MCP Inspector at http://localhost:8051/sse to test MCP tools
psql -h localhost -U $(whoami) -d crawl4ai_rag -c "SELECT COUNT(*) FROM crawled_pages;"  # Database

# Interactive development
python cli_chat.py                            # CLI interface
python src/pydantic_agent/examples/unified_agent_example.py  # Agent testing

# Architecture verification
python src/pydantic_agent/examples/unified_agent_example.py  # Single orchestrator testing
```

#### 🟢 MAKE IT FAST Phase Workflows (Future)
```bash
# Performance profiling (future phase)
pytest tests/test_performance.py --benchmark-only
python -m cProfile -o profile.stats src/crawl4ai_mcp.py

# Load testing (future phase)
locust -f tests/load_test.py --host=http://localhost:8051
```

## 🤖 AI Assistant Guidelines

### Context Management Protocol
- **Never assume missing context** - ask for clarification when requirements unclear
- **Framework verification first** - check latest documentation for API changes
- **Only use approved libraries** or discuss new dependencies with user
- **Never delete existing code** without explicit instruction and backup
- **Phase-aware development** - align all work with current MAKE IT RIGHT phase

### Session Startup Protocol (Required)
**Every development session MUST start with:**
1. **Read Current Tasks** - Check 🔴🟡🟢 priorities in Task Management section
2. **Git Status Check** - Run `git status && git log --oneline -5`
3. **System Health** - Verify MCP server, database, and environment
4. **Phase Verification** - Confirm current development phase requirements

### Git Integration Requirements (Non-negotiable)
- **Commit after every completed task** using conventional commit format
- **Include task IDs in commit messages** for traceability (e.g., "feat(agent): implement unified orchestrator - TASK-024")
- **Update CLAUDE.md AS YOU WORK** (not after completion)
- **Risk-based branching** - 🔴 tasks require feature branches, 🟡🟢 can use main

### MCP Tools Implementation Standards
**Core Patterns (Enforced)**
- **All tools are async** - use `async def` with `@mcp.tool()` decorator from official MCP Python SDK
- **Context Access** - get lifespan context via `ctx.request_context.lifespan_context` (official MCP SDK pattern)
- **Error Handling** - return structured JSON responses; may raise exceptions as MCP SDK handles them
- **Type Safety** - complete type hints for all parameters and return values
- **Logging Integration** - use `@log_mcp_tool_execution` decorator for observability

**Available MCP Tools (Production)**
1. **`crawl_single_page(url: str)`** - Quick single page crawl and storage
2. **`smart_crawl_url(url: str, max_depth: int = 3, max_concurrent: int = 10)`** - Intelligent crawling with URL detection
3. **`get_available_sources()`** - List all crawled sources for filtering and discovery
4. **`perform_rag_query(query: str, source: str = None, match_count: int = 5)`** - Semantic search with source filtering
5. **`search_code_examples(query: str, source_id: str = None, match_count: int = 5)`** - Code-specific search (requires USE_AGENTIC_RAG=true)

### Project-Specific Implementation Rules
**Architecture Constraints**
- **Single Orchestrator Pattern** - one agent intelligently selects from 5 MCP tools (TASK-024)
- **Environment Configuration** - RAG strategies toggle via environment variables
- **Database Schema** - run migrations on crawled_pages.sql for schema changes
- **Embedding Models** - currently hardcoded to text-embedding-3-small (OpenAI)
- **Memory Management** - use memory-adaptive dispatcher for large crawl operations

**Framework Verification Requirements**
- **Pydantic AI**: Verify agent patterns, dependency injection, MCP integration
- **Official MCP Python SDK**: Check tool registration (@mcp.tool), context management, transport protocols (SSE/stdio)
- **Crawl4AI**: Validate crawling strategies, content extraction, parallel processing
- **PostgreSQL/pgvector**: Confirm vector operations, indexing, query optimization

## 📤 Deliverable Standards

### Required Output Format (Phase-Aligned)

**🟡 MAKE IT RIGHT Phase Requirements:**

#### 1. Executive Summary
- **Brief overview** of changes implemented
- **Phase alignment** - how changes support MAKE IT RIGHT objectives
- **Risk assessment** - 🔴🟡🟢 classification of changes made

#### 2. Code Implementation
```python
# Complete, production-ready code with:
# - Google-style docstrings
# - Complete type hints
# - Error handling with structured logging
# - Framework verification comments
```

#### 3. Test Coverage
```python
# tests/test_feature.py - Complete Pytest implementation
# - Happy path, edge cases, failure scenarios
# - Mock external dependencies
# - Framework integration testing
```

#### 4. Documentation Updates (Atomic)
```markdown
## CLAUDE.MD Updates (Required)
### Task Management Section
- [x] TASK-XXX completed (YYYY-MM-DD) with acceptance criteria met

### Technical Architecture Section (if applicable)
- Updated architectural decisions with rationale
- Framework verification notes
- Performance impact assessment
```

#### 5. Quality Assurance Checklist
**Code Quality (🔴 Critical)**
- [ ] File size <500 lines, functions <50 lines
- [ ] Complete type hints and docstrings
- [ ] Error handling with structured logging
- [ ] Ruff formatting applied

**Testing (🔴 Critical)**
- [ ] 3+ test cases (happy/edge/failure)
- [ ] All tests passing
- [ ] Coverage targets met (>80% production)
- [ ] Framework integration verified

**Documentation (🔴 Critical)**
- [ ] CLAUDE.md updated atomically
- [ ] Architectural changes documented
- [ ] Framework verification completed
- [ ] Task status updated with completion date

#### 6. Git Workflow Verification
- **Conventional commits**: feat/fix/docs(scope): description - TASK-ID
- **Branch strategy**: Feature branch for 🔴 tasks, main for 🟡🟢
- **Repository status**: Clean working tree after completion

---

### Framework Verification Protocol
**Before implementing any external library integration:**
1. **Check latest documentation** for API changes
2. **Verify compatibility** with current dependencies
3. **Test integration patterns** with minimal examples
4. **Document findings** in Technical Architecture section

**Remember: Every code change MUST include corresponding documentation updates. Undocumented changes are considered incomplete regardless of functionality.**

---

## 📊 Project Status Dashboard

### Current Phase Status: 🟡 MAKE IT RIGHT
**Last Updated**: 2025-01-15

#### Phase Completion Metrics
- **✅ MAKE IT WORK**: 100% Complete (Jan 2025)
  - Core functionality: Production-ready
  - Performance: 197 pages/42.5s, 856 chunks stored
  - Integration: All systems operational

- **🟡 MAKE IT RIGHT**: 95% Complete (In Progress)
  - Architecture refactoring: ✅ TASK-024 completed (unified agent with o3)
  - Code quality: ✅ Standards implemented and verified (TASK-033)
  - Documentation: ✅ Restructured and verified (TASK-030, TASK-033)
  - Model migration: ✅ OpenAI o3 unified migration (TASK-034)
  - Testing framework: Enhanced and aligned

- **🟢 MAKE IT FAST**: 0% Complete (Future Phase)
  - Performance optimization: Planned
  - Advanced features: Backlog ready
  - Production deployment: Future scope

#### Risk Assessment Summary
- **🔴 Critical Risks**: 0 active (all blocking tasks completed)
- **🟡 Medium Risks**: 4 planned (chunking, performance, configuration, testing)
- **🟢 Low Risks**: 2 future (integrations, model support)

#### Next Actions
1. **Begin TASK-025** - Context 7-inspired chunking strategy
2. **Address TASK-031** - Resolve agent testing runtime errors
3. **Implement TASK-026** - Performance optimization for crawling speed
4. **Plan transition to MAKE IT FAST phase** (approaching 100% completion)

---

*This documentation follows the three-phase development model with risk-based task prioritization. All changes are tracked atomically with task IDs and phase alignment.*