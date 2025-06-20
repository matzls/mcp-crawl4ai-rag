# Crawl4AI RAG MCP Server - Development Guide

## 🎯 Project Overview

### Mission Statement
**Crawl4AI RAG MCP Server**: A production-ready Model Context Protocol server providing intelligent web crawling and RAG capabilities for AI agents and coding assistants, with PostgreSQL vector database integration and Pydantic AI agent orchestration.

### Current Development Phase: 🟡 MAKE IT RIGHT
**Phase Progress: 85%** - Critical gaps identified in file organization and testing infrastructure

**Phase Objectives:**
- [x] Core user journey functional end-to-end
- [x] Clean architecture boundaries established  
- [x] Highest-risk integrations proven and validated
- [x] Core assumptions validated

**Graduation Criteria for MAKE IT FAST:**
- [ ] File size violations resolved (src/utils.py: 866 lines)
- [ ] Comprehensive testing framework implemented (>80% coverage vs. current <2%)
- [ ] All code quality standards met for production
- [ ] Architecture fully supports planned advanced features

## 🏗️ Project Architecture

### Overview
Production-ready MCP server that provides web crawling and RAG capabilities for AI agents. Integrates Crawl4AI for intelligent web scraping with PostgreSQL vector database for semantic search and retrieval. Features unified Pydantic AI agent orchestrator with 5 specialized MCP tools.

### Primary Tech Stack
- **Language**: Python 3.12+ with UV package manager
- **Package Manager**: `uv` for fast dependency management
- **Framework**: Official MCP Python SDK (mcp>=1.9.4) with FastMCP server
- **Database**: PostgreSQL 17 + pgvector extension for vector operations
- **Key Dependencies**: crawl4ai, asyncpg, openai, pydantic-ai[logfire], sentence-transformers

### Key Components
- **MCP Server Core** (`src/mcp_server.py`): FastMCP server with SSE/stdio transport and lifespan management
- **Crawling Tools** (`src/mcp_crawl_tools.py`): Single page and intelligent multi-strategy crawling
- **Search Tools** (`src/mcp_search_tools.py`): Source discovery, RAG queries, and code search
- **Content Processing** (`src/content_processing.py`): Smart chunking and metadata extraction
- **Crawling Strategies** (`src/crawl_strategies.py`): URL detection and parallel processing
- **Unified Agent** (`src/pydantic_agent/unified_agent.py`): Single orchestrator with intelligent tool selection

### Architecture Decisions
| Decision | Rationale | Trade-offs | Status |
|----------|-----------|------------|--------|
| Official MCP Python SDK | Standards compliance, SSE/stdio transport support | Learning curve vs. custom implementation | ✅ |
| Single Orchestrator Agent | Simplified UX, intelligent tool selection | Less granular control vs. 3-agent approach | ✅ |
| PostgreSQL + pgvector | Production-ready vector search, ACID compliance | Setup complexity vs. embedded solutions | ✅ |
| Crawl4AI Integration | JavaScript rendering, clean markdown output | External dependency vs. custom crawler | ✅ |
| OpenAI o3 Model | Advanced reasoning for MCP tool orchestration | Cost vs. local models | ✅ |

### Framework Integration Status
| Framework | Purpose | Last Doc Check | Context7 Verified | Status |
|-----------|---------|----------------|-------------------|--------|
| MCP Python SDK | Server implementation, tool registration | 2025-01-19 | ❓ | ✅ |
| Pydantic AI | Agent orchestration, MCP integration | 2025-01-15 | ❓ | ✅ |
| Crawl4AI | Web content acquisition, HTML→markdown | 2025-01-13 | ❓ | ✅ |
| PostgreSQL/pgvector | Vector storage and semantic search | 2025-01-08 | ❓ | ✅ |
| OpenAI API | Embeddings and LLM processing | 2025-01-15 | ❓ | ✅ |

### High-Risk Integration Points
- **OpenAI API Rate Limits**: Embedding generation and LLM calls - Risk: 🔴 - Validation: Production usage (197 pages/42.5s)
- **PostgreSQL Vector Operations**: Large-scale embedding storage and retrieval - Risk: 🟡 - Validation: 856 chunks stored successfully
- **MCP Protocol Compliance**: SSE transport stability and tool execution - Risk: 🟡 - Validation: All 5 tools verified via MCP Inspector

### Data Flow
1. **URL Input** → Smart detection (sitemap/txt/webpage) via crawling strategies
2. **Crawl4AI Processing** → Web crawling and HTML-to-markdown conversion with parallel processing
3. **Content Processing** → Smart chunking (respects code blocks), metadata extraction
4. **OpenAI Embedding** → text-embedding-3-small with optional contextual enhancement
5. **PostgreSQL Storage** → Vector indices and hybrid search capabilities
6. **RAG Retrieval** → Semantic search with optional reranking and hybrid matching

### External Dependencies
| Dependency | Purpose | Risk Level | Fallback Plan | Status |
|------------|---------|------------|---------------|--------|
| OpenAI API | Embeddings, LLM processing | 🔴 | Local models (future) | ✅ |
| PostgreSQL | Vector database, storage | 🟡 | SQLite fallback | ✅ |
| Crawl4AI Service | Web crawling engine | 🟡 | Direct HTTP requests | ✅ |

## 📋 Current Tasks & Priorities

### 🔴 Active Tasks (Currently Working On)
- [ ] **TASK-044**: Split src/utils.py (866 lines) into focused modules - Framework: PostgreSQL, OpenAI - Risk: 🔴 - Due: 2025-01-20
- [ ] **TASK-044-DOC**: Verify database and embedding framework patterns via Context7
- [ ] **TASK-045**: Implement comprehensive testing framework with >80% coverage - Framework: Pytest - Risk: 🔴 - Due: 2025-01-22
- [ ] **TASK-045-DOC**: Verify Pytest and testing patterns for MCP servers via Context7

### ✅ Recently Completed Tasks
- [x] **TASK-043**: Split src/crawl4ai_mcp.py into 6 modular components (2025-01-19)
- [x] **TASK-049**: Fix unified agent workflow failures and Rich color styling issues (2025-01-19)
- [x] **TASK-048**: Remove legacy agent functions from src/pydantic_agent/agent.py (2025-01-19)

### Phase-Prioritized Backlog

#### 🟡 MAKE IT RIGHT (Current Phase)
**Goal:** Production-grade reliability and maintainability

**🔴 Critical Priority (Do First)**
- [ ] **TASK-046**: Enhance configuration management with centralized settings validation + TASK-046-DOC
- [ ] **TASK-047**: Add custom exception classes and circuit breaker patterns + TASK-047-DOC

**🟡 Important Priority**
- [ ] **TASK-025**: Implement Context7-inspired chunking strategy + TASK-025-DOC
- [ ] **TASK-026**: Performance optimization for crawling speed + TASK-026-DOC

**🟢 Nice-to-Have Priority**
- [ ] **TASK-050**: Enhanced error reporting and debugging tools + TASK-050-DOC

#### 🚀 MAKE IT FAST (Next Phase)
**Goal:** Performance optimization and advanced features
- [ ] **TASK-200**: Performance optimization and monitoring + TASK-200-DOC
- [ ] **TASK-210**: Advanced RAG strategies (hybrid search, reranking) + TASK-210-DOC
- [ ] **TASK-220**: Integration with Archon knowledge engine + TASK-220-DOC
- [ ] **TASK-230**: Multiple embedding model support (Ollama integration) + TASK-230-DOC

### Task Management Protocol
- **Task IDs**: TASK-XXX (sequential numbering, phase-aware ranges)
- **Documentation Tasks**: ALWAYS create TASK-XXX-DOC for framework verification
- **Risk Classification**: 🔴🟡🟢 for priority within phases
- **Context7 Verification**: Required before implementing any framework features
- **Commit References**: Include task ID in all commit messages

## 🔧 Project-Specific Guidelines

### Primary Framework: MCP Python SDK
- **Purpose**: Standards-compliant MCP server implementation with tool registration
- **Documentation**: Always verify via Context7 MCP before implementation
- **Key Topics for Context7**: `tool-registration`, `context-management`, `transport-protocols`
- **Risk Level**: 🟡
- **Common Patterns**: `@mcp.tool()` decorators, lifespan context access, async tool implementation

### Secondary Framework: Pydantic AI
- **Purpose**: Intelligent agent orchestration with MCP tool integration
- **Documentation**: Always verify via Context7 MCP before implementation
- **Key Topics for Context7**: `agent-creation`, `mcp-integration`, `tool-selection`
- **Risk Level**: 🟡
- **Integration Notes**: Single orchestrator pattern with dependency injection

### Framework Documentation Verification Pattern
```python
# TASK-XXX-DOC: Verified [Framework] docs via Context7 - [Date]
# Topics checked: [specific features/patterns used]
# Latest patterns confirmed for [specific use case]
# Implementation following verified current best practices...
```

## 🎯 Phase-Specific Development Standards

### 🟡 MAKE IT RIGHT Phase (Current)
**Focus:** Production-grade reliability and comprehensive testing

**Quality Bar:**
- **Testing**: 3+ cases per feature (happy/edge/failure), >80% coverage
- **Error Handling**: Comprehensive edge cases and graceful degradation
- **Documentation**: Complete API docs with Context7 verification
- **Code Quality**: <500 lines per file, complete type hints, structured logging

**Success Criteria:**
- [ ] All file size violations resolved
- [ ] Comprehensive testing framework implemented
- [ ] Error handling covers all integration points
- [ ] All framework integrations verified via Context7

### 🚀 MAKE IT FAST Phase (Future)
**Focus:** Performance optimization and advanced features

**Quality Bar:**
- **Testing**: Full test suite >90% coverage + performance benchmarks
- **Error Handling**: Production monitoring and alerting
- **Documentation**: Complete user guides and operational runbooks
- **Code Quality**: Optimized, scalable, and fully production-ready

## 🛠️ Environment & Setup

### Development Environment Setup
```bash
# Environment creation
uv venv crawl_venv && source crawl_venv/bin/activate
uv pip install -e .
crawl4ai-setup

# Database setup (PostgreSQL 17 + pgvector)
brew install postgresql@17 pgvector
brew services start postgresql@17
createdb crawl4ai_rag
psql -h localhost -U $(whoami) -d crawl4ai_rag -f crawled_pages.sql
```

### Project-Specific Commands
```bash
# Development workflow
./start_mcp_server.sh                    # Start MCP server (SSE transport)
python cli_chat.py                       # Interactive CLI with unified agent
npx @modelcontextprotocol/inspector      # MCP tools testing

# Quality assurance
ruff format src/ tests/ && ruff check src/ tests/
mypy src/
pytest tests/ -v --cov=src --cov-report=html
```

### Environment Variables Required
```bash
# Core configuration
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql://user:pass@localhost/crawl4ai_rag

# Optional RAG enhancements
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
```

## 📊 Project Status Dashboard

### Current Status: MAKE IT RIGHT - 85% Complete
**Last Updated**: 2025-01-19

**Recent Achievements:**
- Modular architecture refactoring completed (TASK-043 completed 2025-01-19)
- Unified agent workflow issues resolved (TASK-049 completed 2025-01-19)
- Legacy code cleanup finished (TASK-048 completed 2025-01-19)

**Active Blockers:**
- File size violations in src/utils.py (866 lines) preventing proper code review
- Testing infrastructure inadequate (<2% coverage) blocking production readiness

**Next Milestones:**
- Complete file refactoring: Target 2025-01-20 - All files <500 lines
- Implement testing framework: Target 2025-01-22 - >80% coverage achieved

## 📝 Development Log

### Framework Documentation History
- **MCP Python SDK**: Last verified 2025-01-19 via codebase - Topics: tool registration, context management - Status: 🔄
- **Pydantic AI**: Last verified 2025-01-15 via codebase - Topics: agent orchestration, MCP integration - Status: ✅
- **Crawl4AI**: Last verified 2025-01-13 via testing - Topics: web crawling, content extraction - Status: ✅

### Risk Validation Log
- **OpenAI API Integration**: Production validated with 197 pages crawled in 42.5s - High confidence - 2025-01-13
- **PostgreSQL Vector Operations**: 856 chunks stored successfully with semantic search - High confidence - 2025-01-13
- **MCP Protocol Compliance**: All 5 tools verified via MCP Inspector - High confidence - 2025-01-18

### Architecture Evolution
- **2025-01-19**: Modular refactoring - Split 1,121-line monolith into 6 focused modules <500 lines each
- **2025-01-15**: Unified agent architecture - Replaced 3-agent system with single intelligent orchestrator
- **2025-01-13**: Production validation - Achieved performance targets and system stability

### Key Lessons Learned
- **2025-01-19**: File size discipline critical for maintainability - Large files block effective code review
- **2025-01-15**: Single orchestrator pattern improves UX - Users prefer natural language over agent selection
- **2025-01-13**: MCP Inspector invaluable for tool validation - Essential for systematic testing workflow

---

## 🚀 Phase Graduation Checklist

### Ready for MAKE IT FAST Phase?
- [x] All MAKE IT WORK objectives completed
- [ ] Quality bar met for current phase (85% complete)
- [ ] All framework documentation verified via Context7
- [ ] No blocking technical debt or unvalidated risks
- [ ] Architecture proven to support next phase complexity
- [ ] Team confident in core system stability

**Next Phase Preparation:**
- [ ] MAKE IT FAST backlog prioritized and estimated
- [ ] Performance benchmarks and optimization targets set
- [ ] Advanced RAG strategies research completed
- [ ] Integration roadmap for external systems defined

---

**Remember**: This project follows the three-phase development model. Refer to Home CLAUDE.md for universal coding standards and git workflows. All framework usage must be verified via Context7 MCP before implementation.