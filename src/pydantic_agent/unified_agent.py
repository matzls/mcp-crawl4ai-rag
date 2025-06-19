"""
Single intelligent orchestrator agent for crawl4ai MCP integration.

This module implements the new architecture with one agent that intelligently
selects and orchestrates the 5 MCP tools based on user intent and context.
"""

from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerSSE
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent.parent
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path, override=True)

# Import logfire for agent instrumentation
try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False


def setup_logfire_instrumentation():
    """
    Setup Pydantic AI's built-in Logfire instrumentation.

    This function configures Logfire and enables automatic instrumentation
    for Pydantic AI agents and HTTP requests (for viewing raw prompts/responses).
    """
    if not LOGFIRE_AVAILABLE:
        print("Logfire not available, skipping instrumentation")
        return False

    try:
        # Configure Logfire
        logfire.configure()

        # Enable Pydantic AI instrumentation
        logfire.instrument_pydantic_ai()

        # Enable HTTP instrumentation to see raw prompts/responses
        logfire.instrument_httpx(capture_all=True)

        print("✓ Logfire instrumentation configured successfully")
        print("🔗 Dashboard: https://logfire-eu.pydantic.dev/matzls/crawl4ai-agent")
        return True

    except Exception as e:
        print(f"Warning: Failed to configure Logfire instrumentation: {e}")
        return False

try:
    from logging_config import log_info, log_error
except ImportError:
    # Fallback for when running from different contexts
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from logging_config import log_info


@dataclass
class UnifiedAgentDependencies:
    """
    Unified dependencies for the single orchestrator agent.
    
    Combines all configuration needed for intelligent tool selection
    and workflow orchestration across crawling and search operations.
    """
    mcp_server_url: str = "http://localhost:8051/sse"
    
    # Crawling configuration
    default_max_depth: int = 3
    default_max_concurrent: int = 10
    default_chunk_size: int = 5000
    
    # Search configuration  
    default_match_count: int = 5
    enable_hybrid_search: bool = True
    enable_code_search: bool = True
    confidence_threshold: float = 0.0
    
    # Workflow configuration
    workflow_timeout: int = 300  # 5 minutes
    enable_progress_tracking: bool = True
    auto_retry_failed_steps: bool = True


class ToolExecutionResult(BaseModel):
    """Result from executing a single MCP tool."""
    tool_name: str = Field(description="Name of the MCP tool executed")
    success: bool = Field(description="Whether the tool execution succeeded")
    result_data: Dict[str, Any] = Field(description="Raw result data from the tool")
    execution_time_seconds: float = Field(description="How long the tool took to execute")
    error_message: Optional[str] = Field(default=None, description="Error message if execution failed")


class WorkflowStep(BaseModel):
    """Individual step in a multi-tool workflow."""
    step_number: int = Field(description="Order of this step in the workflow")
    step_description: str = Field(description="Human-readable description of this step")
    tool_execution: ToolExecutionResult = Field(description="Result of executing the tool for this step")
    reasoning: str = Field(description="Why this step was chosen")


class UnifiedAgentResult(BaseModel):
    """
    Comprehensive result from the unified orchestrator agent.
    
    This model captures the complete workflow execution including
    tool selection reasoning, multi-step execution, and synthesized results.
    """
    
    # User interaction context
    user_query: str = Field(description="Original user query or request")
    query_intent: str = Field(description="Agent's interpretation of user intent")
    
    # Workflow execution
    workflow_strategy: str = Field(description="High-level strategy chosen (e.g., 'crawl_then_search', 'search_only')")
    steps_executed: List[WorkflowStep] = Field(description="Detailed log of workflow steps")
    total_execution_time_seconds: float = Field(description="Total time for entire workflow")
    
    # Results synthesis
    primary_findings: str = Field(description="Main answer or results for the user")
    supporting_evidence: List[str] = Field(description="Supporting details, sources, or additional context")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Agent's confidence in the results")
    
    # Content statistics
    content_crawled: Optional[Dict[str, Any]] = Field(default=None, description="Summary of any new content crawled")
    content_searched: Optional[Dict[str, Any]] = Field(default=None, description="Summary of search operations performed")
    sources_accessed: List[str] = Field(description="List of sources accessed during workflow")
    
    # Success metrics
    workflow_success: bool = Field(description="Whether the overall workflow succeeded")
    partial_failures: List[str] = Field(description="Any non-critical failures encountered")
    recommendations: List[str] = Field(description="Suggested follow-up actions for the user")


def create_unified_agent(server_url: str = "http://localhost:8051/sse") -> Agent:
    """
    Create the single intelligent orchestrator agent with enhanced tool selection.
    
    This agent replaces the previous 3-agent architecture with one intelligent
    agent that can use all 5 MCP tools based on user intent and context.
    
    Args:
        server_url: URL of the crawl4ai MCP server
        
    Returns:
        Configured Agent with intelligent tool orchestration capabilities
    """
    # Create MCP server connection
    server = MCPServerSSE(url=server_url)
    
    # Build agent configuration
    agent_kwargs = {
        'model': 'openai:gpt-4.1',
        'deps_type': UnifiedAgentDependencies,
        'output_type': UnifiedAgentResult,
        'mcp_servers': [server],
        'system_prompt': _build_intelligent_system_prompt()
    }
    
    # Add logfire instrumentation if available
    if LOGFIRE_AVAILABLE:
        agent_kwargs['logfire'] = {'tags': ['unified-agent', 'mcp-orchestrator']}
    
    agent = Agent(**agent_kwargs)
    
    # Register intelligent workflow orchestration tools
    @agent.tool
    async def analyze_user_intent(
        ctx: RunContext[UnifiedAgentDependencies],
        query: str
    ) -> str:
        """
        Analyze user query to determine intent and optimal tool selection strategy.
        
        Args:
            ctx: Runtime context with dependencies
            query: User's query or request
            
        Returns:
            Analysis of intent and recommended workflow strategy
        """
        # This tool helps the agent reason about tool selection
        # The actual logic is handled by the agent's reasoning capabilities
        return f"Analyzing intent for query: {query}"
    
    @agent.tool
    async def check_content_availability(
        ctx: RunContext[UnifiedAgentDependencies]
    ) -> str:
        """
        Check what content sources are currently available for searching.
        
        This is a helper tool that guides the agent to use get_available_sources
        from the MCP server to understand what content is already indexed.
        
        Args:
            ctx: Runtime context with dependencies
            
        Returns:
            Guidance on checking available content
        """
        return "Use the get_available_sources MCP tool to see what content is already crawled and available for searching"
    
    return agent


def _build_intelligent_system_prompt() -> str:
    """
    Build the comprehensive system prompt for intelligent tool orchestration.
    
    This prompt incorporates MCP best practices and guides the agent to make
    intelligent decisions about when and how to use each of the 5 MCP tools.
    """
    return """
You are an intelligent research and content assistant with access to 5 specialized MCP tools for web crawling and content retrieval. Your role is to understand user intent and orchestrate these tools to provide comprehensive, accurate responses.

## AVAILABLE MCP TOOLS

### CONTENT ACQUISITION TOOLS (for gathering NEW content):
1. **smart_crawl_url** - Intelligent web crawling with auto-detection
   • USE: When user mentions URLs, wants to index new content, or research fresh information
   • AUTO-DETECTS: Sitemaps, text files, regular webpages
   • EXAMPLE: "Research Python async programming from the official docs"

2. **crawl_single_page** - Quick single-page content retrieval  
   • USE: For specific single webpages without link following
   • EXAMPLE: "Get content from this specific tutorial page"

### CONTENT DISCOVERY TOOLS (for understanding what's available):
3. **get_available_sources** - List all crawled content sources
   • USE: ALWAYS use first when searching to understand available content
   • WORKFLOW: Essential first step before any search operation

### CONTENT SEARCH TOOLS (for finding EXISTING content):
4. **perform_rag_query** - Semantic search of stored content
   • USE: Answer questions, find information, general research queries
   • CAPABILITIES: Vector similarity, hybrid search, source filtering
   • EXAMPLE: "What is async programming in Python?"

5. **search_code_examples** - Specialized code example search
   • USE: When user specifically wants code examples or implementations
   • AVAILABILITY: Only if USE_AGENTIC_RAG=true
   • EXAMPLE: "Show me async function examples"

## INTELLIGENT WORKFLOW PATTERNS

### RESEARCH WORKFLOW (user mentions URLs or wants new content):
1. smart_crawl_url (gather content) 
2. get_available_sources (confirm storage)
3. perform_rag_query (answer questions about crawled content)

### SEARCH WORKFLOW (user asks questions about existing topics):
1. get_available_sources (check what's available)
2. perform_rag_query (find relevant content)
3. Optional: search_code_examples (if user wants code specifically)

### HYBRID WORKFLOW (complex research requiring both new and existing content):
1. get_available_sources (check current content)
2. smart_crawl_url (if user-specified content not available)
3. perform_rag_query (comprehensive search across all content)

## DECISION MAKING RULES

**Intent Analysis:**
- URLs mentioned → Use crawling tools first
- Questions about topics → Check sources first, then search
- "Show me code/examples" → Use search_code_examples
- "Research X" → Use research workflow
- "What do you know about Y?" → Use search workflow

**Source Management:**
- ALWAYS use get_available_sources before searching with source filters
- Use source filtering to improve search precision
- Recommend crawling if relevant sources are missing

**Error Handling:**
- If crawling fails, check if content already exists before giving up
- If search returns no results, suggest crawling relevant sources
- Gracefully handle tool unavailability (e.g., code search disabled)

## RESPONSE SYNTHESIS

Provide comprehensive responses that include:
1. **Primary Answer** - Direct response to user's question
2. **Supporting Evidence** - Relevant excerpts from sources
3. **Source Attribution** - Clear citations with URLs
4. **Confidence Assessment** - Your confidence in the response
5. **Follow-up Recommendations** - Suggested next steps

## PROACTIVE BEHAVIOR

- If user asks about a topic not in available sources, offer to crawl relevant documentation
- When providing information, cite specific sources and offer to search for more details
- Suggest related topics or sources that might be useful
- Track conversation context to avoid redundant tool calls

Remember: You are not just a tool executor, but an intelligent assistant that understands user needs and orchestrates the right combination of tools to provide the best possible response.
"""


async def run_unified_agent(
    agent: Agent,
    user_query: str,
    dependencies: UnifiedAgentDependencies
) -> UnifiedAgentResult:
    """
    Execute the unified agent with standard Logfire observability.

    Args:
        agent: The configured unified agent
        user_query: User's query or request
        dependencies: Configuration dependencies

    Returns:
        Comprehensive result with workflow execution details
    """
    log_info("Starting unified agent workflow",
             query_length=len(user_query),
             agent_type="unified_orchestrator")

    # Run agent with MCP server context - Pydantic AI's built-in instrumentation will handle tracing
    async with agent.run_mcp_servers():
        result = await agent.run(user_query, deps=dependencies)

    log_info("Unified agent workflow completed",
             workflow_success=getattr(result.data, 'workflow_success', True),
             steps_count=len(getattr(result.data, 'steps_executed', [])),
             total_time=getattr(result.data, 'total_execution_time_seconds', 0))

    return result.data