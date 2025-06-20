"""
Test suite for unified agent functionality.

Tests the basic unified agent creation and model validation without requiring
external API calls or MCP server connections.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pydantic_agent.unified_agent import (
    UnifiedAgentResult,
    WorkflowStep,
    ToolExecutionResult,
    UnifiedAgentDependencies,
    setup_logfire_instrumentation
)


class TestUnifiedAgentModels:
    """Test Pydantic models used by the unified agent."""
    
    def test_tool_execution_result_creation(self):
        """Test creating a ToolExecutionResult model."""
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=True,
            result_data={"pages": 5, "status": "success"},
            execution_time_seconds=1.5
        )
        
        assert result.tool_name == "test_tool"
        assert result.success is True
        assert result.result_data == {"pages": 5, "status": "success"}
        assert result.execution_time_seconds == 1.5
        assert result.error_message is None
    
    def test_tool_execution_result_with_error(self):
        """Test creating a ToolExecutionResult with error."""
        result = ToolExecutionResult(
            tool_name="failing_tool",
            success=False,
            result_data={"error": "timeout"},
            execution_time_seconds=0.5,
            error_message="Connection timeout"
        )
        
        assert result.success is False
        assert result.error_message == "Connection timeout"
    
    def test_workflow_step_creation(self):
        """Test creating a WorkflowStep model."""
        tool_result = ToolExecutionResult(
            tool_name="crawl_tool",
            success=True,
            result_data={"pages_crawled": 5, "status": "complete"},
            execution_time_seconds=2.0
        )
        
        step = WorkflowStep(
            step_number=1,
            step_description="Crawl website for content",
            tool_execution=tool_result,
            reasoning="User requested content from specific URL"
        )
        
        assert step.step_number == 1
        assert step.step_description == "Crawl website for content"
        assert step.tool_execution.tool_name == "crawl_tool"
        assert step.reasoning == "User requested content from specific URL"
    
    def test_unified_agent_result_creation(self):
        """Test creating a complete UnifiedAgentResult."""
        tool_result = ToolExecutionResult(
            tool_name="search_tool",
            success=True,
            result_data={"results_found": 10, "relevance": "high"},
            execution_time_seconds=1.0
        )
        
        step = WorkflowStep(
            step_number=1,
            step_description="Search for content",
            tool_execution=tool_result,
            reasoning="User asked a question"
        )
        
        result = UnifiedAgentResult(
            user_query="What is machine learning?",
            query_intent="Information request about ML concepts",
            workflow_strategy="search_only",
            steps_executed=[step],
            total_execution_time_seconds=1.5,
            primary_findings="Machine learning is a subset of AI...",
            supporting_evidence=["Source 1: Definition", "Source 2: Examples"],
            confidence_score=0.9,
            sources_accessed=["ml-tutorial.com", "ai-basics.org"],
            workflow_success=True,
            partial_failures=[],
            recommendations=["Read more about deep learning", "Try practical examples"]
        )
        
        assert result.user_query == "What is machine learning?"
        assert result.query_intent == "Information request about ML concepts"
        assert result.workflow_strategy == "search_only"
        assert len(result.steps_executed) == 1
        assert result.confidence_score == 0.9
        assert result.workflow_success is True
        assert len(result.recommendations) == 2
    
    def test_unified_agent_result_confidence_validation(self):
        """Test confidence score validation (must be 0.0-1.0)."""
        # Valid confidence scores
        valid_result = UnifiedAgentResult(
            user_query="test",
            query_intent="test",
            workflow_strategy="test",
            steps_executed=[],
            total_execution_time_seconds=0.0,
            primary_findings="test",
            supporting_evidence=[],
            confidence_score=0.85,
            sources_accessed=[],
            workflow_success=True,
            partial_failures=[],
            recommendations=[]
        )
        assert valid_result.confidence_score == 0.85
        
        # Test edge cases
        edge_result = UnifiedAgentResult(
            user_query="test",
            query_intent="test", 
            workflow_strategy="test",
            steps_executed=[],
            total_execution_time_seconds=0.0,
            primary_findings="test",
            supporting_evidence=[],
            confidence_score=0.0,
            sources_accessed=[],
            workflow_success=True,
            partial_failures=[],
            recommendations=[]
        )
        assert edge_result.confidence_score == 0.0
        
        # Test invalid confidence score
        with pytest.raises(ValueError):
            UnifiedAgentResult(
                user_query="test",
                query_intent="test",
                workflow_strategy="test",
                steps_executed=[],
                total_execution_time_seconds=0.0,
                primary_findings="test",
                supporting_evidence=[],
                confidence_score=1.5,  # Invalid: > 1.0
                sources_accessed=[],
                workflow_success=True,
                partial_failures=[],
                recommendations=[]
            )


class TestUnifiedAgentDependencies:
    """Test the dependencies data class."""
    
    def test_dependencies_creation(self):
        """Test creating UnifiedAgentDependencies."""
        deps = UnifiedAgentDependencies()
        
        # Should have default values
        assert hasattr(deps, '__dict__')  # Is a dataclass
    

class TestLogfireInstrumentation:
    """Test Logfire setup functionality."""
    
    @patch('pydantic_agent.unified_agent.LOGFIRE_AVAILABLE', True)
    @patch('pydantic_agent.unified_agent.logfire')
    def test_setup_logfire_instrumentation_success(self, mock_logfire):
        """Test successful Logfire instrumentation setup."""
        result = setup_logfire_instrumentation()
        
        assert result is True
        mock_logfire.configure.assert_called_once()
        mock_logfire.instrument_pydantic_ai.assert_called_once()
        mock_logfire.instrument_httpx.assert_called_once_with(capture_all=True)
    
    @patch('pydantic_agent.unified_agent.LOGFIRE_AVAILABLE', False)
    def test_setup_logfire_instrumentation_unavailable(self):
        """Test Logfire instrumentation when not available."""
        result = setup_logfire_instrumentation()
        
        assert result is False
    
    @patch('pydantic_agent.unified_agent.LOGFIRE_AVAILABLE', True)
    @patch('pydantic_agent.unified_agent.logfire')
    def test_setup_logfire_instrumentation_error(self, mock_logfire):
        """Test Logfire instrumentation setup with error."""
        mock_logfire.configure.side_effect = Exception("Config error")
        
        result = setup_logfire_instrumentation()
        
        assert result is False


class TestUnifiedAgentCreation:
    """Test unified agent creation (without actual MCP server)."""
    
    @patch('pydantic_agent.unified_agent.MCPServerSSE')
    @patch('pydantic_agent.unified_agent.Agent')
    def test_create_unified_agent_basic(self, mock_agent_class, mock_mcp_server):
        """Test basic unified agent creation."""
        from pydantic_agent.unified_agent import create_unified_agent
        
        # Mock the MCP server
        mock_server = MagicMock()
        mock_mcp_server.return_value = mock_server
        
        # Mock the agent
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        # Create agent
        agent = create_unified_agent("http://test:8051/sse")
        
        # Verify MCP server was created correctly
        mock_mcp_server.assert_called_once_with(url="http://test:8051/sse")
        
        # Verify Agent was called with correct parameters
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args[1]
        
        assert call_kwargs['model'] == 'openai:gpt-4.1-mini'
        assert call_kwargs['deps_type'] == UnifiedAgentDependencies
        assert call_kwargs['output_type'] == UnifiedAgentResult
        assert call_kwargs['mcp_servers'] == [mock_server]
        assert 'system_prompt' in call_kwargs
        
        # Should return the mocked agent
        assert agent == mock_agent