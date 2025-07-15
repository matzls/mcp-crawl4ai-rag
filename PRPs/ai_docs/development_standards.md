# Development Standards

This document defines the specific technology stack, coding patterns, and implementation approaches for this development framework.

## Technology Stack

### Core Technologies
- **Primary Language**: Python 3.11+
- **Data Validation**: Pydantic v2 for all data models and validation
- **AI Agent Framework**: Pydantic AI for agent development and management
- **Package Management**: UV for fast, reliable dependency management
- **Code Formatting**: Ruff for linting and code formatting
- **Type Checking**: MyPy for static type analysis
- **Testing Framework**: Pytest for all testing needs

### Supporting Tools
- **Environment Management**: UV virtual environments
- **Documentation**: Google-style docstrings with type hints
- **Configuration**: Environment-based configuration with Pydantic models
- **Monitoring**: Logfire for structured logging and tracing (optional)

## Critical Implementation Patterns

### 🤖 Pydantic AI Agent Reuse (Performance Critical)

**REQUIRED**: Agents must be instantiated once as module globals and reused throughout the application lifecycle.

```python
# ✅ CORRECT: Module-global agent instantiation
from pydantic_ai import Agent

document_analyzer = Agent(
    'claude-3-sonnet-20240229',
    system_prompt='You are a skilled document analyst...',
    result_type=str
)

async def analyze_document(content: str) -> str:
    """Analyze document content using the global agent instance."""
    return await document_analyzer.run(content)

# ❌ FORBIDDEN: Creating new agents for each operation
async def bad_analyze_document(content: str) -> str:
    agent = Agent(...)  # Performance killer - creates new agent every call
    return await agent.run(content)
```

**Rationale**: Agent instantiation is expensive. Reusing global instances provides significant performance benefits.

### 🏭 Configuration Factory Pattern

**REQUIRED**: All Pydantic models with configuration dependencies must use factory pattern to avoid hardcoded defaults that defeat environment configuration.

```python
from pydantic import BaseModel, Field
from typing import Protocol

# ❌ FORBIDDEN: Hardcoded defaults
class BadModel(BaseModel):
    timeout: int = Field(30, description="Request timeout")  # Defeats env config
    max_retries: int = Field(3, description="Max retry attempts")

# ✅ CORRECT: Factory-driven configuration
class ConfigProtocol(Protocol):
    DEFAULT_TIMEOUT: int
    MAX_RETRIES: int

class ModelFactory:
    """Factory for creating properly configured models."""
    
    def __init__(self, config: ConfigProtocol):
        self.config = config
    
    def create_model(self, **overrides) -> 'GoodModel':
        """Create model with configuration defaults and optional overrides."""
        return GoodModel(
            timeout=overrides.get('timeout', self.config.DEFAULT_TIMEOUT),
            max_retries=overrides.get('max_retries', self.config.MAX_RETRIES),
            **{k: v for k, v in overrides.items() if k not in ['timeout', 'max_retries']}
        )

class GoodModel(BaseModel):
    timeout: int = Field(description="Request timeout in seconds")
    max_retries: int = Field(description="Maximum retry attempts")
```

**Rationale**: Factory pattern ensures environment configuration is respected while maintaining type safety.

## Code Style & Conventions

### Python Standards
- **PEP 8 Compliance**: Follow PEP 8 style guidelines strictly
- **Type Hints**: Every function parameter and return value must have type hints
- **Line Length**: 88 characters maximum (Black/Ruff default)
- **Import Organization**: Follow isort standards with Ruff configuration

### Data Models & Type Safety
```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
from datetime import datetime

class ExampleModel(BaseModel):
    """Example of proper Pydantic v2 model structure.
    
    All models must use Pydantic v2 patterns with comprehensive validation.
    """
    
    # Required fields with descriptive Field definitions
    name: str = Field(min_length=1, max_length=100, description="Entity name")
    status: Literal["active", "inactive", "pending"] = Field(description="Current status")
    
    # Optional fields with sensible defaults
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")
    
    # Lists with proper typing
    tags: List[str] = Field(default_factory=list, description="Associated tags")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and normalize name field."""
        return v.strip().title()
    
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "frozen": False  # Set to True for immutable models
    }
```

### Documentation Standards
```python
def example_function(param1: str, param2: Optional[int] = None) -> tuple[str, bool]:
    """Brief one-line summary of function purpose.
    
    Longer description if needed, explaining the function's behavior,
    use cases, and any important implementation details.
    
    Args:
        param1: Description of the first parameter and its constraints.
        param2: Description of optional parameter with default behavior.
        
    Returns:
        tuple containing:
            - str: Description of first return value
            - bool: Description of second return value
            
    Raises:
        ValueError: When param1 is empty or contains invalid characters.
        TypeError: When param2 is not an integer.
        
    Example:
        >>> result = example_function("test", 42)
        >>> print(result)
        ("Test", True)
    """
    # Implementation with inline comments for complex logic
    if not param1.strip():  # Reason: Empty strings not allowed
        raise ValueError("param1 cannot be empty")
    
    processed = param1.strip().title()
    success = param2 is None or param2 > 0
    
    return processed, success
```

## Environment & Tooling Setup

### UV Package Management
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # Unix/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies from pyproject.toml
uv sync

# Add new dependencies
uv add requests pydantic

# Add development dependencies
uv add --dev pytest ruff mypy

# Remove packages
uv remove package-name

# Install package in development mode
uv pip install -e .
```

### Development Commands
```bash
# Code Quality (run before every commit)
uv run ruff format .                    # Format code
uv run ruff check .                     # Lint code  
uv run mypy src/                        # Type checking

# Testing
uv run pytest                           # Run all tests
uv run pytest src/module/tests/ -v     # Run specific module tests
uv run pytest --cov=src --cov-report=html  # Coverage report

# Combined quality check
uv run ruff format . && uv run ruff check . && uv run mypy src/ && uv run pytest
```

### Tool Configuration (pyproject.toml)
```toml
[tool.ruff]
line-length = 88
target-version = "py311"
extend-select = ["E", "W", "F", "B", "I", "UP"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["src"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-ra -q --strict-markers"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests"
]
```

## Testing Patterns

### Test Structure & Organization
```python
import pytest
import uuid
from unittest.mock import Mock, patch
from pydantic import ValidationError

from src.module.models import ExampleModel
from src.module.service import ExampleService

class TestExampleModel:
    """Test suite for ExampleModel validation and behavior."""
    
    def test_valid_model_creation(self):
        """Test successful model creation with valid data."""
        # Use uuid4() for unique test data to avoid conflicts
        test_id = str(uuid.uuid4())
        
        model = ExampleModel(
            name=f"test-{test_id}",
            status="active"
        )
        
        assert model.name == f"Test-{test_id}"  # Validates title case conversion
        assert model.status == "active"
        assert isinstance(model.created_at, datetime)
    
    def test_invalid_status_raises_validation_error(self):
        """Test that invalid status values raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ExampleModel(name="test", status="invalid_status")
        
        assert "Input should be 'active', 'inactive' or 'pending'" in str(exc_info.value)
    
    @pytest.mark.parametrize("invalid_name", ["", "   ", "x" * 101])
    def test_name_validation_failures(self, invalid_name: str):
        """Test various name validation failure cases."""
        with pytest.raises(ValidationError):
            ExampleModel(name=invalid_name, status="active")

class TestExampleService:
    """Test suite for ExampleService business logic."""
    
    @pytest.fixture
    def mock_config(self):
        """Provide mock configuration for service testing."""
        config = Mock()
        config.DEFAULT_TIMEOUT = 30
        config.MAX_RETRIES = 3
        return config
    
    @pytest.fixture  
    def service(self, mock_config):
        """Provide configured service instance for testing."""
        return ExampleService(config=mock_config)
    
    async def test_process_data_success(self, service):
        """Test successful data processing workflow."""
        test_data = {"name": "test", "status": "active"}
        
        result = await service.process_data(test_data)
        
        assert result.success is True
        assert result.data.name == "Test"
```

### Test Categories & Markers
```python
# Unit tests - Fast, isolated, no external dependencies
@pytest.mark.unit
def test_pure_function():
    pass

# Integration tests - Test component interactions
@pytest.mark.integration
async def test_database_integration():
    pass

# Slow tests - Can be skipped during development
@pytest.mark.slow
def test_comprehensive_processing():
    pass
```

## Monitoring & Observability (Optional)

### Logfire Integration
```python
import logfire
from pydantic_ai import Agent

# Configure once per application
logfire.configure(
    token=config.logfire_token,
    project_name=config.project_name
)

# Use throughout agent implementations
async def monitored_agent_operation(prompt: str) -> str:
    """Example of agent operation with structured logging."""
    with logfire.span("agent.document_analysis") as span:
        span.set_attribute("prompt.length", len(prompt))
        
        try:
            result = await document_analyzer.run(prompt)
            span.set_attribute("result.length", len(result))
            return result
        except Exception as e:
            span.record_exception(e)
            raise
```

## MCP Server Development (When Applicable)

### Standard MCP Server Structure
```
src/
    agents/          # AI agent implementations
    auth/           # Authentication and authorization
    models/         # Pydantic data models
    services/       # Business logic services
    tools/          # MCP tools implementation
    ui/             # User interface components
    utils/          # Utility functions and helpers
```

### MCP Tool Implementation Pattern
```python
from mcp.types import Tool, TextContent
from pydantic import BaseModel, Field

class ToolInput(BaseModel):
    """Input model for MCP tool with validation."""
    query: str = Field(min_length=1, description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Result limit")

@tool.mcp_tool("search_documents")
async def search_documents(input: ToolInput) -> str:
    """Search documents using validated input parameters.
    
    This tool demonstrates proper MCP tool structure with Pydantic validation.
    """
    # Tool implementation with proper error handling
    try:
        results = await document_service.search(input.query, input.limit)
        return f"Found {len(results)} documents matching '{input.query}'"
    except Exception as e:
        logger.error(f"Document search failed: {e}")
        return f"Search failed: {str(e)}"
```

## Behavioral Guidelines

### Development Practices
- **Never assume or guess**: When uncertain, ask for clarification
- **Verify before using**: Always confirm file paths and module names exist
- **Comment complex logic**: Add `# Reason:` comments explaining why, not just what
- **Maintain documentation**: Keep README.md and CLAUDE.md files current

### Code Quality Standards
- **Type safety first**: Use Pydantic models and type hints comprehensively
- **Error handling**: Provide specific, actionable error messages
- **Performance awareness**: Consider implications of agent instantiation and reuse
- **Testing discipline**: Write tests for all new functionality

### Project Maintenance
- **Dependency management**: Use UV for all package operations
- **Code formatting**: Always use Ruff for consistent formatting
- **Documentation updates**: Update relevant documentation with code changes
- **Pattern consistency**: Follow established patterns within the project

---

**Framework Philosophy**: These standards prioritize type safety, performance, and maintainability while leveraging the strengths of the Python ecosystem and Pydantic AI framework for robust, production-ready applications.