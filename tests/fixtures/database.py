"""
Database test fixtures for PostgreSQL testing.

This module provides mock database connections, test data, and database fixtures
for unit and integration testing.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import List, Dict, Any


@pytest.fixture
def mock_postgres_pool():
    """Mock PostgreSQL connection pool for testing."""
    pool = AsyncMock()
    
    # Mock connection context manager
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.fetch = AsyncMock()
    mock_conn.fetchval = AsyncMock()
    
    # Setup pool.acquire() context manager
    pool.acquire = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    
    return pool, mock_conn


@pytest.fixture
def sample_crawled_documents():
    """Sample crawled documents for testing."""
    return [
        {
            "url": "https://example.com/page1",
            "chunk_number": 0,
            "content": "This is the first chunk of content from page 1.",
            "metadata": {"chunk_size": 47, "page_title": "Example Page 1"},
            "source_id": "example.com",
            "embedding": [0.1, 0.2, 0.3] * 512  # Mock 1536-dim embedding
        },
        {
            "url": "https://example.com/page2", 
            "chunk_number": 0,
            "content": "This is content from page 2 with some technical information.",
            "metadata": {"chunk_size": 63, "page_title": "Technical Documentation"},
            "source_id": "example.com",
            "embedding": [0.2, 0.3, 0.4] * 512
        }
    ]


@pytest.fixture 
def sample_code_examples():
    """Sample code examples for testing."""
    return [
        {
            "url": "https://docs.example.com/api",
            "chunk_number": 0,
            "content": """
def hello_world():
    '''A simple hello world function.'''
    print("Hello, World!")
    return "success"

# Usage example
if __name__ == "__main__":
    result = hello_world()
    print(f"Function returned: {result}")
""",
            "summary": "Simple hello world function with usage example",
            "metadata": {
                "language": "python",
                "code_length": 234,
                "has_context": True
            },
            "source_id": "docs.example.com",
            "embedding": [0.3, 0.4, 0.5] * 512
        }
    ]


@pytest.fixture
def mock_search_results():
    """Mock search results for testing."""
    return [
        {
            "url": "https://example.com/result1",
            "content": "First search result with relevant content",
            "similarity": 0.95,
            "source_id": "example.com",
            "metadata": {"content_type": "documentation"}
        },
        {
            "url": "https://example.com/result2", 
            "content": "Second search result with related information",
            "similarity": 0.87,
            "source_id": "example.com",
            "metadata": {"content_type": "tutorial"}
        },
        {
            "url": "https://docs.example.com/result3",
            "content": "Third result from different source",
            "similarity": 0.72,
            "source_id": "docs.example.com", 
            "metadata": {"content_type": "api_documentation"}
        }
    ]


@pytest.fixture
def mock_database_responses():
    """Mock database query responses for testing."""
    return {
        "crawled_pages_count": 150,
        "sources_list": [
            {"source_id": "example.com", "summary": "Example website", "total_word_count": 5000},
            {"source_id": "docs.example.com", "summary": "API documentation", "total_word_count": 12000}
        ],
        "search_results": [
            {
                "url": "https://example.com/page1",
                "content": "Relevant content for search query",
                "similarity": 0.92,
                "source_id": "example.com"
            }
        ]
    }


@pytest.fixture
def postgresql_test_config():
    """Test configuration for PostgreSQL connections."""
    return {
        "host": "localhost",
        "port": 5432,
        "database": "test_crawl4ai_rag",
        "user": "test_user",
        "password": "test_password",
        "min_size": 1,
        "max_size": 2
    }


class MockAsyncPGRecord:
    """Mock asyncpg Record for testing."""
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
    
    def __getitem__(self, key):
        return self._data[key]
    
    def get(self, key, default=None):
        return self._data.get(key, default)
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def items(self):
        return self._data.items()


@pytest.fixture
def mock_asyncpg_records(mock_search_results):
    """Convert mock search results to asyncpg-like records."""
    return [MockAsyncPGRecord(result) for result in mock_search_results]


@pytest.fixture
def database_error_scenarios():
    """Database error scenarios for testing error handling."""
    return {
        "connection_timeout": asyncio.TimeoutError("Connection timeout"),
        "invalid_query": Exception("Invalid SQL query"),
        "constraint_violation": Exception("Unique constraint violation"),
        "pool_exhausted": Exception("Connection pool exhausted")
    }