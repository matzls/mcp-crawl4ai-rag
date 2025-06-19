"""
Mock data generators for testing.

This module provides realistic test data for various components of the system
including markdown content, embeddings, and API responses.
"""
import pytest
import random
from typing import List, Dict, Any


@pytest.fixture
def sample_markdown_content():
    """Sample markdown content with code blocks for testing."""
    return """
# API Documentation

This is a comprehensive guide to our API.

## Getting Started

To get started with our API, you need to authenticate first.

```python
import requests

def authenticate(api_key):
    '''Authenticate with the API using your API key.'''
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    response = requests.get('https://api.example.com/auth', headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f'Authentication failed: {response.status_code}')

# Example usage
api_key = "your-api-key-here"
auth_result = authenticate(api_key)
print(f"Authentication successful: {auth_result}")
```

## Making API Calls

Once authenticated, you can make API calls:

```javascript
// JavaScript example for making API calls
async function fetchUserData(userId) {
    const response = await fetch(`https://api.example.com/users/${userId}`, {
        method: 'GET',
        headers: {
            'Authorization': 'Bearer ' + localStorage.getItem('token'),
            'Content-Type': 'application/json'
        }
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const userData = await response.json();
    return userData;
}

// Usage
fetchUserData(123)
    .then(user => console.log('User data:', user))
    .catch(error => console.error('Error:', error));
```

## Error Handling

Always implement proper error handling in your applications.

## Rate Limiting

Our API has rate limits to ensure fair usage.
"""


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    def generate_embedding(dimension: int = 1536) -> List[float]:
        return [random.uniform(-1, 1) for _ in range(dimension)]
    
    return {
        "single_embedding": generate_embedding(),
        "batch_embeddings": [generate_embedding() for _ in range(5)],
        "zero_embedding": [0.0] * 1536,
        "normalized_embedding": [x / 1536 for x in range(1536)]
    }


@pytest.fixture
def sample_urls():
    """Sample URLs for testing crawling functionality."""
    return [
        "https://docs.python.org/3/tutorial/",
        "https://github.com/microsoft/vscode/blob/main/README.md",
        "https://fastapi.tiangolo.com/tutorial/",
        "https://example.com/api/docs",
        "https://stackoverflow.com/questions/tagged/python",
        "https://realpython.com/python-requests/",
        "https://httpbin.org/html",
        "https://httpbin.org/json"
    ]


@pytest.fixture
def sample_chunked_content():
    """Sample chunked content for testing."""
    return [
        {
            "chunk_number": 0,
            "content": "This is the first chunk of content. It contains introductory information about the topic.",
            "metadata": {"chunk_size": 89, "section": "introduction"}
        },
        {
            "chunk_number": 1, 
            "content": "This is the second chunk with more detailed information. It covers implementation details and examples.",
            "metadata": {"chunk_size": 104, "section": "implementation"}
        },
        {
            "chunk_number": 2,
            "content": "The final chunk contains conclusion and references. It summarizes the key points.",
            "metadata": {"chunk_size": 87, "section": "conclusion"}
        }
    ]


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return [
        {
            "page_title": "Getting Started Guide",
            "author": "Documentation Team",
            "last_updated": "2024-01-15",
            "content_type": "tutorial",
            "difficulty": "beginner"
        },
        {
            "page_title": "API Reference",
            "author": "Engineering Team", 
            "last_updated": "2024-01-18",
            "content_type": "api_documentation",
            "version": "v2.1"
        },
        {
            "page_title": "Advanced Topics",
            "author": "Senior Engineers",
            "last_updated": "2024-01-20", 
            "content_type": "documentation",
            "difficulty": "advanced"
        }
    ]


@pytest.fixture
def mcp_context_mock():
    """Mock MCP context for testing MCP tools."""
    class MockContext:
        def __init__(self):
            self.request_context = MockRequestContext()
    
    class MockRequestContext:
        def __init__(self):
            self.lifespan_context = MockLifespanContext()
    
    class MockLifespanContext:
        def __init__(self):
            from unittest.mock import AsyncMock
            self.crawler = AsyncMock()
            self.postgres_pool = AsyncMock()
            self.reranking_model = None
    
    return MockContext()


@pytest.fixture
def openai_api_responses():
    """Mock OpenAI API responses for testing."""
    return {
        "embedding_response": {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3] * 512,  # 1536 dimensions
                    "index": 0
                }
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        },
        "chat_completion_response": {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1640995200,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response from the chat completion API."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 12, "total_tokens": 27}
        }
    }


@pytest.fixture
def crawl4ai_responses():
    """Mock Crawl4AI responses for testing."""
    return {
        "successful_crawl": {
            "url": "https://example.com",
            "html": "<html><body><h1>Example</h1><p>Content</p></body></html>",
            "cleaned_html": "<h1>Example</h1><p>Content</p>", 
            "markdown": "# Example\n\nContent",
            "extracted_content": "Example\n\nContent",
            "success": True,
            "status_code": 200,
            "links": {
                "internal": ["https://example.com/about"],
                "external": ["https://external.com"]
            },
            "media": {
                "images": [{"src": "https://example.com/image.jpg", "alt": "Example image"}]
            }
        },
        "failed_crawl": {
            "url": "https://nonexistent.com",
            "success": False,
            "status_code": 404,
            "error": "Page not found"
        }
    }


@pytest.fixture 
def search_query_examples():
    """Sample search queries for testing."""
    return [
        "How to authenticate with API",
        "Python code examples",
        "Error handling best practices", 
        "Rate limiting implementation",
        "Database connection setup",
        "async function tutorial",
        "REST API design patterns",
        "Unit testing with pytest"
    ]


@pytest.fixture
def performance_benchmarks():
    """Performance benchmarks for testing."""
    return {
        "embedding_creation": {
            "single_text": {"max_time": 2.0, "avg_time": 0.5},
            "batch_10": {"max_time": 5.0, "avg_time": 2.0},
            "batch_50": {"max_time": 15.0, "avg_time": 8.0}
        },
        "database_operations": {
            "insert_single": {"max_time": 1.0, "avg_time": 0.2},
            "insert_batch": {"max_time": 10.0, "avg_time": 3.0},
            "search_query": {"max_time": 3.0, "avg_time": 1.0}
        },
        "crawling": {
            "single_page": {"max_time": 10.0, "avg_time": 3.0},
            "sitemap_crawl": {"max_time": 60.0, "avg_time": 30.0}
        }
    }