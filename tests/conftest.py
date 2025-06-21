"""
Pytest configuration and global fixtures for testing.

This module provides centralized mock configuration and fixtures to ensure
consistent test behavior across all test modules.
"""
import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Generator, Any


@pytest.fixture(autouse=True)
def mock_environment():
    """Automatically mock environment variables for all tests."""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-api-key',
        'MODEL_CHOICE': 'gpt-4',
        'DATABASE_URL': 'postgresql://test:test@localhost:5432/test_db',
        'POSTGRES_HOST': 'localhost',
        'POSTGRES_PORT': '5432',
        'POSTGRES_DB': 'test_db',
        'POSTGRES_USER': 'test_user',
        'POSTGRES_PASSWORD': 'test_password',
        'USE_CONTEXTUAL_EMBEDDINGS': 'false',
        'USE_HYBRID_SEARCH': 'false',
        'USE_AGENTIC_RAG': 'false'
    }, clear=False):
        yield


@pytest.fixture(autouse=True)
def mock_openai_client():
    """Automatically mock OpenAI client for all tests."""
    # Mock the OpenAI module-level client
    with patch('openai.chat.completions.create') as mock_chat, \
         patch('openai.embeddings.create') as mock_embeddings:
        
        # Setup default chat completion mock
        mock_chat_response = MagicMock()
        mock_chat_response.choices = [
            MagicMock(message=MagicMock(content="Mocked response"))
        ]
        mock_chat.return_value = mock_chat_response
        
        # Setup default embedding mock
        mock_embed_response = MagicMock()
        mock_embed_response.data = [
            MagicMock(embedding=[0.1] * 1536)
        ]
        mock_embeddings.return_value = mock_embed_response
        
        yield {
            'chat': mock_chat,
            'embeddings': mock_embeddings
        }


@pytest.fixture(autouse=True)
def mock_asyncpg():
    """Automatically mock asyncpg database connections for all tests."""
    with patch('asyncpg.connect') as mock_connect, \
         patch('asyncpg.create_pool') as mock_create_pool:
        
        # Setup default connection mock
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.fetchval = AsyncMock(return_value=None)
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_conn.close = AsyncMock()
        
        # Make connect return an async mock
        mock_connect.return_value = mock_conn
        
        # Setup default pool mock
        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_pool.close = AsyncMock()
        
        # Make create_pool async
        async def mock_create_pool_func(*args, **kwargs):
            return mock_pool
        mock_create_pool.side_effect = mock_create_pool_func
        
        yield {
            'connect': mock_connect,
            'create_pool': mock_create_pool,
            'connection': mock_conn,
            'pool': mock_pool
        }


@pytest.fixture
def sample_embedding():
    """Provide a sample embedding vector for testing."""
    return [0.1, 0.2, 0.3] * 512  # 1536 dimensions


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        {
            'id': 'doc1',
            'source_id': 'test_source',
            'content': 'This is a test document about Python programming.',
            'url': 'https://example.com/doc1',
            'chunk_index': 0,
            'similarity': 0.95
        },
        {
            'id': 'doc2', 
            'source_id': 'test_source',
            'content': 'Another test document about web development.',
            'url': 'https://example.com/doc2',
            'chunk_index': 1,
            'similarity': 0.87
        }
    ]


@pytest.fixture
def sample_code_examples():
    """Provide sample code examples for testing."""
    return [
        {
            'id': 'code1',
            'source_id': 'test_source',
            'language': 'python',
            'code': 'def hello_world():\n    print("Hello, World!")',
            'summary': 'A simple hello world function',
            'context_before': 'Here is a basic example:',
            'context_after': 'This demonstrates basic Python syntax.',
            'similarity': 0.92
        }
    ]