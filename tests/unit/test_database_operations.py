"""
Unit tests for database operations module.

Tests all database-related functionality including connection management,
document storage, and search operations.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import json

# Import structure relies on pytest.ini pythonpath = src

from database.operations import (
    get_postgres_connection,
    create_postgres_pool,
    add_documents_to_postgres,
    add_code_examples_to_postgres,
    update_source_info,
    search_documents,
    search_code_examples,
    _embedding_to_vector_string
)


class TestPostgreSQLConnection:
    """Test PostgreSQL connection functions."""
    
    @pytest.mark.asyncio
    async def test_get_postgres_connection_with_database_url(self):
        """Test connection using DATABASE_URL environment variable."""
        with patch.dict('os.environ', {'DATABASE_URL': 'postgresql://user:pass@localhost/test'}):
            with patch('asyncpg.connect') as mock_connect:
                mock_connect.return_value = AsyncMock()
                
                result = await get_postgres_connection()
                
                mock_connect.assert_called_once_with('postgresql://user:pass@localhost/test')
                assert result is not None
    
    @pytest.mark.asyncio
    async def test_get_postgres_connection_with_components(self):
        """Test connection using individual environment variables."""
        env_vars = {
            'POSTGRES_HOST': 'localhost',
            'POSTGRES_PORT': '5432',
            'POSTGRES_DB': 'test_db',
            'POSTGRES_USER': 'test_user',
            'POSTGRES_PASSWORD': 'test_pass'
        }
        
        with patch.dict('os.environ', env_vars):
            with patch('asyncpg.connect') as mock_connect:
                mock_connect.return_value = AsyncMock()
                
                result = await get_postgres_connection()
                
                mock_connect.assert_called_once_with(
                    host='localhost',
                    port=5432,
                    database='test_db',
                    user='test_user',
                    password='test_pass'
                )
    
    @pytest.mark.asyncio
    async def test_get_postgres_connection_missing_password(self):
        """Test connection failure when password is missing."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="Either DATABASE_URL or POSTGRES_PASSWORD must be set"):
                await get_postgres_connection()
    
    @pytest.mark.asyncio
    async def test_create_postgres_pool_with_database_url(self):
        """Test pool creation using DATABASE_URL."""
        with patch.dict('os.environ', {'DATABASE_URL': 'postgresql://user:pass@localhost/test'}):
            with patch('asyncpg.create_pool') as mock_create_pool:
                mock_create_pool.return_value = AsyncMock()
                
                result = await create_postgres_pool()
                
                mock_create_pool.assert_called_once_with(
                    'postgresql://user:pass@localhost/test',
                    min_size=2,
                    max_size=10
                )
    
    @pytest.mark.asyncio 
    async def test_create_postgres_pool_with_whoami_substitution(self):
        """Test pool creation with $(whoami) substitution."""
        with patch.dict('os.environ', {'DATABASE_URL': 'postgresql://$(whoami)@localhost/test'}):
            with patch('getpass.getuser', return_value='testuser'):
                with patch('asyncpg.create_pool') as mock_create_pool:
                    mock_create_pool.return_value = AsyncMock()
                    
                    result = await create_postgres_pool()
                    
                    mock_create_pool.assert_called_once_with(
                        'postgresql://testuser@localhost/test',
                        min_size=2,
                        max_size=10
                    )


class TestDocumentStorage:
    """Test document storage operations."""
    
    @pytest.fixture
    def sample_document_data(self):
        """Sample document data for testing."""
        return {
            "urls": ["https://example.com/page1", "https://example.com/page2"],
            "chunk_numbers": [0, 1],
            "contents": ["First chunk content", "Second chunk content"],
            "metadatas": [{"type": "intro"}, {"type": "details"}],
            "embeddings": [[0.1, 0.2, 0.3] * 512, [0.2, 0.3, 0.4] * 512]
        }
    
    @pytest.mark.asyncio
    async def test_add_documents_to_postgres_success(self, sample_document_data):
        """Test successful document addition."""
        # Setup mocks
        pool = AsyncMock()
        mock_conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Test the function
        await add_documents_to_postgres(
            pool=pool,
            urls=sample_document_data["urls"],
            chunk_numbers=sample_document_data["chunk_numbers"],
            contents=sample_document_data["contents"],
            metadatas=sample_document_data["metadatas"],
            embeddings=sample_document_data["embeddings"]
        )
        
        # Verify deletion and insertion calls
        assert mock_conn.execute.call_count >= 3  # At least delete + 2 inserts
    
    @pytest.mark.asyncio
    async def test_add_documents_to_postgres_with_retry(self, sample_document_data):
        """Test document addition with retry logic."""
        pool = AsyncMock()
        mock_conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # First call fails, second succeeds
        mock_conn.execute.side_effect = [None, Exception("Temporary error"), None, None]
        
        with patch('time.sleep'):  # Speed up the test
            await add_documents_to_postgres(
                pool=pool,
                urls=sample_document_data["urls"][:1],  # Use only one document
                chunk_numbers=sample_document_data["chunk_numbers"][:1],
                contents=sample_document_data["contents"][:1],
                metadatas=sample_document_data["metadatas"][:1],
                embeddings=sample_document_data["embeddings"][:1]
            )
        
        # Should have attempted multiple times
        assert mock_conn.execute.call_count >= 2


class TestCodeExampleStorage:
    """Test code example storage operations."""
    
    @pytest.fixture
    def sample_code_data(self):
        """Sample code example data for testing."""
        return {
            "urls": ["https://docs.example.com/api"],
            "chunk_numbers": [0],
            "code_examples": ["def hello(): return 'world'"],
            "summaries": ["Simple hello function"],
            "metadatas": [{"language": "python"}],
            "embeddings": [[0.3, 0.4, 0.5] * 512]
        }
    
    @pytest.mark.asyncio
    async def test_add_code_examples_to_postgres_success(self, sample_code_data):
        """Test successful code example addition."""
        pool = AsyncMock()
        mock_conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        await add_code_examples_to_postgres(
            pool=pool,
            urls=sample_code_data["urls"],
            chunk_numbers=sample_code_data["chunk_numbers"],
            code_examples=sample_code_data["code_examples"],
            summaries=sample_code_data["summaries"],
            metadatas=sample_code_data["metadatas"],
            embeddings=sample_code_data["embeddings"]
        )
        
        # Verify calls were made
        assert mock_conn.execute.call_count >= 2  # Delete + insert
    
    @pytest.mark.asyncio
    async def test_add_code_examples_empty_list(self):
        """Test handling of empty code examples list."""
        pool = AsyncMock()
        
        await add_code_examples_to_postgres(
            pool=pool,
            urls=[],
            chunk_numbers=[],
            code_examples=[],
            summaries=[],
            metadatas=[],
            embeddings=[]
        )
        
        # Should return early without any database calls
        pool.acquire.assert_not_called()


class TestSourceManagement:
    """Test source information management."""
    
    @pytest.mark.asyncio
    async def test_update_source_info_new_source(self):
        """Test creating a new source."""
        pool = AsyncMock()
        mock_conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock UPDATE returning 0 rows (no existing source)
        mock_conn.execute.side_effect = ["UPDATE 0", None]
        
        await update_source_info(pool, "example.com", "Test summary", 1000)
        
        # Should call UPDATE then INSERT
        assert mock_conn.execute.call_count == 2
    
    @pytest.mark.asyncio
    async def test_update_source_info_existing_source(self):
        """Test updating an existing source."""
        pool = AsyncMock()
        mock_conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock UPDATE returning 1 row (existing source updated)
        mock_conn.execute.return_value = "UPDATE 1"
        
        await update_source_info(pool, "example.com", "Updated summary", 2000)
        
        # Should only call UPDATE
        assert mock_conn.execute.call_count == 1
    
    @pytest.mark.asyncio
    async def test_update_source_info_error_handling(self):
        """Test error handling in source update."""
        pool = AsyncMock()
        mock_conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock database error
        mock_conn.execute.side_effect = Exception("Database error")
        
        # Should not raise exception, just print error
        await update_source_info(pool, "example.com", "Test summary", 1000)


class TestSearchOperations:
    """Test search operations."""
    
    @pytest.mark.asyncio
    async def test_search_documents_success(self):
        """Test successful document search."""
        pool = AsyncMock()
        mock_conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock search results
        mock_records = [
            {"url": "https://example.com", "content": "Test content", "similarity": 0.9}
        ]
        mock_conn.fetch.return_value = mock_records
        
        with patch('embeddings.generator.create_embedding') as mock_create_embedding:
            mock_create_embedding.return_value = [0.1] * 1536
            
            results = await search_documents(pool, "test query", match_count=5)
            
            assert len(results) == 1
            assert results[0]["url"] == "https://example.com"
    
    @pytest.mark.asyncio
    async def test_search_documents_with_filters(self):
        """Test document search with metadata filters."""
        pool = AsyncMock()
        mock_conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        mock_conn.fetch.return_value = []
        
        with patch('embeddings.generator.create_embedding') as mock_create_embedding:
            mock_create_embedding.return_value = [0.1] * 1536
            
            filter_metadata = {"source": "example.com"}
            results = await search_documents(
                pool, "test query", match_count=5, filter_metadata=filter_metadata
            )
            
            # Verify the function was called with filters
            mock_conn.fetch.assert_called_once()
            call_args = mock_conn.fetch.call_args[0]
            assert '"source": "example.com"' in call_args[2]  # filter_json parameter
    
    @pytest.mark.asyncio
    async def test_search_code_examples_success(self):
        """Test successful code example search."""
        pool = AsyncMock()
        mock_conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        mock_records = [
            {"url": "https://docs.example.com", "content": "def test(): pass", "similarity": 0.8}
        ]
        mock_conn.fetch.return_value = mock_records
        
        with patch('embeddings.generator.create_embedding') as mock_create_embedding:
            mock_create_embedding.return_value = [0.1] * 1536
            
            results = await search_code_examples(pool, "python function", match_count=3)
            
            assert len(results) == 1
            assert "def test()" in results[0]["content"]
    
    @pytest.mark.asyncio
    async def test_search_error_handling(self):
        """Test search error handling."""
        pool = AsyncMock()
        mock_conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock database error
        mock_conn.fetch.side_effect = Exception("Database error")
        
        with patch('embeddings.generator.create_embedding') as mock_create_embedding:
            mock_create_embedding.return_value = [0.1] * 1536
            
            results = await search_documents(pool, "test query")
            
            # Should return empty list on error
            assert results == []


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_embedding_to_vector_string(self):
        """Test embedding to vector string conversion."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = _embedding_to_vector_string(embedding)
        
        assert result == "[0.1,0.2,0.3,0.4,0.5]"
    
    def test_embedding_to_vector_string_empty(self):
        """Test empty embedding conversion."""
        embedding = []
        result = _embedding_to_vector_string(embedding)
        
        assert result == "[]"
    
    def test_embedding_to_vector_string_single_value(self):
        """Test single value embedding conversion."""
        embedding = [1.0]
        result = _embedding_to_vector_string(embedding)
        
        assert result == "[1.0]"


class TestBatchOperations:
    """Test batch processing functionality."""
    
    @pytest.mark.asyncio
    async def test_batch_processing_multiple_batches(self):
        """Test processing multiple batches."""
        pool = AsyncMock()
        mock_conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Create data for multiple batches (batch_size=2)
        urls = [f"https://example.com/page{i}" for i in range(5)]
        chunk_numbers = list(range(5))
        contents = [f"Content for page {i}" for i in range(5)]
        metadatas = [{"page": i} for i in range(5)]
        embeddings = [[0.1 * i] * 1536 for i in range(5)]
        
        await add_documents_to_postgres(
            pool=pool,
            urls=urls,
            chunk_numbers=chunk_numbers,
            contents=contents,
            metadatas=metadatas,
            embeddings=embeddings,
            batch_size=2
        )
        
        # Should have processed 3 batches (2+2+1)
        # Each batch involves deletion + insertions
        assert mock_conn.execute.call_count >= 8  # 1 delete + 5 inserts + retries
    
    @pytest.mark.asyncio
    async def test_batch_processing_exact_batch_size(self):
        """Test processing when data exactly matches batch size.""" 
        pool = AsyncMock()
        mock_conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Create data that exactly matches batch size
        batch_size = 3
        urls = [f"https://example.com/page{i}" for i in range(batch_size)]
        chunk_numbers = list(range(batch_size))
        contents = [f"Content {i}" for i in range(batch_size)]
        metadatas = [{"index": i} for i in range(batch_size)]
        embeddings = [[0.1] * 1536 for _ in range(batch_size)]
        
        await add_documents_to_postgres(
            pool=pool,
            urls=urls,
            chunk_numbers=chunk_numbers,
            contents=contents,
            metadatas=metadatas,
            embeddings=embeddings,
            batch_size=batch_size
        )
        
        # Should process exactly one batch
        assert mock_conn.execute.call_count >= 4  # 1 delete + 3 inserts