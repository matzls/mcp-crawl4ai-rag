"""
Unit tests for embeddings generator module.

Tests all embedding-related functionality including OpenAI API integration,
batch processing, contextual embeddings, and error handling.
"""
import pytest
from unittest.mock import patch, MagicMock
import time

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from embeddings.generator import (
    create_embeddings_batch,
    create_embedding,
    embedding_to_vector_string,
    generate_contextual_embedding,
    process_chunk_with_context,
    process_documents_with_contextual_embeddings
)


class TestEmbeddingCreation:
    """Test embedding creation functionality."""
    
    @patch('openai.embeddings.create')
    def test_create_embeddings_batch_success(self, mock_create):
        """Test successful batch embedding creation."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3] * 512),
            MagicMock(embedding=[0.2, 0.3, 0.4] * 512)
        ]
        mock_create.return_value = mock_response
        
        texts = ["First text", "Second text"]
        result = create_embeddings_batch(texts)
        
        assert len(result) == 2
        assert len(result[0]) == 1536  # Standard embedding dimension
        assert len(result[1]) == 1536
        mock_create.assert_called_once_with(
            model="text-embedding-3-small",
            input=texts
        )
    
    @patch('openai.embeddings.create')
    def test_create_embeddings_batch_empty_input(self, mock_create):
        """Test batch embedding with empty input."""
        result = create_embeddings_batch([])
        
        assert result == []
        mock_create.assert_not_called()
    
    @patch('openai.embeddings.create')
    @patch('time.sleep')  # Speed up retry tests
    def test_create_embeddings_batch_with_retry(self, mock_sleep, mock_create):
        """Test batch embedding with retry logic."""
        # First two calls fail, third succeeds
        mock_create.side_effect = [
            Exception("Rate limit exceeded"),
            Exception("Service unavailable"), 
            MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])
        ]
        
        texts = ["Test text"]
        result = create_embeddings_batch(texts)
        
        assert len(result) == 1
        assert len(result[0]) == 1536
        assert mock_create.call_count == 3
        assert mock_sleep.call_count == 2
    
    @patch('openai.embeddings.create')
    @patch('time.sleep')
    def test_create_embeddings_batch_fallback_to_individual(self, mock_sleep, mock_create):
        """Test fallback to individual embedding creation."""
        # Batch calls fail, individual calls succeed
        def side_effect(*args, **kwargs):
            if isinstance(kwargs['input'], list) and len(kwargs['input']) > 1:
                raise Exception("Batch failed")
            else:
                return MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])
        
        mock_create.side_effect = side_effect
        
        texts = ["First text", "Second text"]
        result = create_embeddings_batch(texts)
        
        assert len(result) == 2
        # Should be called multiple times (failed batch + individual calls)
        assert mock_create.call_count >= 2
    
    @patch('embeddings.generator.create_embeddings_batch')
    def test_create_embedding_single_text(self, mock_batch):
        """Test single embedding creation."""
        mock_batch.return_value = [[0.1, 0.2, 0.3] * 512]
        
        result = create_embedding("Test text")
        
        assert len(result) == 1536
        mock_batch.assert_called_once_with(["Test text"])
    
    @patch('embeddings.generator.create_embeddings_batch')
    def test_create_embedding_error_fallback(self, mock_batch):
        """Test single embedding error fallback."""
        mock_batch.side_effect = Exception("API error")
        
        result = create_embedding("Test text")
        
        # Should return zero embedding on error
        assert result == [0.0] * 1536


class TestVectorConversion:
    """Test vector string conversion functionality."""
    
    def test_embedding_to_vector_string_normal(self):
        """Test normal embedding to vector string conversion."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = embedding_to_vector_string(embedding)
        
        assert result == "[0.1,0.2,0.3,0.4,0.5]"
    
    def test_embedding_to_vector_string_empty(self):
        """Test empty embedding conversion."""
        embedding = []
        result = embedding_to_vector_string(embedding)
        
        assert result == "[]"
    
    def test_embedding_to_vector_string_negative_values(self):
        """Test embedding with negative values."""
        embedding = [-0.1, 0.0, 0.2, -0.3]
        result = embedding_to_vector_string(embedding)
        
        assert result == "[-0.1,0.0,0.2,-0.3]"
    
    def test_embedding_to_vector_string_large_values(self):
        """Test embedding with large values."""
        embedding = [1000.0, -500.5, 0.123456789]
        result = embedding_to_vector_string(embedding)
        
        assert result == "[1000.0,-500.5,0.123456789]"


class TestContextualEmbedding:
    """Test contextual embedding functionality."""
    
    @patch('openai.chat.completions.create')
    def test_generate_contextual_embedding_success(self, mock_create):
        """Test successful contextual embedding generation."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="This code demonstrates authentication"))
        ]
        mock_create.return_value = mock_response
        
        full_document = "# API Documentation\n\nThis guide covers authentication..."
        chunk = "def authenticate(key): return validate(key)"
        
        with patch.dict('os.environ', {'MODEL_CHOICE': 'gpt-4'}):
            result, success = generate_contextual_embedding(full_document, chunk)
        
        assert success is True
        assert "This code demonstrates authentication" in result
        assert chunk in result
        assert "---" in result  # Context separator
    
    @patch('openai.chat.completions.create')
    def test_generate_contextual_embedding_api_error(self, mock_create):
        """Test contextual embedding with API error."""
        mock_create.side_effect = Exception("API error")
        
        full_document = "Documentation content"
        chunk = "Code chunk"
        
        with patch.dict('os.environ', {'MODEL_CHOICE': 'gpt-4'}):
            result, success = generate_contextual_embedding(full_document, chunk)
        
        assert success is False
        assert result == chunk  # Should return original chunk on error
    
    @patch('openai.chat.completions.create')
    def test_generate_contextual_embedding_long_document(self, mock_create):
        """Test contextual embedding with long document truncation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Context for long doc"))]
        mock_create.return_value = mock_response
        
        # Create document longer than 25000 characters
        full_document = "A" * 30000
        chunk = "def test(): pass"
        
        with patch.dict('os.environ', {'MODEL_CHOICE': 'gpt-4'}):
            result, success = generate_contextual_embedding(full_document, chunk)
        
        assert success is True
        # Check that API was called with truncated document
        call_args = mock_create.call_args[1]['messages']
        user_message = call_args[1]['content']
        # Document should be truncated to 25000 chars
        assert len(user_message) < 30000
    
    def test_process_chunk_with_context(self):
        """Test chunk processing with context wrapper."""
        with patch('embeddings.generator.generate_contextual_embedding') as mock_generate:
            mock_generate.return_value = ("contextualized chunk", True)
            
            args = ("https://example.com", "chunk content", "full document")
            result = process_chunk_with_context(args)
            
            assert result == ("contextualized chunk", True)
            mock_generate.assert_called_once_with("full document", "chunk content")


class TestBatchContextualProcessing:
    """Test batch contextual processing functionality."""
    
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_process_documents_with_contextual_embeddings_enabled(self, mock_executor):
        """Test batch processing with contextual embeddings enabled."""
        # Mock executor behavior
        mock_future = MagicMock()
        mock_future.result.return_value = ("contextualized content", True)
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.submit.return_value = mock_future
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.__exit__.return_value = None
        mock_executor.return_value = mock_executor_instance
        
        # Mock as_completed to return futures immediately
        with patch('concurrent.futures.as_completed') as mock_as_completed:
            mock_as_completed.return_value = [mock_future]
            
            urls = ["https://example.com"]
            contents = ["original content"]
            metadatas = [{"type": "test"}]
            url_to_full_document = {"https://example.com": "full document"}
            
            with patch.dict('os.environ', {'USE_CONTEXTUAL_EMBEDDINGS': 'true'}):
                result_contents, result_metadatas = process_documents_with_contextual_embeddings(
                    urls, contents, metadatas, url_to_full_document
                )
            
            assert len(result_contents) == 1
            assert result_metadatas[0]["contextual_embedding"] is True
    
    def test_process_documents_with_contextual_embeddings_disabled(self):
        """Test batch processing with contextual embeddings disabled."""
        urls = ["https://example.com"]
        contents = ["original content"]
        metadatas = [{"type": "test"}]
        url_to_full_document = {"https://example.com": "full document"}
        
        with patch.dict('os.environ', {'USE_CONTEXTUAL_EMBEDDINGS': 'false'}):
            result_contents, result_metadatas = process_documents_with_contextual_embeddings(
                urls, contents, metadatas, url_to_full_document
            )
        
        # Should return original data unchanged
        assert result_contents == contents
        assert result_metadatas == metadatas
    
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_process_documents_with_partial_failures(self, mock_executor):
        """Test batch processing with some failures."""
        # Mock some successes and some failures
        futures = []
        for i in range(3):
            mock_future = MagicMock()
            if i == 1:  # Second future fails
                mock_future.result.side_effect = Exception("Processing error")
            else:
                mock_future.result.return_value = (f"contextualized content {i}", True)
            futures.append(mock_future)
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.submit.side_effect = futures
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.__exit__.return_value = None
        mock_executor.return_value = mock_executor_instance
        
        with patch('concurrent.futures.as_completed') as mock_as_completed:
            mock_as_completed.return_value = futures
            
            urls = [f"https://example.com/page{i}" for i in range(3)]
            contents = [f"original content {i}" for i in range(3)]
            metadatas = [{"type": "test"} for _ in range(3)]
            url_to_full_document = {url: f"full document {i}" for i, url in enumerate(urls)}
            
            with patch.dict('os.environ', {'USE_CONTEXTUAL_EMBEDDINGS': 'true'}):
                result_contents, result_metadatas = process_documents_with_contextual_embeddings(
                    urls, contents, metadatas, url_to_full_document
                )
            
            # Should handle failure gracefully
            assert len(result_contents) == 3
            # Failed item should fall back to original content
            assert "original content 1" in result_contents


class TestErrorHandling:
    """Test error handling in embeddings generation."""
    
    @patch('openai.embeddings.create')
    @patch('time.sleep')
    def test_rate_limit_handling(self, mock_sleep, mock_create):
        """Test handling of rate limit errors."""
        # Simulate rate limit then success
        mock_create.side_effect = [
            Exception("Rate limit exceeded"),
            MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])
        ]
        
        result = create_embeddings_batch(["test"])
        
        assert len(result) == 1
        assert mock_create.call_count == 2
        assert mock_sleep.call_count == 1
    
    @patch('openai.embeddings.create')
    @patch('time.sleep')
    def test_max_retries_exceeded(self, mock_sleep, mock_create):
        """Test behavior when max retries are exceeded."""
        # Always fail
        mock_create.side_effect = Exception("Persistent API error")
        
        result = create_embeddings_batch(["test"])
        
        # Should fall back to individual processing, then zero embeddings
        assert len(result) == 1
        assert result[0] == [0.0] * 1536
        assert mock_create.call_count >= 3  # Batch attempts + individual attempts
    
    @patch('openai.chat.completions.create')
    def test_contextual_embedding_timeout(self, mock_create):
        """Test contextual embedding with timeout."""
        mock_create.side_effect = Exception("Request timeout")
        
        with patch.dict('os.environ', {'MODEL_CHOICE': 'gpt-4'}):
            result, success = generate_contextual_embedding("doc", "chunk")
        
        assert success is False
        assert result == "chunk"


class TestPerformance:
    """Test performance-related functionality."""
    
    @patch('openai.embeddings.create')
    def test_batch_size_efficiency(self, mock_create):
        """Test that batch processing is more efficient than individual calls."""
        mock_create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536) for _ in range(10)]
        )
        
        texts = [f"Text {i}" for i in range(10)]
        
        start_time = time.time()
        result = create_embeddings_batch(texts)
        end_time = time.time()
        
        assert len(result) == 10
        assert mock_create.call_count == 1  # Single batch call
        
        # Verify all embeddings are correct size
        for embedding in result:
            assert len(embedding) == 1536
    
    def test_vector_string_conversion_performance(self):
        """Test vector string conversion performance."""
        embedding = [0.1] * 1536
        
        start_time = time.time()
        for _ in range(1000):
            result = embedding_to_vector_string(embedding)
        end_time = time.time()
        
        # Should be very fast
        assert end_time - start_time < 1.0
        assert result.startswith("[0.1,")
        assert result.endswith(",0.1]")


class TestEnvironmentConfiguration:
    """Test environment configuration handling."""
    
    def test_openai_api_key_loading(self):
        """Test OpenAI API key loading from environment."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key-123'}):
            # Import should load the API key
            import openai
            # Note: This tests the module loading behavior
    
    def test_model_choice_configuration(self):
        """Test model choice configuration."""
        with patch.dict('os.environ', {'MODEL_CHOICE': 'gpt-4-turbo'}):
            with patch('openai.chat.completions.create') as mock_create:
                mock_create.return_value = MagicMock(
                    choices=[MagicMock(message=MagicMock(content="test response"))]
                )
                
                generate_contextual_embedding("doc", "chunk")
                
                # Verify correct model was used
                mock_create.assert_called_once()
                call_kwargs = mock_create.call_args[1]
                assert call_kwargs['model'] == 'gpt-4-turbo'
    
    def test_contextual_embeddings_toggle(self):
        """Test contextual embeddings environment toggle."""
        urls = ["https://example.com"]
        contents = ["content"]
        metadatas = [{}]
        url_to_full_document = {"https://example.com": "full doc"}
        
        # Test enabled
        with patch.dict('os.environ', {'USE_CONTEXTUAL_EMBEDDINGS': 'true'}):
            with patch('concurrent.futures.ThreadPoolExecutor'):
                result_contents, _ = process_documents_with_contextual_embeddings(
                    urls, contents, metadatas, url_to_full_document
                )
        
        # Test disabled  
        with patch.dict('os.environ', {'USE_CONTEXTUAL_EMBEDDINGS': 'false'}):
            result_contents_disabled, _ = process_documents_with_contextual_embeddings(
                urls, contents, metadatas, url_to_full_document
            )
        
        # Disabled should return original content unchanged
        assert result_contents_disabled == contents