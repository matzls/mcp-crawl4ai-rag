"""
OpenAI API mocks for testing.

This module provides mock implementations of OpenAI API calls to avoid
making real API calls during testing.
"""
import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any


class MockEmbeddingResponse:
    """Mock OpenAI embedding response."""
    
    def __init__(self, embeddings: List[List[float]]):
        self.data = [MockEmbeddingData(emb) for emb in embeddings]


class MockEmbeddingData:
    """Mock embedding data object."""
    
    def __init__(self, embedding: List[float]):
        self.embedding = embedding


class MockChatCompletion:
    """Mock OpenAI chat completion response."""
    
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]


class MockChoice:
    """Mock chat completion choice."""
    
    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockMessage:
    """Mock chat completion message."""
    
    def __init__(self, content: str):
        self.content = content
    
    def strip(self):
        return self.content.strip()


@pytest.fixture
def mock_openai_embeddings():
    """Mock OpenAI embeddings.create method."""
    def create_mock_embedding(model: str, input: List[str]) -> MockEmbeddingResponse:
        # Generate mock embeddings based on input text
        embeddings = []
        for text in input:
            # Create deterministic embedding based on text hash
            hash_val = hash(text) % 1000
            embedding = [(hash_val + i) / 1000.0 for i in range(1536)]
            embeddings.append(embedding)
        return MockEmbeddingResponse(embeddings)
    
    with patch('openai.embeddings.create', side_effect=create_mock_embedding):
        yield create_mock_embedding


@pytest.fixture
def mock_openai_chat():
    """Mock OpenAI chat.completions.create method."""
    def create_mock_completion(model: str, messages: List[Dict], **kwargs) -> MockChatCompletion:
        # Generate mock response based on the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # Generate contextual responses based on message content
        if "context" in user_message.lower():
            response = "This code demonstrates authentication and error handling patterns."
        elif "summary" in user_message.lower():
            response = "This documentation covers API usage and best practices."
        elif "code example" in user_message.lower():
            response = "Example function showing proper implementation techniques."
        else:
            response = "Mock response for testing purposes."
        
        return MockChatCompletion(response)
    
    with patch('openai.chat.completions.create', side_effect=create_mock_completion):
        yield create_mock_completion


@pytest.fixture
def mock_openai_api_key():
    """Mock OpenAI API key for testing."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-api-key-12345'}):
        yield 'test-api-key-12345'


@pytest.fixture
def openai_error_scenarios():
    """OpenAI API error scenarios for testing error handling."""
    return {
        "rate_limit": Exception("Rate limit exceeded"),
        "invalid_api_key": Exception("Invalid API key"),
        "service_unavailable": Exception("Service temporarily unavailable"),
        "timeout": Exception("Request timeout"),
        "quota_exceeded": Exception("Quota exceeded")
    }


@pytest.fixture
def mock_openai_with_retries():
    """Mock OpenAI API with retry logic testing."""
    call_count = 0
    
    def embedding_with_failures(model: str, input: List[str]):
        nonlocal call_count
        call_count += 1
        
        # Fail first two attempts, succeed on third
        if call_count <= 2:
            raise Exception("Temporary API error")
        
        # Success on third attempt
        embeddings = []
        for text in input:
            embedding = [0.1] * 1536  # Simple mock embedding
            embeddings.append(embedding)
        return MockEmbeddingResponse(embeddings)
    
    with patch('openai.embeddings.create', side_effect=embedding_with_failures):
        yield embedding_with_failures


@pytest.fixture
def mock_contextual_embedding():
    """Mock contextual embedding generation."""
    responses = {
        "python": "This Python code demonstrates API authentication using requests library.",
        "javascript": "This JavaScript example shows async/await patterns for API calls.", 
        "tutorial": "This tutorial section explains the fundamental concepts and usage patterns.",
        "api": "This API documentation describes endpoint usage and response formats.",
        "default": "This content provides relevant information for the topic."
    }
    
    def mock_generate_context(full_document: str, chunk: str):
        # Determine response based on chunk content
        chunk_lower = chunk.lower()
        for key, response in responses.items():
            if key in chunk_lower:
                return f"{response}\n---\n{chunk}", True
        
        return f"{responses['default']}\n---\n{chunk}", True
    
    return mock_generate_context


@pytest.fixture
def mock_embedding_batch_processor():
    """Mock batch embedding processor for testing."""
    def process_batch(texts: List[str], max_retries: int = 3):
        results = []
        for text in texts:
            # Simulate successful embedding creation
            hash_val = hash(text) % 1000
            embedding = [(hash_val + i) / 1536.0 for i in range(1536)]
            results.append(embedding)
        return results
    
    return process_batch


@pytest.fixture  
def mock_openai_streaming():
    """Mock OpenAI streaming responses for testing."""
    def create_streaming_response(model: str, messages: List[Dict], stream: bool = False):
        if not stream:
            return MockChatCompletion("Non-streaming response")
        
        # Mock streaming chunks
        chunks = [
            {"choices": [{"delta": {"content": "This "}}]},
            {"choices": [{"delta": {"content": "is "}}]},
            {"choices": [{"delta": {"content": "a "}}]},
            {"choices": [{"delta": {"content": "streaming "}}]},
            {"choices": [{"delta": {"content": "response."}}]},
            {"choices": [{"delta": {}}]}  # End marker
        ]
        return chunks
    
    return create_streaming_response


class MockOpenAIClient:
    """Mock OpenAI client for integration testing."""
    
    def __init__(self):
        self.embeddings = MockEmbeddingsAPI()
        self.chat = MockChatAPI()


class MockEmbeddingsAPI:
    """Mock embeddings API."""
    
    def create(self, model: str, input: List[str]):
        embeddings = []
        for text in input:
            # Generate deterministic embedding
            hash_val = abs(hash(text)) % 1000
            embedding = [float(hash_val + i) / 1536.0 for i in range(1536)]
            embeddings.append(embedding)
        return MockEmbeddingResponse(embeddings)


class MockChatAPI:
    """Mock chat completions API."""
    
    def __init__(self):
        self.completions = self
    
    def create(self, model: str, messages: List[Dict], **kwargs):
        # Extract user message
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
        
        # Generate response based on content
        if "summary" in user_content.lower():
            response = "This is a comprehensive summary of the provided content."
        elif "context" in user_content.lower():
            response = "This content provides important context for understanding the topic."
        else:
            response = "This is a mock response for testing purposes."
        
        return MockChatCompletion(response)


@pytest.fixture
def mock_openai_client():
    """Provide mock OpenAI client for testing."""
    return MockOpenAIClient()