"""
Unit tests for content processor module.

Tests all content processing functionality including code extraction,
summarization, and content analysis.
"""
import pytest
from unittest.mock import patch, MagicMock

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from content.processor import (
    extract_code_blocks,
    generate_code_example_summary,
    extract_source_summary,
    process_code_examples_batch,
    analyze_content_structure
)


class TestCodeExtraction:
    """Test code block extraction functionality."""
    
    def test_extract_code_blocks_python(self):
        """Test extraction of Python code blocks."""
        markdown = """
# Python Tutorial

Here's a simple function:

```python
def hello_world():
    '''A greeting function.'''
    print("Hello, World!")
    return "success"

# Usage example
if __name__ == "__main__":
    result = hello_world()
    print(f"Result: {result}")
```

That's how you write a basic function.
"""
        
        blocks = extract_code_blocks(markdown, min_length=50)
        
        assert len(blocks) == 1
        assert blocks[0]['language'] == 'python'
        assert 'def hello_world()' in blocks[0]['code']
        assert 'print("Hello, World!")' in blocks[0]['code']
        assert 'Here\'s a simple function:' in blocks[0]['context_before']
        assert 'That\'s how you write' in blocks[0]['context_after']
    
    def test_extract_code_blocks_javascript(self):
        """Test extraction of JavaScript code blocks."""
        markdown = """
# JavaScript Guide

Async function example:

```javascript
async function fetchData(url) {
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching data:', error);
        throw error;
    }
}

// Usage
fetchData('https://api.example.com/data')
    .then(data => console.log(data))
    .catch(error => console.error(error));
```

This demonstrates async/await patterns.
"""
        
        blocks = extract_code_blocks(markdown, min_length=100)
        
        assert len(blocks) == 1
        assert blocks[0]['language'] == 'javascript'
        assert 'async function fetchData' in blocks[0]['code']
        assert 'await fetch(url)' in blocks[0]['code']
    
    def test_extract_code_blocks_no_language(self):
        """Test extraction of code blocks without language specification."""
        markdown = """
# Generic Code

```
function generic() {
    console.log("No language specified");
    return true;
}

var result = generic();
```
"""
        
        blocks = extract_code_blocks(markdown, min_length=20)
        
        assert len(blocks) == 1
        assert blocks[0]['language'] == ''
        assert 'function generic()' in blocks[0]['code']
    
    def test_extract_code_blocks_multiple_blocks(self):
        """Test extraction of multiple code blocks."""
        markdown = """
# Multiple Examples

First example:

```python
def first_function():
    return "first"
```

Second example:

```javascript
function secondFunction() {
    return "second";
}
```

Third example:

```bash
#!/bin/bash
echo "Third example"
```
"""
        
        blocks = extract_code_blocks(markdown, min_length=20)
        
        assert len(blocks) == 3
        languages = [block['language'] for block in blocks]
        assert 'python' in languages
        assert 'javascript' in languages
        assert 'bash' in languages
    
    def test_extract_code_blocks_min_length_filter(self):
        """Test minimum length filtering."""
        markdown = """
# Short Code

```python
x = 1
```

```python
def long_function():
    '''This is a longer function that should be included.'''
    for i in range(10):
        print(f"Processing item {i}")
        if i % 2 == 0:
            print("Even number")
        else:
            print("Odd number")
    return "completed"
```
"""
        
        # High minimum length should filter out short code
        blocks = extract_code_blocks(markdown, min_length=100)
        
        assert len(blocks) == 1
        assert 'def long_function()' in blocks[0]['code']
    
    def test_extract_code_blocks_skip_initial_backticks(self):
        """Test skipping initial backticks when content starts with them."""
        markdown = """```
# This content starts with backticks

```python
def example():
    return "test"
```

Normal content after.
```"""
        
        blocks = extract_code_blocks(markdown, min_length=10)
        
        assert len(blocks) == 1
        assert blocks[0]['language'] == 'python'
        assert 'def example()' in blocks[0]['code']
    
    def test_extract_code_blocks_empty_markdown(self):
        """Test extraction from empty markdown."""
        blocks = extract_code_blocks("", min_length=1)
        assert len(blocks) == 0
    
    def test_extract_code_blocks_no_code(self):
        """Test extraction from markdown with no code blocks."""
        markdown = """
# Just Text

This is regular markdown content without any code blocks.
It has paragraphs and headers but no code.

## Another Section

More text content here.
"""
        
        blocks = extract_code_blocks(markdown, min_length=1)
        assert len(blocks) == 0


class TestCodeSummarization:
    """Test code example summarization functionality."""
    
    @patch('openai.chat.completions.create')
    def test_generate_code_example_summary_success(self, mock_create):
        """Test successful code summary generation."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="This function demonstrates API authentication using requests."))
        ]
        mock_create.return_value = mock_response
        
        code = """
def authenticate(api_key):
    headers = {'Authorization': f'Bearer {api_key}'}
    response = requests.get('https://api.example.com/auth', headers=headers)
    return response.json()
"""
        context_before = "To authenticate with our API:"
        context_after = "This will return your auth token."
        
        with patch.dict('os.environ', {'MODEL_CHOICE': 'gpt-4'}):
            summary = generate_code_example_summary(code, context_before, context_after)
        
        assert summary == "This function demonstrates API authentication using requests."
        mock_create.assert_called_once()
    
    @patch('openai.chat.completions.create')
    def test_generate_code_example_summary_api_error(self, mock_create):
        """Test code summary generation with API error."""
        mock_create.side_effect = Exception("API error")
        
        with patch.dict('os.environ', {'MODEL_CHOICE': 'gpt-4'}):
            summary = generate_code_example_summary("code", "before", "after")
        
        assert summary == "Code example for demonstration purposes."
    
    @patch('openai.chat.completions.create')
    def test_generate_code_example_summary_long_content(self, mock_create):
        """Test code summary with long context content."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Summary of long content"))]
        mock_create.return_value = mock_response
        
        code = "def test(): pass" * 100  # Long code
        context_before = "This is a very long context. " * 50  # Long context
        context_after = "More context here. " * 50
        
        with patch.dict('os.environ', {'MODEL_CHOICE': 'gpt-4'}):
            summary = generate_code_example_summary(code, context_before, context_after)
        
        # Verify API was called with truncated content
        call_kwargs = mock_create.call_args[1]
        prompt = call_kwargs['messages'][1]['content']
        
        # Should truncate long content
        assert len(prompt) < len(code) + len(context_before) + len(context_after)
        assert summary == "Summary of long content"


class TestSourceSummarization:
    """Test source summarization functionality."""
    
    @patch('openai.chat.completions.create')
    def test_extract_source_summary_success(self, mock_create):
        """Test successful source summary extraction."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="FastAPI is a modern web framework for building APIs."))
        ]
        mock_create.return_value = mock_response
        
        content = """
# FastAPI Documentation

FastAPI is a modern, fast (high-performance), web framework for building APIs 
with Python 3.6+ based on standard Python type hints.

## Features
- Fast to code
- Fast to run
- Fewer bugs
- Intuitive
"""
        
        with patch.dict('os.environ', {'MODEL_CHOICE': 'gpt-4'}):
            summary = extract_source_summary("fastapi.tiangolo.com", content)
        
        assert "FastAPI is a modern web framework" in summary
        mock_create.assert_called_once()
    
    @patch('openai.chat.completions.create')
    def test_extract_source_summary_empty_content(self, mock_create):
        """Test source summary with empty content."""
        summary = extract_source_summary("example.com", "")
        
        assert summary == "Content from example.com"
        mock_create.assert_not_called()
    
    @patch('openai.chat.completions.create')
    def test_extract_source_summary_api_error(self, mock_create):
        """Test source summary with API error."""
        mock_create.side_effect = Exception("API error")
        
        with patch.dict('os.environ', {'MODEL_CHOICE': 'gpt-4'}):
            summary = extract_source_summary("example.com", "Some content")
        
        assert summary == "Content from example.com"
    
    @patch('openai.chat.completions.create')
    def test_extract_source_summary_max_length(self, mock_create):
        """Test source summary with length limit."""
        long_summary = "A" * 1000  # Very long summary
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=long_summary))]
        mock_create.return_value = mock_response
        
        with patch.dict('os.environ', {'MODEL_CHOICE': 'gpt-4'}):
            summary = extract_source_summary("example.com", "Content", max_length=100)
        
        # Should be truncated to max_length + "..."
        assert len(summary) == 103  # 100 + "..."
        assert summary.endswith("...")
    
    @patch('openai.chat.completions.create')
    def test_extract_source_summary_long_content_truncation(self, mock_create):
        """Test source summary with content truncation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Summary of truncated content"))]
        mock_create.return_value = mock_response
        
        # Content longer than 25000 characters
        long_content = "This is documentation content. " * 1000
        
        with patch.dict('os.environ', {'MODEL_CHOICE': 'gpt-4'}):
            summary = extract_source_summary("example.com", long_content)
        
        # Verify API was called with truncated content
        call_kwargs = mock_create.call_args[1]
        prompt = call_kwargs['messages'][1]['content']
        assert len(prompt) < len(long_content)


class TestBatchProcessing:
    """Test batch processing of code examples."""
    
    @patch('content.processor.extract_code_blocks')
    @patch('content.processor.generate_code_example_summary')
    def test_process_code_examples_batch_success(self, mock_summary, mock_extract):
        """Test successful batch processing of code examples."""
        # Mock code block extraction
        mock_extract.return_value = [
            {
                'code': 'def test(): pass',
                'language': 'python',
                'context_before': 'Before context',
                'context_after': 'After context',
                'full_context': 'Before context\n\ndef test(): pass\n\nAfter context'
            }
        ]
        
        # Mock summary generation
        mock_summary.return_value = "Simple test function"
        
        markdown_content = "# Test\n```python\ndef test(): pass\n```"
        url = "https://example.com/docs"
        
        results = process_code_examples_batch(markdown_content, url)
        
        assert len(results) == 1
        assert results[0]['url'] == url
        assert results[0]['chunk_number'] == 0
        assert results[0]['code'] == 'def test(): pass'
        assert results[0]['summary'] == "Simple test function"
        assert results[0]['metadata']['language'] == 'python'
    
    @patch('content.processor.extract_code_blocks')
    def test_process_code_examples_batch_no_code(self, mock_extract):
        """Test batch processing with no code blocks."""
        mock_extract.return_value = []
        
        results = process_code_examples_batch("No code here", "https://example.com")
        
        assert len(results) == 0
    
    @patch('content.processor.extract_code_blocks')
    @patch('content.processor.generate_code_example_summary')
    def test_process_code_examples_batch_multiple_blocks(self, mock_summary, mock_extract):
        """Test batch processing with multiple code blocks."""
        mock_extract.return_value = [
            {
                'code': 'def func1(): pass',
                'language': 'python',
                'context_before': 'Context 1',
                'context_after': 'After 1',
                'full_context': 'Full context 1'
            },
            {
                'code': 'function func2() {}',
                'language': 'javascript',
                'context_before': 'Context 2',
                'context_after': 'After 2',
                'full_context': 'Full context 2'
            }
        ]
        
        mock_summary.side_effect = ["Python function", "JavaScript function"]
        
        results = process_code_examples_batch("Multiple code blocks", "https://example.com")
        
        assert len(results) == 2
        assert results[0]['chunk_number'] == 0
        assert results[1]['chunk_number'] == 1
        assert results[0]['metadata']['language'] == 'python'
        assert results[1]['metadata']['language'] == 'javascript'


class TestContentAnalysis:
    """Test content structure analysis functionality."""
    
    def test_analyze_content_structure_documentation(self):
        """Test analysis of documentation content."""
        content = """
# API Documentation

## Getting Started

This section covers the basics.

### Authentication

Use your API key:

```python
headers = {'Authorization': 'Bearer token'}
```

## Endpoints

### GET /users

Returns user data.

### POST /users  

Creates new user.
"""
        
        analysis = analyze_content_structure(content)
        
        assert analysis['word_count'] > 0
        assert analysis['line_count'] > 10
        assert analysis['header_count'] == 6  # H1, H2, H3, H2, H3, H3
        assert analysis['code_block_count'] == 1
        assert analysis['content_type'] == 'api_documentation'
        assert len(analysis['header_structure']) == 6
        assert analysis['estimated_reading_time'] >= 1
    
    def test_analyze_content_structure_tutorial(self):
        """Test analysis of tutorial content."""
        content = """
# Python Tutorial: Getting Started

This tutorial will teach you Python basics.

## Step 1: Installation

First, install Python from python.org.

## Step 2: Your First Program

```python
print("Hello, World!")
```

Follow these steps carefully.
"""
        
        analysis = analyze_content_structure(content)
        
        assert analysis['content_type'] == 'tutorial'
        assert analysis['header_count'] == 3
        assert analysis['code_block_count'] == 1
        assert 'tutorial' in content.lower()
    
    def test_analyze_content_structure_code_heavy(self):
        """Test analysis of code-heavy content."""
        content = """
# Code Examples

```python
def func1(): pass
```

```javascript  
function func2() {}
```

```bash
echo "test"
```

```sql
SELECT * FROM users;
```

```python
class Example:
    pass
```

More code examples:

```java
public class Test {}
```
"""
        
        analysis = analyze_content_structure(content)
        
        assert analysis['content_type'] == 'documentation'  # 6 blocks < 10 threshold 
        assert analysis['code_block_count'] == 6
    
    def test_analyze_content_structure_header_levels(self):
        """Test header structure analysis."""
        content = """
# Level 1

## Level 2 A

### Level 3 A

### Level 3 B

## Level 2 B

#### Level 4

# Another Level 1
"""
        
        analysis = analyze_content_structure(content)
        
        header_structure = analysis['header_structure']
        assert len(header_structure) == 7
        
        levels = [h['level'] for h in header_structure]
        assert levels == [1, 2, 3, 3, 2, 4, 1]
        
        texts = [h['text'] for h in header_structure]
        assert 'Level 1' in texts[0]
        assert 'Level 2 A' in texts[1]
    
    def test_analyze_content_structure_empty_content(self):
        """Test analysis of empty content."""
        analysis = analyze_content_structure("")
        
        assert analysis['word_count'] == 0
        assert analysis['line_count'] == 1
        assert analysis['header_count'] == 0
        assert analysis['code_block_count'] == 0
        assert analysis['content_type'] == 'documentation'
        assert analysis['estimated_reading_time'] == 1  # Minimum 1 minute
    
    def test_analyze_content_structure_reading_time(self):
        """Test reading time estimation."""
        # Create content with known word count
        words = ['word'] * 400  # 400 words
        content = ' '.join(words)
        
        analysis = analyze_content_structure(content)
        
        assert analysis['word_count'] == 400
        assert analysis['estimated_reading_time'] == 2  # 400 words / 200 wpm = 2 minutes


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_extract_code_blocks_malformed_markdown(self):
        """Test extraction from malformed markdown."""
        malformed = """
# Title

```python
def incomplete_function(
# Missing closing backticks

More content here
```
"""
        
        blocks = extract_code_blocks(malformed, min_length=10)
        
        # Should handle malformed content gracefully
        assert isinstance(blocks, list)
    
    def test_extract_code_blocks_nested_backticks(self):
        """Test extraction with nested backticks in code."""
        markdown = """
# Example

```python
def example():
    code = '''
    This is a triple-quoted string
    '''
    return code
```
"""
        
        blocks = extract_code_blocks(markdown, min_length=20)
        
        assert len(blocks) == 1
        assert "triple-quoted string" in blocks[0]['code']
    
    def test_generate_summary_with_special_characters(self):
        """Test summary generation with special characters."""
        with patch('openai.chat.completions.create') as mock_create:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Summary with émojis 🚀"))]
            mock_create.return_value = mock_response
            
            code = "def test(): return '🚀'"
            
            with patch.dict('os.environ', {'MODEL_CHOICE': 'gpt-4'}):
                summary = generate_code_example_summary(code, "", "")
            
            assert "émojis 🚀" in summary