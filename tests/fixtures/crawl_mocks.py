"""
Crawl4AI mocks for testing.

This module provides mock implementations of Crawl4AI components to avoid
actual web crawling during testing.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import List, Dict, Any, Optional


class MockCrawlResult:
    """Mock crawl result from Crawl4AI."""
    
    def __init__(self, url: str, success: bool = True, markdown: str = "", 
                 extracted_content: str = "", links: Dict = None, 
                 status_code: int = 200, error: str = ""):
        self.url = url
        self.success = success
        self.markdown = markdown or self._generate_mock_markdown(url)
        self.extracted_content = extracted_content or self.markdown
        self.links = links or {"internal": [], "external": []}
        self.status_code = status_code
        self.error = error
        self.html = f"<html><body>{self.markdown}</body></html>"
        self.cleaned_html = self.markdown
        self.media = {"images": []}
    
    def _generate_mock_markdown(self, url: str) -> str:
        """Generate realistic mock markdown based on URL."""
        domain = url.split("//")[-1].split("/")[0]
        path = url.split("/")[-1] if "/" in url else "index"
        
        return f"""# Documentation for {domain}

## Overview

This is mock documentation content for {url}.

## Getting Started

To get started, follow these steps:

1. Install the required dependencies
2. Configure your environment
3. Run the application

```python
# Example code for {path}
def example_function():
    '''This is an example function.'''
    return "Hello from {domain}"

# Usage
result = example_function()
print(result)
```

## API Reference

### Methods

- `get_data()` - Retrieves data from the API
- `post_data(data)` - Sends data to the API
- `delete_data(id)` - Deletes data by ID

## Conclusion

This concludes the documentation for {domain}.
"""


@pytest.fixture
def mock_async_web_crawler():
    """Mock AsyncWebCrawler for testing."""
    crawler = AsyncMock()
    
    async def mock_arun(url: str, **kwargs) -> MockCrawlResult:
        """Mock crawl execution."""
        # Simulate different responses based on URL
        if "error" in url or "404" in url:
            return MockCrawlResult(url, success=False, status_code=404, error="Page not found")
        elif "timeout" in url:
            return MockCrawlResult(url, success=False, error="Request timeout")
        else:
            return MockCrawlResult(url, success=True)
    
    crawler.arun = mock_arun
    return crawler


@pytest.fixture
def mock_crawl_results():
    """Pre-defined mock crawl results for testing."""
    return {
        "successful_crawl": MockCrawlResult(
            url="https://example.com/docs",
            success=True,
            markdown="""# API Documentation

## Authentication

Use your API key to authenticate requests.

```python
import requests

headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
}

response = requests.get('https://api.example.com/data', headers=headers)
```

## Endpoints

### GET /users
Retrieves user information.

### POST /users
Creates a new user.
""",
            links={"internal": ["https://example.com/guide"], "external": []},
            status_code=200
        ),
        "failed_crawl": MockCrawlResult(
            url="https://nonexistent.com",
            success=False,
            status_code=404,
            error="Page not found"
        ),
        "timeout_crawl": MockCrawlResult(
            url="https://slow.example.com",
            success=False,
            error="Request timeout"
        )
    }


@pytest.fixture
def mock_sitemap_content():
    """Mock sitemap XML content for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://example.com/</loc>
        <lastmod>2024-01-15</lastmod>
        <priority>1.0</priority>
    </url>
    <url>
        <loc>https://example.com/docs</loc>
        <lastmod>2024-01-15</lastmod>
        <priority>0.8</priority>
    </url>
    <url>
        <loc>https://example.com/api</loc>
        <lastmod>2024-01-15</lastmod>
        <priority>0.8</priority>
    </url>
    <url>
        <loc>https://example.com/guide</loc>
        <lastmod>2024-01-15</lastmod>
        <priority>0.6</priority>
    </url>
</urlset>"""


@pytest.fixture
def mock_txt_file_content():
    """Mock text file content with URLs for testing."""
    return """# URL list for crawling
https://example.com/page1
https://example.com/page2
https://docs.example.com/api
https://docs.example.com/guide

# Comments are ignored
# https://commented.com/ignored

https://github.com/example/project
https://stackoverflow.com/questions/12345"""


class MockBrowserConfig:
    """Mock browser configuration."""
    
    def __init__(self, headless: bool = True, **kwargs):
        self.headless = headless
        self.kwargs = kwargs


@pytest.fixture
def mock_browser_config():
    """Mock browser configuration for testing."""
    return MockBrowserConfig()


@pytest.fixture
def mock_crawler_run_config():
    """Mock crawler run configuration."""
    class MockCrawlerRunConfig:
        def __init__(self, **kwargs):
            self.cache_mode = kwargs.get('cache_mode', 'bypass')
            self.js_code = kwargs.get('js_code', [])
            self.wait_for = kwargs.get('wait_for', '')
            self.css_selector = kwargs.get('css_selector', '')
            self.word_count_threshold = kwargs.get('word_count_threshold', 10)
    
    return MockCrawlerRunConfig


@pytest.fixture
def mock_concurrent_crawler():
    """Mock concurrent crawler for batch operations."""
    async def mock_concurrent_crawl(urls: List[str], max_concurrent: int = 10) -> List[MockCrawlResult]:
        """Mock concurrent crawling of multiple URLs."""
        results = []
        for url in urls:
            if "error" in url:
                result = MockCrawlResult(url, success=False, status_code=500, error="Server error")
            else:
                result = MockCrawlResult(url, success=True)
            results.append(result)
        return results
    
    return mock_concurrent_crawl


@pytest.fixture
def crawl_performance_metrics():
    """Mock performance metrics for crawling operations."""
    return {
        "single_page": {
            "avg_time": 2.5,
            "max_time": 10.0,
            "success_rate": 0.95
        },
        "batch_crawl": {
            "avg_time_per_page": 1.8,
            "max_time_per_page": 8.0,
            "success_rate": 0.92,
            "concurrent_limit": 10
        },
        "sitemap_crawl": {
            "avg_pages_per_minute": 30,
            "max_pages": 1000,
            "success_rate": 0.88
        }
    }


class MockMemoryManager:
    """Mock memory management for crawler."""
    
    def __init__(self, max_memory_mb: int = 1000):
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0
    
    def check_memory(self) -> Dict[str, Any]:
        return {
            "current_mb": self.current_memory_mb,
            "max_mb": self.max_memory_mb,
            "usage_percent": (self.current_memory_mb / self.max_memory_mb) * 100
        }
    
    def adjust_concurrency(self, current_concurrent: int) -> int:
        """Adjust concurrency based on memory usage."""
        memory_usage = self.current_memory_mb / self.max_memory_mb
        if memory_usage > 0.8:
            return max(1, current_concurrent - 2)
        elif memory_usage < 0.5:
            return min(20, current_concurrent + 2)
        return current_concurrent


@pytest.fixture
def mock_memory_manager():
    """Mock memory manager for testing."""
    return MockMemoryManager()


@pytest.fixture
def crawl_error_scenarios():
    """Crawl error scenarios for testing error handling."""
    return {
        "network_error": {
            "url": "https://network-error.com",
            "success": False,
            "error": "Network connection failed",
            "status_code": 0
        },
        "permission_denied": {
            "url": "https://forbidden.com",
            "success": False,
            "error": "Permission denied",
            "status_code": 403
        },
        "not_found": {
            "url": "https://example.com/not-found",
            "success": False,
            "error": "Page not found",
            "status_code": 404
        },
        "server_error": {
            "url": "https://server-error.com",
            "success": False,
            "error": "Internal server error",
            "status_code": 500
        },
        "timeout": {
            "url": "https://timeout.com",
            "success": False,
            "error": "Request timeout",
            "status_code": 0
        }
    }


@pytest.fixture
def mock_extract_links():
    """Mock link extraction functionality."""
    def extract_links(html_content: str, base_url: str) -> Dict[str, List[str]]:
        """Extract internal and external links from HTML."""
        # Simple mock implementation
        if "example.com" in base_url:
            return {
                "internal": [
                    "https://example.com/docs",
                    "https://example.com/api",
                    "https://example.com/guide"
                ],
                "external": [
                    "https://github.com/example/project",
                    "https://stackoverflow.com/questions/123"
                ]
            }
        return {"internal": [], "external": []}
    
    return extract_links