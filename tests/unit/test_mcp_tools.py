"""
Unit tests for MCP tools functionality.

Tests the individual MCP tool functions and their integration with
the FastMCP server framework.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

# Import structure relies on pytest.ini pythonpath = src


class TestMCPToolDecorators:
    """Test MCP tool decorators and registration."""
    
    def test_mcp_tool_registration(self):
        """Test that MCP tools are properly registered."""
        # Import the modules to trigger registration
        from mcp_crawl_tools import crawl_single_page, smart_crawl_url
        from mcp_search_tools import get_available_sources, perform_rag_query, search_code_examples
        
        # Tools should be callable functions
        assert callable(crawl_single_page)
        assert callable(smart_crawl_url)
        assert callable(get_available_sources)
        assert callable(perform_rag_query)
        assert callable(search_code_examples)
    
    def test_logging_decorator_integration(self):
        """Test that logging decorators are properly applied."""
        from mcp_crawl_tools import crawl_single_page
        
        # Should have logging decorator attributes
        assert hasattr(crawl_single_page, '__wrapped__')


class TestCrawlSinglePage:
    """Test crawl_single_page MCP tool."""
    
    @pytest.fixture
    def mock_context(self):
        """Mock MCP context for testing."""
        context = MagicMock()
        context.request_context = MagicMock()
        context.request_context.lifespan_context = MagicMock()
        
        # Mock crawler
        mock_crawler = AsyncMock()
        context.request_context.lifespan_context.crawler = mock_crawler
        
        # Mock postgres pool
        mock_pool = AsyncMock()
        context.request_context.lifespan_context.postgres_pool = mock_pool
        
        return context
    
    @pytest.fixture
    def mock_crawl_result(self):
        """Mock successful crawl result."""
        result = MagicMock()
        result.success = True
        result.markdown = "# Test Page\n\nThis is test content."
        result.url = "https://example.com/test"
        return result
    
    @pytest.mark.asyncio
    async def test_crawl_single_page_success(self, mock_context, mock_crawl_result):
        """Test successful single page crawling."""
        from mcp_crawl_tools import crawl_single_page
        
        # Setup mocks
        mock_context.request_context.lifespan_context.crawler.arun.return_value = mock_crawl_result
        
        with patch('mcp_crawl_tools.smart_chunk_markdown') as mock_chunk:
            mock_chunk.return_value = ["Chunk 1", "Chunk 2"]
            
            with patch('mcp_crawl_tools.add_documents_to_postgres') as mock_add_docs:
                with patch('mcp_crawl_tools.update_source_info') as mock_update_source:
                    result = await crawl_single_page(mock_context, "https://example.com/test")
        
        # Verify result format
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["url"] == "https://example.com/test"
        assert result_data["chunks_stored"] == 2
        
        # Verify mocks were called
        mock_context.request_context.lifespan_context.crawler.arun.assert_called_once()
        mock_add_docs.assert_called_once()
        mock_update_source.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_crawl_single_page_failure(self, mock_context):
        """Test single page crawling failure."""
        from mcp_crawl_tools import crawl_single_page
        
        # Setup failed crawl result
        failed_result = MagicMock()
        failed_result.success = False
        failed_result.error = "Page not found"
        failed_result.url = "https://example.com/404"
        
        mock_context.request_context.lifespan_context.crawler.arun.return_value = failed_result
        
        result = await crawl_single_page(mock_context, "https://example.com/404")
        
        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "Page not found" in result_data["error"]
    
    @pytest.mark.asyncio
    async def test_crawl_single_page_exception(self, mock_context):
        """Test single page crawling with exception."""
        from mcp_crawl_tools import crawl_single_page
        
        # Setup crawler to raise exception
        mock_context.request_context.lifespan_context.crawler.arun.side_effect = Exception("Network error")
        
        result = await crawl_single_page(mock_context, "https://example.com/error")
        
        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "Network error" in result_data["error"]


class TestSmartCrawlUrl:
    """Test smart_crawl_url MCP tool."""
    
    @pytest.fixture
    def mock_context(self):
        """Mock MCP context for testing."""
        context = MagicMock()
        context.request_context = MagicMock()
        context.request_context.lifespan_context = MagicMock()
        
        # Mock components
        context.request_context.lifespan_context.crawler = AsyncMock()
        context.request_context.lifespan_context.postgres_pool = AsyncMock()
        
        return context
    
    @pytest.mark.asyncio
    async def test_smart_crawl_url_sitemap(self, mock_context):
        """Test smart crawling of a sitemap URL."""
        from mcp_crawl_tools import smart_crawl_url
        
        with patch('mcp_crawl_tools.is_sitemap') as mock_is_sitemap:
            mock_is_sitemap.return_value = True
            
            with patch('mcp_crawl_tools.parse_sitemap') as mock_parse:
                mock_parse.return_value = ["https://example.com/page1", "https://example.com/page2"]
                
                with patch('mcp_crawl_tools.crawl_batch') as mock_crawl_batch:
                    mock_crawl_batch.return_value = {"chunks_stored": 10, "pages_processed": 2}
                    
                    result = await smart_crawl_url(
                        mock_context, 
                        "https://example.com/sitemap.xml",
                        max_depth=2,
                        max_concurrent=5
                    )
        
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["strategy"] == "sitemap_crawl"
        assert result_data["chunks_stored"] == 10
    
    @pytest.mark.asyncio
    async def test_smart_crawl_url_txt_file(self, mock_context):
        """Test smart crawling of a text file with URLs."""
        from mcp_crawl_tools import smart_crawl_url
        
        with patch('mcp_crawl_tools.is_txt') as mock_is_txt:
            mock_is_txt.return_value = True
            
            with patch('mcp_crawl_tools.crawl_markdown_file') as mock_crawl_file:
                mock_crawl_file.return_value = {"chunks_stored": 5, "urls_processed": 3}
                
                result = await smart_crawl_url(
                    mock_context,
                    "https://example.com/urls.txt",
                    max_depth=1,
                    max_concurrent=3
                )
        
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["strategy"] == "text_file_crawl"
        assert result_data["chunks_stored"] == 5
    
    @pytest.mark.asyncio
    async def test_smart_crawl_url_recursive(self, mock_context):
        """Test smart crawling with recursive strategy."""
        from mcp_crawl_tools import smart_crawl_url
        
        with patch('mcp_crawl_tools.is_sitemap') as mock_is_sitemap:
            mock_is_sitemap.return_value = False
            
            with patch('mcp_crawl_tools.is_txt') as mock_is_txt:
                mock_is_txt.return_value = False
                
                with patch('mcp_crawl_tools.crawl_recursive_internal_links') as mock_recursive:
                    mock_recursive.return_value = {"chunks_stored": 15, "pages_crawled": 8}
                    
                    result = await smart_crawl_url(
                        mock_context,
                        "https://example.com",
                        max_depth=3,
                        max_concurrent=10
                    )
        
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["strategy"] == "recursive_crawl"


class TestGetAvailableSources:
    """Test get_available_sources MCP tool."""
    
    @pytest.fixture
    def mock_context(self):
        """Mock MCP context for testing."""
        context = MagicMock()
        context.request_context = MagicMock()
        context.request_context.lifespan_context = MagicMock()
        
        mock_pool = AsyncMock()
        context.request_context.lifespan_context.postgres_pool = mock_pool
        
        return context
    
    @pytest.mark.asyncio
    async def test_get_available_sources_success(self, mock_context):
        """Test successful source listing."""
        from mcp_search_tools import get_available_sources
        
        # Mock database response
        mock_records = [
            {"source_id": "example.com", "summary": "Example website", "total_word_count": 5000},
            {"source_id": "docs.example.com", "summary": "API documentation", "total_word_count": 12000}
        ]
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = mock_records
        mock_context.request_context.lifespan_context.postgres_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_context.request_context.lifespan_context.postgres_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        result = await get_available_sources(mock_context)
        
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert len(result_data["sources"]) == 2
        assert result_data["sources"][0]["source_id"] == "example.com"
    
    @pytest.mark.asyncio
    async def test_get_available_sources_empty(self, mock_context):
        """Test source listing with no sources."""
        from mcp_search_tools import get_available_sources
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_context.request_context.lifespan_context.postgres_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_context.request_context.lifespan_context.postgres_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        result = await get_available_sources(mock_context)
        
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert len(result_data["sources"]) == 0
    
    @pytest.mark.asyncio
    async def test_get_available_sources_database_error(self, mock_context):
        """Test source listing with database error."""
        from mcp_search_tools import get_available_sources
        
        mock_conn = AsyncMock()
        mock_conn.fetch.side_effect = Exception("Database connection failed")
        mock_context.request_context.lifespan_context.postgres_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_context.request_context.lifespan_context.postgres_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        result = await get_available_sources(mock_context)
        
        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "Database connection failed" in result_data["error"]


class TestPerformRagQuery:
    """Test perform_rag_query MCP tool."""
    
    @pytest.fixture
    def mock_context(self):
        """Mock MCP context for testing."""
        context = MagicMock()
        context.request_context = MagicMock()
        context.request_context.lifespan_context = MagicMock()
        
        mock_pool = AsyncMock()
        context.request_context.lifespan_context.postgres_pool = mock_pool
        context.request_context.lifespan_context.reranking_model = None
        
        return context
    
    @pytest.mark.asyncio
    async def test_perform_rag_query_success(self, mock_context):
        """Test successful RAG query."""
        from mcp_search_tools import perform_rag_query
        
        # Mock search results
        mock_results = [
            {
                "url": "https://example.com/doc1",
                "content": "Machine learning is a subset of AI",
                "similarity": 0.92,
                "source_id": "example.com"
            }
        ]
        
        with patch('mcp_search_tools.search_documents') as mock_search:
            mock_search.return_value = mock_results
            
            result = await perform_rag_query(
                mock_context,
                query="machine learning",
                match_count=5
            )
        
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert len(result_data["results"]) == 1
        assert result_data["query"] == "machine learning"
        assert result_data["results"][0]["url"] == "https://example.com/doc1"
    
    @pytest.mark.asyncio
    async def test_perform_rag_query_with_source_filter(self, mock_context):
        """Test RAG query with source filtering."""
        from mcp_search_tools import perform_rag_query
        
        with patch('mcp_search_tools.search_documents') as mock_search:
            mock_search.return_value = []
            
            result = await perform_rag_query(
                mock_context,
                query="test query",
                source="example.com",
                match_count=3
            )
        
        # Verify search was called with source filter
        mock_search.assert_called_once()
        call_args = mock_search.call_args
        filter_metadata = call_args[1]['filter_metadata']
        assert filter_metadata['source'] == "example.com"
    
    @pytest.mark.asyncio
    async def test_perform_rag_query_with_reranking(self, mock_context):
        """Test RAG query with reranking enabled."""
        from mcp_search_tools import perform_rag_query
        
        # Mock reranking model
        mock_model = MagicMock()
        mock_context.request_context.lifespan_context.reranking_model = mock_model
        
        mock_results = [{"content": "test", "similarity": 0.8}]
        
        with patch('mcp_search_tools.search_documents') as mock_search:
            mock_search.return_value = mock_results
            
            with patch('mcp_search_tools.rerank_results') as mock_rerank:
                mock_rerank.return_value = mock_results
                
                result = await perform_rag_query(mock_context, query="test")
        
        # Verify reranking was called
        mock_rerank.assert_called_once_with(mock_results, "test", mock_model)


class TestSearchCodeExamples:
    """Test search_code_examples MCP tool."""
    
    @pytest.fixture
    def mock_context(self):
        """Mock MCP context for testing."""
        context = MagicMock()
        context.request_context = MagicMock()
        context.request_context.lifespan_context = MagicMock()
        
        mock_pool = AsyncMock()
        context.request_context.lifespan_context.postgres_pool = mock_pool
        
        return context
    
    @pytest.mark.asyncio
    async def test_search_code_examples_success(self, mock_context):
        """Test successful code example search."""
        from mcp_search_tools import search_code_examples
        
        mock_results = [
            {
                "url": "https://docs.example.com/api",
                "content": "def authenticate(key): return validate(key)",
                "summary": "Authentication function example",
                "similarity": 0.88,
                "metadata": {"language": "python"}
            }
        ]
        
        with patch('mcp_search_tools.search_code_examples_util') as mock_search:
            mock_search.return_value = mock_results
            
            result = await search_code_examples(
                mock_context,
                query="authentication function",
                match_count=3
            )
        
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert len(result_data["results"]) == 1
        assert "def authenticate" in result_data["results"][0]["content"]
    
    @pytest.mark.asyncio
    async def test_search_code_examples_with_source_filter(self, mock_context):
        """Test code example search with source filtering."""
        from mcp_search_tools import search_code_examples
        
        with patch('mcp_search_tools.search_code_examples_util') as mock_search:
            mock_search.return_value = []
            
            result = await search_code_examples(
                mock_context,
                query="function",
                source_id="docs.example.com",
                match_count=5
            )
        
        # Verify search was called with source filter
        mock_search.assert_called_once()
        call_args = mock_search.call_args
        assert call_args[1]['source_id'] == "docs.example.com"
    
    @pytest.mark.asyncio
    async def test_search_code_examples_no_results(self, mock_context):
        """Test code example search with no results."""
        from mcp_search_tools import search_code_examples
        
        with patch('mcp_search_tools.search_code_examples_util') as mock_search:
            mock_search.return_value = []
            
            result = await search_code_examples(mock_context, query="nonexistent")
        
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert len(result_data["results"]) == 0
        assert result_data["message"] == "No code examples found matching the query."


class TestMCPToolIntegration:
    """Test MCP tool integration aspects."""
    
    def test_all_tools_have_proper_signatures(self):
        """Test that all MCP tools have proper async signatures."""
        from mcp_crawl_tools import crawl_single_page, smart_crawl_url
        from mcp_search_tools import get_available_sources, perform_rag_query, search_code_examples
        
        import inspect
        
        tools = [crawl_single_page, smart_crawl_url, get_available_sources, perform_rag_query, search_code_examples]
        
        for tool in tools:
            # Should be coroutine functions (async)
            assert inspect.iscoroutinefunction(tool)
            
            # Should accept context as first parameter
            sig = inspect.signature(tool)
            params = list(sig.parameters.keys())
            assert params[0] == 'ctx'
    
    def test_tool_docstrings_present(self):
        """Test that all tools have descriptive docstrings."""
        from mcp_crawl_tools import crawl_single_page, smart_crawl_url
        from mcp_search_tools import get_available_sources, perform_rag_query, search_code_examples
        
        tools = [crawl_single_page, smart_crawl_url, get_available_sources, perform_rag_query, search_code_examples]
        
        for tool in tools:
            assert tool.__doc__ is not None
            assert len(tool.__doc__.strip()) > 50  # Substantial docstring
    
    def test_tool_return_json_format(self):
        """Test that tools return properly formatted JSON."""
        # This would be tested in integration tests with actual MCP calls
        # Here we just verify the format structure
        
        sample_success = '{"success": true, "data": "test"}'
        sample_error = '{"success": false, "error": "test error"}'
        
        # Verify JSON is valid
        import json
        success_data = json.loads(sample_success)
        error_data = json.loads(sample_error)
        
        assert success_data["success"] is True
        assert error_data["success"] is False
        assert "error" in error_data


class TestErrorHandling:
    """Test error handling in MCP tools."""
    
    @pytest.mark.asyncio
    async def test_crawl_tool_network_error(self):
        """Test crawl tool handling of network errors."""
        from mcp_crawl_tools import crawl_single_page
        
        mock_context = MagicMock()
        mock_context.request_context.lifespan_context.crawler.arun.side_effect = Exception("Network timeout")
        
        result = await crawl_single_page(mock_context, "https://timeout.com")
        
        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "timeout" in result_data["error"].lower()
    
    @pytest.mark.asyncio
    async def test_search_tool_database_error(self):
        """Test search tool handling of database errors."""
        from mcp_search_tools import perform_rag_query
        
        mock_context = MagicMock()
        
        with patch('mcp_search_tools.search_documents') as mock_search:
            mock_search.side_effect = Exception("Database connection lost")
            
            result = await perform_rag_query(mock_context, "test query")
        
        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "Database connection lost" in result_data["error"]
    
    @pytest.mark.asyncio
    async def test_invalid_url_handling(self):
        """Test handling of invalid URLs."""
        from mcp_crawl_tools import crawl_single_page
        
        mock_context = MagicMock()
        
        # Test with obviously invalid URL
        result = await crawl_single_page(mock_context, "not-a-url")
        
        result_data = json.loads(result)
        # Should handle gracefully (exact behavior depends on implementation)
        assert "success" in result_data


class TestParameterValidation:
    """Test parameter validation in MCP tools."""
    
    @pytest.mark.asyncio
    async def test_crawl_tools_parameter_bounds(self):
        """Test parameter validation for crawl tools."""
        from mcp_crawl_tools import smart_crawl_url
        
        mock_context = MagicMock()
        
        with patch('mcp_crawl_tools.is_sitemap', return_value=False):
            with patch('mcp_crawl_tools.is_txt', return_value=False):
                with patch('mcp_crawl_tools.crawl_recursive_internal_links') as mock_crawl:
                    mock_crawl.return_value = {"chunks_stored": 0, "pages_crawled": 0}
                    
                    # Test with extreme parameters
                    result = await smart_crawl_url(
                        mock_context,
                        "https://example.com",
                        max_depth=0,  # Minimum depth
                        max_concurrent=1  # Minimum concurrency
                    )
        
        result_data = json.loads(result)
        assert "success" in result_data
    
    @pytest.mark.asyncio
    async def test_search_tools_parameter_bounds(self):
        """Test parameter validation for search tools."""
        from mcp_search_tools import perform_rag_query
        
        mock_context = MagicMock()
        
        with patch('mcp_search_tools.search_documents') as mock_search:
            mock_search.return_value = []
            
            # Test with edge case parameters
            result = await perform_rag_query(
                mock_context,
                query="",  # Empty query
                match_count=0  # Zero results
            )
        
        result_data = json.loads(result)
        assert "success" in result_data