"""
Unit tests for search engine module.

Tests all search-related functionality including result processing, reranking,
query enhancement, and result filtering.
"""
import pytest
from unittest.mock import MagicMock

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from search.engine import (
    rerank_results,
    filter_search_results,
    enhance_search_query,
    process_search_results,
    extract_relevant_snippet,
    highlight_query_terms,
    analyze_relevance,
    format_metadata_for_display,
    calculate_search_quality_score
)


class TestReranking:
    """Test search result reranking functionality."""
    
    def test_rerank_results_with_model(self):
        """Test reranking with a cross-encoder model."""
        # Mock cross-encoder model
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.7, 0.8]
        
        results = [
            {"content": "First result content", "similarity": 0.8},
            {"content": "Second result content", "similarity": 0.9}, 
            {"content": "Third result content", "similarity": 0.7}
        ]
        
        reranked = rerank_results(results, "test query", mock_model)
        
        # Should be reordered by rerank score (0.9, 0.8, 0.7)
        assert reranked[0]["rerank_score"] == 0.9
        assert reranked[1]["rerank_score"] == 0.8
        assert reranked[2]["rerank_score"] == 0.7
        
        # Verify model was called correctly
        mock_model.predict.assert_called_once()
        call_args = mock_model.predict.call_args[0][0]
        assert len(call_args) == 3  # Three query-document pairs
    
    def test_rerank_results_without_model(self):
        """Test reranking without a model returns original results."""
        results = [
            {"content": "First result", "similarity": 0.8},
            {"content": "Second result", "similarity": 0.9}
        ]
        
        reranked = rerank_results(results, "test query", model=None)
        
        # Should return original results unchanged
        assert reranked == results
    
    def test_rerank_results_empty_list(self):
        """Test reranking with empty results."""
        mock_model = MagicMock()
        
        reranked = rerank_results([], "test query", mock_model)
        
        assert reranked == []
        mock_model.predict.assert_not_called()
    
    def test_rerank_results_error_handling(self):
        """Test reranking error handling."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Model error")
        
        results = [{"content": "Test content", "similarity": 0.8}]
        
        reranked = rerank_results(results, "test query", mock_model)
        
        # Should return original results on error
        assert reranked == results


class TestResultFiltering:
    """Test search result filtering functionality."""
    
    def test_filter_search_results_by_similarity(self):
        """Test filtering by similarity threshold."""
        results = [
            {"content": "High similarity", "similarity": 0.9},
            {"content": "Medium similarity", "similarity": 0.7},
            {"content": "Low similarity", "similarity": 0.3}
        ]
        
        filtered = filter_search_results(results, min_similarity=0.6)
        
        assert len(filtered) == 2
        assert all(r["similarity"] >= 0.6 for r in filtered)
    
    def test_filter_search_results_by_source(self):
        """Test filtering by source."""
        results = [
            {"content": "Result 1", "source_id": "example.com"},
            {"content": "Result 2", "source_id": "docs.example.com"},
            {"content": "Result 3", "source_id": "example.com"}
        ]
        
        filters = {"source": "example.com"}
        filtered = filter_search_results(results, filters=filters)
        
        assert len(filtered) == 2
        assert all(r["source_id"] == "example.com" for r in filtered)
    
    def test_filter_search_results_by_language(self):
        """Test filtering by programming language."""
        results = [
            {"content": "Python code", "metadata": {"language": "python"}},
            {"content": "JavaScript code", "metadata": {"language": "javascript"}},
            {"content": "Python tutorial", "metadata": {"language": "python"}}
        ]
        
        filters = {"language": "python"}
        filtered = filter_search_results(results, filters=filters)
        
        assert len(filtered) == 2
        assert all(r["metadata"]["language"] == "python" for r in filtered)
    
    def test_filter_search_results_max_results(self):
        """Test limiting maximum number of results."""
        results = [{"content": f"Result {i}", "similarity": 0.8} for i in range(10)]
        
        filtered = filter_search_results(results, max_results=5)
        
        assert len(filtered) == 5
    
    def test_filter_search_results_multiple_filters(self):
        """Test applying multiple filters simultaneously."""
        results = [
            {"content": "Python high", "similarity": 0.9, "source_id": "example.com", "metadata": {"language": "python"}},
            {"content": "Python low", "similarity": 0.4, "source_id": "example.com", "metadata": {"language": "python"}},
            {"content": "JS high", "similarity": 0.8, "source_id": "example.com", "metadata": {"language": "javascript"}},
            {"content": "Python other", "similarity": 0.7, "source_id": "other.com", "metadata": {"language": "python"}}
        ]
        
        filters = {"source": "example.com", "language": "python"}
        filtered = filter_search_results(results, filters=filters, min_similarity=0.6)
        
        assert len(filtered) == 1
        assert filtered[0]["content"] == "Python high"
    
    def test_filter_search_results_empty_list(self):
        """Test filtering empty results."""
        filtered = filter_search_results([], min_similarity=0.5)
        assert filtered == []


class TestQueryEnhancement:
    """Test search query enhancement functionality."""
    
    def test_enhance_search_query_general(self):
        """Test general query enhancement."""
        query = "machine learning"
        enhanced = enhance_search_query(query, search_type='general')
        
        assert enhanced == "machine learning"  # No change for general
    
    def test_enhance_search_query_code(self):
        """Test code search query enhancement."""
        query = "authentication function"
        enhanced = enhance_search_query(query, search_type='code')
        
        assert "Code example for authentication function" == enhanced
    
    def test_enhance_search_query_api(self):
        """Test API search query enhancement."""
        query = "user endpoints"
        enhanced = enhance_search_query(query, search_type='api')
        
        assert "API documentation for user endpoints" == enhanced
    
    def test_enhance_search_query_tutorial(self):
        """Test tutorial search query enhancement."""
        query = "getting started"
        enhanced = enhance_search_query(query, search_type='tutorial')
        
        assert "Tutorial guide for getting started" == enhanced
    
    def test_enhance_search_query_already_enhanced(self):
        """Test that already enhanced queries are not double-enhanced."""
        query = "Code example for authentication"
        enhanced = enhance_search_query(query, search_type='code')
        
        # Should not add "Code example" again
        assert enhanced == query
    
    def test_enhance_search_query_api_already_has_api(self):
        """Test API query that already mentions API."""
        query = "API methods for users"
        enhanced = enhance_search_query(query, search_type='api')
        
        # Should not add "API documentation" since "API" is already there
        assert enhanced == query


class TestSnippetExtraction:
    """Test content snippet extraction functionality."""
    
    def test_extract_relevant_snippet_short_content(self):
        """Test snippet extraction from short content."""
        content = "This is a short piece of content."
        snippet = extract_relevant_snippet(content, "content", max_length=200)
        
        assert snippet == content
    
    def test_extract_relevant_snippet_with_query_match(self):
        """Test snippet extraction around query terms."""
        content = "This is a long piece of content. " * 20 + "Here is the important information about machine learning algorithms. " + "More content here. " * 20
        
        snippet = extract_relevant_snippet(content, "machine learning", max_length=100)
        
        assert "machine learning" in snippet
        assert len(snippet) <= 103  # max_length + "..."
        assert "..." in snippet
    
    def test_extract_relevant_snippet_no_query_match(self):
        """Test snippet extraction when query doesn't match."""
        content = "This is a very long piece of content. " * 50
        
        snippet = extract_relevant_snippet(content, "nonexistent term", max_length=100)
        
        # Should return beginning of content
        assert snippet.startswith("This is a very long")
        assert snippet.endswith("...")
        assert len(snippet) <= 103
    
    def test_extract_relevant_snippet_word_boundaries(self):
        """Test that snippet respects word boundaries."""
        content = "Supercalifragilisticexpialidocious is a very long word that appears in Mary Poppins."
        
        snippet = extract_relevant_snippet(content, "word", max_length=50)
        
        # Should not cut words in the middle
        assert not snippet.startswith("fragilistic")


class TestQueryHighlighting:
    """Test query term highlighting functionality."""
    
    def test_highlight_query_terms_basic(self):
        """Test basic query term highlighting."""
        text = "This is a test of the highlighting system."
        query_terms = {"test", "highlighting"}
        
        highlighted = highlight_query_terms(text, query_terms)
        
        assert "**test**" in highlighted
        assert "**highlighting**" in highlighted
    
    def test_highlight_query_terms_case_insensitive(self):
        """Test case-insensitive highlighting."""
        text = "Test the Testing functionality."
        query_terms = {"test"}
        
        highlighted = highlight_query_terms(text, query_terms)
        
        assert "**test**" in highlighted.lower()
    
    def test_highlight_query_terms_short_terms_ignored(self):
        """Test that very short terms are not highlighted."""
        text = "This is a test of short terms."
        query_terms = {"is", "a", "of"}  # All 2 chars or less
        
        highlighted = highlight_query_terms(text, query_terms)
        
        # Should not highlight short terms
        assert "**is**" not in highlighted
        assert "**a**" not in highlighted
    
    def test_highlight_query_terms_empty_terms(self):
        """Test highlighting with empty query terms."""
        text = "This is test content."
        highlighted = highlight_query_terms(text, set())
        
        assert highlighted == text


class TestRelevanceAnalysis:
    """Test relevance analysis functionality."""
    
    def test_analyze_relevance_high_similarity(self):
        """Test relevance analysis for high similarity results."""
        result = {
            "content": "machine learning algorithms and neural networks",
            "similarity": 0.95,
            "source_id": "docs.example.com",
            "metadata": {"content_type": "documentation"}
        }
        
        relevance = analyze_relevance(result, "machine learning")
        
        assert relevance["similarity_level"] == "high"
        assert relevance["query_coverage"] > 0.5  # "machine" and "learning" found
        assert relevance["source_reliability"] == "high"  # docs.* domain
    
    def test_analyze_relevance_medium_similarity(self):
        """Test relevance analysis for medium similarity results."""
        result = {
            "content": "programming tutorials and examples",
            "similarity": 0.7,
            "source_id": "example.com",
            "metadata": {"content_type": "tutorial"}
        }
        
        relevance = analyze_relevance(result, "programming tutorial")
        
        assert relevance["similarity_level"] == "medium"
        assert relevance["query_coverage"] == 1.0  # Both terms found
    
    def test_analyze_relevance_low_similarity(self):
        """Test relevance analysis for low similarity results."""
        result = {
            "content": "unrelated content about cooking",
            "similarity": 0.3,
            "source_id": "cooking.com",
            "metadata": {"content_type": "recipe"}
        }
        
        relevance = analyze_relevance(result, "machine learning")
        
        assert relevance["similarity_level"] == "low"
        assert relevance["query_coverage"] == 0.0  # No terms found
        assert relevance["source_reliability"] == "medium"  # Not a docs domain


class TestMetadataFormatting:
    """Test metadata formatting for display."""
    
    def test_format_metadata_for_display_complete(self):
        """Test formatting complete metadata."""
        metadata = {
            "chunk_size": 1500,
            "language": "python",
            "content_type": "api_documentation",
            "word_count": 300,
            "contextual_embedding": True
        }
        
        formatted = format_metadata_for_display(metadata)
        
        assert formatted["Content Length"] == "1500 characters"
        assert formatted["Language"] == "Python"
        assert formatted["Content Type"] == "Api Documentation"
        assert formatted["Word Count"] == "300"
        assert formatted["Enhanced Context"] == "Yes"
    
    def test_format_metadata_for_display_partial(self):
        """Test formatting partial metadata."""
        metadata = {
            "chunk_size": 500,
            "language": "javascript"
        }
        
        formatted = format_metadata_for_display(metadata)
        
        assert "Content Length" in formatted
        assert "Language" in formatted
        assert "Content Type" not in formatted  # Not present in input
    
    def test_format_metadata_for_display_empty(self):
        """Test formatting empty metadata."""
        formatted = format_metadata_for_display({})
        
        assert formatted == {}


class TestSearchQuality:
    """Test search quality calculation."""
    
    def test_calculate_search_quality_score_high_quality(self):
        """Test quality score for high-quality results."""
        results = [
            {"content": "machine learning tutorial", "similarity": 0.95},
            {"content": "deep learning guide", "similarity": 0.88},
            {"content": "AI algorithms overview", "similarity": 0.82}
        ]
        
        score = calculate_search_quality_score(results, "machine learning")
        
        assert 0.7 <= score <= 1.0  # Should be high quality
    
    def test_calculate_search_quality_score_low_quality(self):
        """Test quality score for low-quality results."""
        results = [
            {"content": "unrelated content", "similarity": 0.3},
            {"content": "another unrelated topic", "similarity": 0.25},
            {"content": "completely different", "similarity": 0.2}
        ]
        
        score = calculate_search_quality_score(results, "machine learning")
        
        assert 0.0 <= score <= 0.4  # Should be low quality
    
    def test_calculate_search_quality_score_empty_results(self):
        """Test quality score for empty results."""
        score = calculate_search_quality_score([], "machine learning")
        
        assert score == 0.0
    
    def test_calculate_search_quality_score_position_weight(self):
        """Test that position affects quality score."""
        # Same similarity but different positions
        results_good_order = [
            {"content": "machine learning tutorial", "similarity": 0.9},
            {"content": "unrelated content", "similarity": 0.3}
        ]
        
        results_bad_order = [
            {"content": "unrelated content", "similarity": 0.3},
            {"content": "machine learning tutorial", "similarity": 0.9}
        ]
        
        score_good = calculate_search_quality_score(results_good_order, "machine learning")
        score_bad = calculate_search_quality_score(results_bad_order, "machine learning")
        
        # Good order should have higher score due to position weighting
        assert score_good > score_bad


class TestSearchResultProcessing:
    """Test comprehensive search result processing."""
    
    def test_process_search_results_with_snippets(self):
        """Test processing results with snippet extraction."""
        results = [
            {
                "content": "This is a long piece of content about machine learning algorithms and their applications in data science.",
                "url": "https://example.com/ml",
                "similarity": 0.9,
                "metadata": {"content_type": "tutorial"}
            }
        ]
        
        processed = process_search_results(results, "machine learning", include_snippets=True)
        
        assert len(processed) == 1
        assert "snippet" in processed[0]
        assert "machine learning" in processed[0]["snippet"]
        assert "relevance_indicators" in processed[0]
        assert "formatted_metadata" in processed[0]
    
    def test_process_search_results_without_snippets(self):
        """Test processing results without snippets."""
        results = [
            {
                "content": "Test content",
                "similarity": 0.8,
                "metadata": {"type": "test"}
            }
        ]
        
        processed = process_search_results(results, "test", include_snippets=False)
        
        assert "snippet" not in processed[0]
        assert "relevance_indicators" in processed[0]
    
    def test_process_search_results_with_highlighting(self):
        """Test processing with query term highlighting."""
        results = [
            {
                "content": "This content discusses machine learning and artificial intelligence topics.",
                "similarity": 0.85
            }
        ]
        
        processed = process_search_results(results, "machine learning")
        
        assert "highlighted_snippet" in processed[0]
        # Should contain highlighted terms
        highlighted = processed[0]["highlighted_snippet"]
        assert "**machine**" in highlighted or "**learning**" in highlighted
    
    def test_process_search_results_empty_list(self):
        """Test processing empty results list."""
        processed = process_search_results([], "test query")
        
        assert processed == []


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_rerank_results_with_missing_content(self):
        """Test reranking when results have missing content fields."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8]
        
        results = [{"summary": "Has summary but no content", "similarity": 0.7}]
        
        reranked = rerank_results(results, "test query", mock_model)
        
        # Should handle missing content gracefully
        assert len(reranked) == 1
    
    def test_filter_search_results_with_missing_fields(self):
        """Test filtering with results missing expected fields."""
        results = [
            {"content": "Complete result", "similarity": 0.8, "source_id": "example.com"},
            {"content": "Missing similarity"},  # No similarity field
            {"similarity": 0.7}  # No content field
        ]
        
        filtered = filter_search_results(results, min_similarity=0.5)
        
        # Should handle missing fields gracefully
        assert len(filtered) >= 1
    
    def test_highlight_query_terms_with_special_regex_chars(self):
        """Test highlighting with special regex characters in query."""
        text = "Test content with [brackets] and (parentheses) and +plus signs."
        query_terms = {"[brackets]", "(parentheses)", "+plus"}
        
        # Should not crash due to regex special characters
        highlighted = highlight_query_terms(text, query_terms)
        
        assert isinstance(highlighted, str)
    
    def test_extract_relevant_snippet_unicode_content(self):
        """Test snippet extraction with Unicode content."""
        content = "This content has émojis 🚀 and special characters like café and naïve."
        
        snippet = extract_relevant_snippet(content, "émojis", max_length=50)
        
        assert "émojis 🚀" in snippet
        assert isinstance(snippet, str)