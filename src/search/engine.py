"""
Search engine utilities for vector and hybrid search operations.

This module provides search result processing, reranking capabilities,
and advanced search strategies for the Crawl4AI RAG system.
"""
from typing import List, Dict, Any, Optional
import re

__all__ = [
    "rerank_results",
    "filter_search_results",
    "enhance_search_query",
    "process_search_results"
]


def rerank_results(results: List[Dict[str, Any]], query: str, model=None) -> List[Dict[str, Any]]:
    """
    Rerank search results using a cross-encoder model for improved relevance.
    
    Args:
        results: List of search results to rerank
        query: The original search query
        model: Optional cross-encoder model for reranking
        
    Returns:
        Reranked list of search results
    """
    if not results or not model:
        return results
    
    try:
        # Prepare query-document pairs for reranking
        pairs = []
        for result in results:
            # Use content or summary for reranking
            doc_text = result.get('content', result.get('summary', ''))
            pairs.append([query, doc_text[:512]])  # Limit text length for efficiency
        
        # Get reranking scores
        scores = model.predict(pairs)
        
        # Add reranking scores to results and sort
        for i, result in enumerate(results):
            result['rerank_score'] = float(scores[i])
        
        # Sort by reranking score (higher is better)
        reranked_results = sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        return reranked_results
    
    except Exception as e:
        print(f"Error during reranking: {e}. Returning original results.")
        return results


def filter_search_results(
    results: List[Dict[str, Any]], 
    filters: Optional[Dict[str, Any]] = None,
    min_similarity: float = 0.0,
    max_results: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Filter search results based on various criteria.
    
    Args:
        results: List of search results to filter
        filters: Dictionary of filter criteria
        min_similarity: Minimum similarity score threshold
        max_results: Maximum number of results to return
        
    Returns:
        Filtered list of search results
    """
    if not results:
        return results
    
    filtered_results = results.copy()
    
    # Apply similarity threshold
    if min_similarity > 0:
        filtered_results = [
            r for r in filtered_results 
            if r.get('similarity', 0) >= min_similarity
        ]
    
    # Apply metadata filters
    if filters:
        for key, value in filters.items():
            if key == 'source':
                filtered_results = [
                    r for r in filtered_results 
                    if r.get('source_id', '').lower() == value.lower()
                ]
            elif key == 'language':
                filtered_results = [
                    r for r in filtered_results 
                    if r.get('metadata', {}).get('language', '').lower() == value.lower()
                ]
            elif key == 'content_type':
                filtered_results = [
                    r for r in filtered_results 
                    if r.get('metadata', {}).get('content_type', '').lower() == value.lower()
                ]
    
    # Limit number of results
    if max_results and max_results > 0:
        filtered_results = filtered_results[:max_results]
    
    return filtered_results


def enhance_search_query(query: str, search_type: str = 'general') -> str:
    """
    Enhance the search query for better retrieval results.
    
    Args:
        query: Original search query
        search_type: Type of search ('general', 'code', 'api', 'tutorial')
        
    Returns:
        Enhanced search query
    """
    enhanced_query = query.strip()
    
    if search_type == 'code':
        # For code searches, add context about looking for code examples
        if not any(word in enhanced_query.lower() for word in ['example', 'code', 'function', 'method']):
            enhanced_query = f"Code example for {enhanced_query}"
    
    elif search_type == 'api':
        # For API searches, add context about API documentation
        if not any(word in enhanced_query.lower() for word in ['api', 'endpoint', 'method']):
            enhanced_query = f"API documentation for {enhanced_query}"
    
    elif search_type == 'tutorial':
        # For tutorial searches, add context about learning
        if not any(word in enhanced_query.lower() for word in ['tutorial', 'guide', 'how to']):
            enhanced_query = f"Tutorial guide for {enhanced_query}"
    
    return enhanced_query


def process_search_results(
    results: List[Dict[str, Any]], 
    query: str,
    include_snippets: bool = True,
    snippet_length: int = 200
) -> List[Dict[str, Any]]:
    """
    Process and enhance search results with snippets and highlights.
    
    Args:
        results: Raw search results
        query: Original search query
        include_snippets: Whether to include content snippets
        snippet_length: Length of content snippets
        
    Returns:
        Processed search results with enhancements
    """
    processed_results = []
    
    # Create query terms for highlighting
    query_terms = set(query.lower().split())
    
    for result in results:
        processed_result = result.copy()
        
        # Add snippet if requested
        if include_snippets and 'content' in result:
            content = result['content']
            snippet = extract_relevant_snippet(content, query, snippet_length)
            processed_result['snippet'] = snippet
            
            # Add highlight information
            if query_terms:
                highlighted_snippet = highlight_query_terms(snippet, query_terms)
                processed_result['highlighted_snippet'] = highlighted_snippet
        
        # Add relevance indicators
        processed_result['relevance_indicators'] = analyze_relevance(result, query)
        
        # Format metadata for display
        if 'metadata' in result:
            processed_result['formatted_metadata'] = format_metadata_for_display(result['metadata'])
        
        processed_results.append(processed_result)
    
    return processed_results


def extract_relevant_snippet(content: str, query: str, max_length: int = 200) -> str:
    """
    Extract the most relevant snippet from content based on query.
    
    Args:
        content: Full content text
        query: Search query
        max_length: Maximum snippet length
        
    Returns:
        Relevant content snippet
    """
    if len(content) <= max_length:
        return content
    
    # Find the best position to extract snippet
    query_terms = query.lower().split()
    content_lower = content.lower()
    
    # Find positions of query terms
    positions = []
    for term in query_terms:
        pos = content_lower.find(term)
        if pos != -1:
            positions.append(pos)
    
    if positions:
        # Start snippet around the first found term
        start_pos = max(0, min(positions) - max_length // 3)
        end_pos = min(len(content), start_pos + max_length)
        
        # Adjust to word boundaries
        if start_pos > 0:
            start_pos = content.find(' ', start_pos) + 1
        if end_pos < len(content):
            end_pos = content.rfind(' ', start_pos, end_pos)
        
        snippet = content[start_pos:end_pos].strip()
        
        # Add ellipsis if needed
        if start_pos > 0:
            snippet = "..." + snippet
        if end_pos < len(content):
            snippet = snippet + "..."
            
        return snippet
    
    # Fallback: return beginning of content
    snippet = content[:max_length].strip()
    if len(content) > max_length:
        snippet += "..."
    return snippet


def highlight_query_terms(text: str, query_terms: set) -> str:
    """
    Highlight query terms in text using simple markup.
    
    Args:
        text: Text to highlight
        query_terms: Set of terms to highlight
        
    Returns:
        Text with highlighted terms
    """
    highlighted = text
    
    for term in query_terms:
        if len(term) > 2:  # Only highlight meaningful terms
            # Use case-insensitive replacement with highlighting
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(f"**{term}**", highlighted)
    
    return highlighted


def analyze_relevance(result: Dict[str, Any], query: str) -> Dict[str, Any]:
    """
    Analyze relevance indicators for a search result.
    
    Args:
        result: Search result dictionary
        query: Search query
        
    Returns:
        Dictionary of relevance indicators
    """
    indicators = {}
    
    # Similarity score indicator
    similarity = result.get('similarity', 0)
    if similarity > 0.8:
        indicators['similarity_level'] = 'high'
    elif similarity > 0.6:
        indicators['similarity_level'] = 'medium'
    else:
        indicators['similarity_level'] = 'low'
    
    # Query term coverage
    query_terms = set(query.lower().split())
    content = result.get('content', '').lower()
    
    found_terms = sum(1 for term in query_terms if term in content)
    coverage = found_terms / len(query_terms) if query_terms else 0
    indicators['query_coverage'] = coverage
    
    # Content type relevance
    metadata = result.get('metadata', {})
    content_type = metadata.get('content_type', 'unknown')
    indicators['content_type'] = content_type
    
    # Source reliability (based on source_id)
    source_id = result.get('source_id', '')
    if any(domain in source_id for domain in ['github.com', 'docs.', 'documentation']):
        indicators['source_reliability'] = 'high'
    else:
        indicators['source_reliability'] = 'medium'
    
    return indicators


def format_metadata_for_display(metadata: Dict[str, Any]) -> Dict[str, str]:
    """
    Format metadata for user-friendly display.
    
    Args:
        metadata: Raw metadata dictionary
        
    Returns:
        Formatted metadata for display
    """
    formatted = {}
    
    # Format common metadata fields
    if 'chunk_size' in metadata:
        formatted['Content Length'] = f"{metadata['chunk_size']} characters"
    
    if 'language' in metadata:
        formatted['Language'] = metadata['language'].title()
    
    if 'content_type' in metadata:
        formatted['Content Type'] = metadata['content_type'].replace('_', ' ').title()
    
    if 'word_count' in metadata:
        formatted['Word Count'] = str(metadata['word_count'])
    
    if 'contextual_embedding' in metadata:
        formatted['Enhanced Context'] = 'Yes' if metadata['contextual_embedding'] else 'No'
    
    return formatted


def calculate_search_quality_score(results: List[Dict[str, Any]], query: str) -> float:
    """
    Calculate an overall quality score for search results.
    
    Args:
        results: List of search results
        query: Original search query
        
    Returns:
        Quality score between 0 and 1
    """
    if not results:
        return 0.0
    
    total_score = 0.0
    
    for i, result in enumerate(results[:10]):  # Consider top 10 results
        # Base similarity score
        similarity = result.get('similarity', 0)
        
        # Position penalty (earlier results should be more relevant)
        position_weight = 1.0 / (i + 1)
        
        # Query coverage bonus
        query_terms = set(query.lower().split())
        content = result.get('content', '').lower()
        coverage = sum(1 for term in query_terms if term in content) / len(query_terms) if query_terms else 0
        
        # Calculate weighted score
        result_score = (similarity * 0.7 + coverage * 0.3) * position_weight
        total_score += result_score
    
    # Normalize by number of results considered
    average_score = total_score / min(len(results), 10)
    
    return min(1.0, average_score)  # Cap at 1.0