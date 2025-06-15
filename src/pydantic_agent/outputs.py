"""
Structured output models for Pydantic AI agents.

This module defines Pydantic models that ensure type-safe, validated responses
from agent operations. These models mirror the JSON structure returned by the
existing MCP tools while adding additional metadata for workflow tracking.
"""

from datetime import datetime
from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field


class CrawlResult(BaseModel):
    """
    Structured output for web crawling operations.
    
    This model represents the result of crawling one or more web pages,
    including metadata about the crawling process and storage operations.
    """
    success: bool = Field(description="Whether the crawling operation succeeded")
    strategy_used: Literal["single", "smart", "recursive"] = Field(
        description="The crawling strategy that was employed"
    )
    pages_crawled: int = Field(description="Number of pages successfully crawled", ge=0)
    chunks_stored: int = Field(description="Number of content chunks stored in database", ge=0)
    code_examples_stored: int = Field(description="Number of code examples extracted and stored", ge=0)
    sources_updated: List[str] = Field(description="List of source domains that were updated")
    urls_processed: List[str] = Field(description="List of URLs that were processed")
    total_content_length: int = Field(description="Total character count of all content", ge=0)
    processing_time_seconds: float = Field(description="Time taken for the crawling operation", ge=0.0)
    summary: str = Field(description="Human-readable summary of the crawling results")
    error_details: Optional[str] = Field(default=None, description="Error details if operation failed")


class RAGResult(BaseModel):
    """
    Structured output for RAG query operations.
    
    This model represents the result of semantic search and content retrieval
    operations, including the query, results, and metadata about the search process.
    """
    success: bool = Field(description="Whether the RAG query succeeded")
    query: str = Field(description="The original query that was processed")
    search_mode: Literal["vector", "hybrid", "code"] = Field(
        description="The search mode that was used"
    )
    results_found: int = Field(description="Number of relevant results found", ge=0)
    sources_used: List[str] = Field(description="List of source domains that contributed results")
    answer: str = Field(description="The synthesized answer based on retrieved content")
    confidence_score: Optional[float] = Field(
        default=None, 
        description="Confidence score for the answer (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    retrieved_chunks: List[Dict[str, Any]] = Field(
        description="The actual content chunks that were retrieved"
    )
    processing_time_seconds: float = Field(description="Time taken for the RAG operation", ge=0.0)
    reranking_applied: bool = Field(default=False, description="Whether reranking was applied to results")
    error_details: Optional[str] = Field(default=None, description="Error details if operation failed")


class WorkflowStep(BaseModel):
    """
    Represents a single step in a multi-step workflow.
    """
    step_name: str = Field(description="Name of the workflow step")
    step_type: Literal["crawl", "rag", "analysis", "synthesis"] = Field(
        description="Type of operation performed in this step"
    )
    started_at: datetime = Field(description="When this step started")
    completed_at: Optional[datetime] = Field(default=None, description="When this step completed")
    success: bool = Field(description="Whether this step succeeded")
    output_summary: str = Field(description="Summary of what this step produced")
    error_details: Optional[str] = Field(default=None, description="Error details if step failed")


class WorkflowResult(BaseModel):
    """
    Structured output for complete multi-step workflows.
    
    This model represents the result of complex workflows that combine
    multiple operations like crawling, analysis, and querying.
    """
    success: bool = Field(description="Whether the entire workflow succeeded")
    workflow_type: str = Field(description="Type of workflow that was executed")
    started_at: datetime = Field(description="When the workflow started")
    completed_at: Optional[datetime] = Field(default=None, description="When the workflow completed")
    total_time_seconds: float = Field(description="Total time for workflow completion", ge=0.0)
    steps_completed: List[WorkflowStep] = Field(description="List of completed workflow steps")
    final_result: Union[CrawlResult, RAGResult, str] = Field(
        description="The final output of the workflow"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the workflow execution"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for follow-up actions or improvements"
    )
    error_details: Optional[str] = Field(default=None, description="Error details if workflow failed")


class SourceAnalysis(BaseModel):
    """
    Analysis of available sources in the database.
    """
    source_id: str = Field(description="Identifier for the source")
    summary: str = Field(description="AI-generated summary of the source content")
    total_words: int = Field(description="Total word count for this source", ge=0)
    last_updated: datetime = Field(description="When this source was last updated")
    content_quality: Literal["high", "medium", "low"] = Field(
        description="Assessed quality of the content"
    )
    recommended_for: List[str] = Field(
        description="Types of queries this source is recommended for"
    )


class ContentDiscoveryResult(BaseModel):
    """
    Result of content discovery and source analysis operations.
    """
    success: bool = Field(description="Whether the discovery operation succeeded")
    total_sources: int = Field(description="Total number of sources discovered", ge=0)
    sources_analyzed: List[SourceAnalysis] = Field(description="Detailed analysis of each source")
    coverage_gaps: List[str] = Field(description="Identified gaps in content coverage")
    recommendations: List[str] = Field(description="Recommendations for additional content")
    processing_time_seconds: float = Field(description="Time taken for discovery", ge=0.0)
    error_details: Optional[str] = Field(default=None, description="Error details if operation failed")
