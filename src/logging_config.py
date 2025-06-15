"""
Centralized logging configuration using Logfire for comprehensive observability.

This module provides unified logging setup for both MCP server operations and
Pydantic AI agent interactions, following best practices for structured logging.
"""

import os
import sys
from typing import Any, Dict, Optional
from functools import wraps
import time
import asyncio

try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False


def configure_logfire() -> bool:
    """
    Configure logfire with environment settings.
    
    Returns:
        bool: True if logfire is configured successfully, False otherwise
    """
    if not LOGFIRE_AVAILABLE:
        print("Warning: logfire not available, falling back to console logging")
        return False
        
    try:
        # Configure logfire with environment variables
        logfire_token = os.getenv('LOGFIRE_TOKEN')
        if not logfire_token:
            print("Warning: LOGFIRE_TOKEN not set, falling back to console logging")
            return False
            
        # Configure logfire
        logfire.configure(
            token=logfire_token,
            service_name="crawl4ai-mcp-server",
            service_version="0.1.0",
            environment=os.getenv('ENVIRONMENT', 'development'),
        )
        
        print("✓ Logfire configured successfully")
        return True
        
    except Exception as e:
        print(f"Warning: Failed to configure logfire: {e}")
        return False


def get_logger():
    """
    Get configured logger instance.
    
    Returns:
        Logger instance (logfire or fallback)
    """
    if LOGFIRE_AVAILABLE and configure_logfire():
        return logfire
    else:
        # Fallback to basic console logging
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("crawl4ai-mcp")


# Global logger instance
logger = get_logger()


def log_mcp_tool_execution(tool_name: str):
    """
    Enhanced decorator for logging MCP tool execution with comprehensive metrics and context.

    Args:
        tool_name: Name of the MCP tool being executed
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            # Extract parameters for logging (excluding sensitive data)
            safe_kwargs = {k: v for k, v in kwargs.items()
                          if not any(sensitive in k.lower() for sensitive in ['token', 'key', 'password'])}

            # Extract context information if available
            ctx = args[0] if args and hasattr(args[0], 'request_context') else None

            if LOGFIRE_AVAILABLE:
                with logger.span(f"mcp_tool_{tool_name}") as span:
                    span.set_attributes({
                        'tool.name': tool_name,
                        'tool.parameters': safe_kwargs,
                        'tool.args_count': len(args),
                        'execution.start_time': start_time,
                        'tool.category': _get_tool_category(tool_name),
                        'tool.expected_duration': _get_expected_duration(tool_name)
                    })

                    # Add context-specific attributes
                    if ctx:
                        span.set_attributes({
                            'context.has_crawler': hasattr(ctx.request_context.lifespan_context, 'crawler'),
                            'context.has_postgres': hasattr(ctx.request_context.lifespan_context, 'postgres_pool'),
                            'context.has_reranking': hasattr(ctx.request_context.lifespan_context, 'reranking_model')
                        })

                    try:
                        result = await func(*args, **kwargs)
                        execution_time = time.time() - start_time

                        # Parse result for enhanced logging
                        result_data = _parse_tool_result(result, tool_name)

                        # Log successful execution with enhanced metrics
                        span.set_attributes({
                            'execution.success': True,
                            'execution.duration_seconds': execution_time,
                            'result.type': type(result).__name__,
                            'result.size_bytes': len(str(result)),
                            **result_data
                        })

                        # Performance analysis
                        expected_duration = _get_expected_duration(tool_name)
                        if execution_time > expected_duration * 2:
                            span.set_attribute('performance.slow_execution', True)
                            logger.warning(f"MCP tool {tool_name} executed slower than expected",
                                         execution_time=execution_time,
                                         expected_duration=expected_duration,
                                         performance_ratio=execution_time / expected_duration)

                        logger.info(f"MCP tool {tool_name} completed successfully",
                                  execution_time=execution_time,
                                  result_type=type(result).__name__,
                                  result_summary=result_data)

                        return result

                    except Exception as e:
                        execution_time = time.time() - start_time

                        # Enhanced error logging
                        error_context = _analyze_error_context(e, tool_name, safe_kwargs)

                        span.set_attributes({
                            'execution.success': False,
                            'execution.duration_seconds': execution_time,
                            'error.type': type(e).__name__,
                            'error.message': str(e),
                            'error.category': error_context['category'],
                            'error.severity': error_context['severity'],
                            'error.recoverable': error_context['recoverable']
                        })

                        logger.error(f"MCP tool {tool_name} failed",
                                   error=str(e),
                                   error_type=type(e).__name__,
                                   execution_time=execution_time,
                                   error_context=error_context,
                                   tool_parameters=safe_kwargs)

                        raise
            else:
                # Enhanced fallback logging
                try:
                    logger.info(f"Executing MCP tool: {tool_name}", parameters=safe_kwargs)
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    result_summary = _parse_tool_result(result, tool_name)
                    logger.info(f"MCP tool {tool_name} completed in {execution_time:.2f}s",
                              result_summary=result_summary)
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    error_context = _analyze_error_context(e, tool_name, safe_kwargs)
                    logger.error(f"MCP tool {tool_name} failed after {execution_time:.2f}s: {e}",
                               error_context=error_context)
                    raise

        return wrapper
    return decorator


def _get_tool_category(tool_name: str) -> str:
    """Categorize MCP tools for better organization."""
    if tool_name in ['smart_crawl_url', 'crawl_single_page']:
        return 'content_acquisition'
    elif tool_name in ['perform_rag_query', 'search_code_examples']:
        return 'content_search'
    elif tool_name == 'get_available_sources':
        return 'content_discovery'
    else:
        return 'utility'


def _get_expected_duration(tool_name: str) -> float:
    """Get expected duration for performance analysis."""
    durations = {
        'crawl_single_page': 5.0,
        'smart_crawl_url': 15.0,
        'get_available_sources': 0.5,
        'perform_rag_query': 2.0,
        'search_code_examples': 2.0
    }
    return durations.get(tool_name, 5.0)


def _parse_tool_result(result: Any, tool_name: str) -> Dict[str, Any]:
    """Parse tool result for enhanced logging."""
    result_data = {}

    try:
        if isinstance(result, str):
            import json
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                result_data['result.success'] = parsed.get('success', False)

                # Tool-specific result parsing
                if tool_name in ['smart_crawl_url', 'crawl_single_page']:
                    result_data.update({
                        'result.pages_crawled': parsed.get('pages_crawled', 0),
                        'result.chunks_stored': parsed.get('chunks_stored', 0),
                        'result.code_examples_stored': parsed.get('code_examples_stored', 0),
                        'result.content_length': parsed.get('content_length', 0)
                    })
                elif tool_name in ['perform_rag_query', 'search_code_examples']:
                    result_data.update({
                        'result.matches_found': parsed.get('count', 0),
                        'result.search_mode': parsed.get('search_mode', 'unknown'),
                        'result.reranking_applied': parsed.get('reranking_applied', False)
                    })
                elif tool_name == 'get_available_sources':
                    result_data.update({
                        'result.sources_count': parsed.get('count', 0)
                    })
    except Exception:
        # If parsing fails, just record basic info
        result_data['result.parse_error'] = True

    return result_data


def _analyze_error_context(error: Exception, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze error context for better debugging."""
    error_msg = str(error).lower()
    error_type = type(error).__name__

    # Categorize error
    if 'network' in error_msg or 'connection' in error_msg or 'timeout' in error_msg:
        category = 'network'
        severity = 'medium'
        recoverable = True
    elif 'database' in error_msg or 'postgres' in error_msg or 'sql' in error_msg:
        category = 'database'
        severity = 'high'
        recoverable = True
    elif 'permission' in error_msg or 'access' in error_msg or 'forbidden' in error_msg:
        category = 'permission'
        severity = 'high'
        recoverable = False
    elif 'validation' in error_msg or 'invalid' in error_msg:
        category = 'validation'
        severity = 'low'
        recoverable = False
    else:
        category = 'unknown'
        severity = 'medium'
        recoverable = True

    return {
        'category': category,
        'severity': severity,
        'recoverable': recoverable,
        'error_type': error_type,
        'tool_name': tool_name,
        'has_url_param': 'url' in parameters,
        'has_query_param': 'query' in parameters
    }


def log_agent_interaction(agent_type: str):
    """
    Decorator for logging Pydantic AI agent interactions.
    
    Args:
        agent_type: Type of agent (crawl, rag, workflow)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Extract prompt if available
            prompt = kwargs.get('prompt', args[1] if len(args) > 1 else 'Unknown')
            prompt_preview = prompt[:100] + "..." if len(str(prompt)) > 100 else str(prompt)
            
            if LOGFIRE_AVAILABLE:
                with logger.span(f"agent_{agent_type}_interaction") as span:
                    span.set_attributes({
                        'agent.type': agent_type,
                        'agent.prompt_preview': prompt_preview,
                        'agent.prompt_length': len(str(prompt)),
                        'interaction.start_time': start_time,
                    })
                    
                    try:
                        result = await func(*args, **kwargs)
                        execution_time = time.time() - start_time
                        
                        # Extract result information
                        result_summary = {}
                        if hasattr(result, 'data'):
                            if hasattr(result.data, 'summary'):
                                result_summary['summary'] = result.data.summary[:200]
                            if hasattr(result.data, 'confidence_score'):
                                result_summary['confidence_score'] = result.data.confidence_score
                        
                        span.set_attributes({
                            'interaction.success': True,
                            'interaction.duration_seconds': execution_time,
                            'result.type': type(result).__name__,
                            **{f'result.{k}': v for k, v in result_summary.items()}
                        })
                        
                        logger.info(f"Agent {agent_type} interaction completed", 
                                  execution_time=execution_time, 
                                  prompt_length=len(str(prompt)),
                                  result_type=type(result).__name__)
                        
                        return result
                        
                    except Exception as e:
                        execution_time = time.time() - start_time
                        
                        span.set_attributes({
                            'interaction.success': False,
                            'interaction.duration_seconds': execution_time,
                            'error.type': type(e).__name__,
                            'error.message': str(e),
                        })
                        
                        logger.error(f"Agent {agent_type} interaction failed", 
                                   error=str(e), error_type=type(e).__name__,
                                   execution_time=execution_time)
                        
                        raise
            else:
                # Fallback logging
                try:
                    logger.info(f"Agent {agent_type} processing prompt: {prompt_preview}")
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    logger.info(f"Agent {agent_type} completed in {execution_time:.2f}s")
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(f"Agent {agent_type} failed after {execution_time:.2f}s: {e}")
                    raise
                    
        return wrapper
    return decorator


def log_database_operation(operation: str):
    """
    Decorator for logging database operations.
    
    Args:
        operation: Type of database operation (insert, query, update, etc.)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            if LOGFIRE_AVAILABLE:
                with logger.span(f"db_{operation}") as span:
                    span.set_attributes({
                        'db.operation': operation,
                        'db.start_time': start_time,
                    })
                    
                    try:
                        result = await func(*args, **kwargs)
                        execution_time = time.time() - start_time
                        
                        # Log operation metrics
                        result_count = len(result) if isinstance(result, (list, tuple)) else 1
                        
                        span.set_attributes({
                            'db.success': True,
                            'db.duration_seconds': execution_time,
                            'db.result_count': result_count,
                        })
                        
                        logger.info(f"Database {operation} completed", 
                                  execution_time=execution_time, result_count=result_count)
                        
                        return result
                        
                    except Exception as e:
                        execution_time = time.time() - start_time
                        
                        span.set_attributes({
                            'db.success': False,
                            'db.duration_seconds': execution_time,
                            'db.error.type': type(e).__name__,
                            'db.error.message': str(e),
                        })
                        
                        logger.error(f"Database {operation} failed", 
                                   error=str(e), execution_time=execution_time)
                        
                        raise
            else:
                # Fallback logging
                try:
                    logger.info(f"Database operation: {operation}")
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    logger.info(f"Database {operation} completed in {execution_time:.2f}s")
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(f"Database {operation} failed after {execution_time:.2f}s: {e}")
                    raise
                    
        return wrapper
    return decorator


def log_crawling_operation(url: str, operation_type: str):
    """
    Log crawling operation with URL and metrics.
    
    Args:
        url: URL being crawled
        operation_type: Type of crawling operation
    """
    if LOGFIRE_AVAILABLE:
        logger.info("Crawling operation started", 
                   url=url, operation_type=operation_type)
    else:
        logger.info(f"Crawling {operation_type}: {url}")


def log_rag_query(query: str, match_count: int, source: Optional[str] = None):
    """
    Log RAG query operation.
    
    Args:
        query: Search query
        match_count: Number of matches requested
        source: Optional source filter
    """
    if LOGFIRE_AVAILABLE:
        logger.info("RAG query executed", 
                   query_preview=query[:100], query_length=len(query),
                   match_count=match_count, source=source)
    else:
        logger.info(f"RAG query: {query[:100]}... (count: {match_count}, source: {source})")


def log_system_startup(service_name: str, version: str):
    """
    Log system startup information.
    
    Args:
        service_name: Name of the service starting up
        version: Version of the service
    """
    if LOGFIRE_AVAILABLE:
        logger.info("System startup", 
                   service=service_name, version=version, 
                   environment=os.getenv('ENVIRONMENT', 'development'))
    else:
        logger.info(f"Starting {service_name} v{version}")