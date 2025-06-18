"""
Centralized logging configuration using Logfire for comprehensive observability.

This module provides unified logging setup for both MCP server operations and
Pydantic AI agent interactions, following best practices for structured logging.
"""

import os
from typing import Optional
from functools import wraps
import time

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

# Helper functions for compatibility between logfire and standard logging
def is_logfire_logger(logger_instance) -> bool:
    """Check if logger instance supports logfire structured logging."""
    return LOGFIRE_AVAILABLE and hasattr(logger_instance, 'span') and hasattr(logger_instance, 'info')

def log_info(message: str, **kwargs):
    """Log info message with compatibility for both logfire and standard logging."""
    if is_logfire_logger(logger):
        logger.info(message, **kwargs)
    else:
        # Format structured data for standard logging
        extra_info = ", ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
        full_message = f"{message}" + (f" ({extra_info})" if extra_info else "")
        logger.info(full_message)

def log_error(message: str, **kwargs):
    """Log error message with compatibility for both logfire and standard logging."""
    if is_logfire_logger(logger):
        logger.error(message, **kwargs)
    else:
        # Format structured data for standard logging
        extra_info = ", ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
        full_message = f"{message}" + (f" ({extra_info})" if extra_info else "")
        logger.error(full_message)

def log_warning(message: str, **kwargs):
    """Log warning message with compatibility for both logfire and standard logging."""
    if is_logfire_logger(logger):
        logger.warning(message, **kwargs)
    else:
        # Format structured data for standard logging
        extra_info = ", ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
        full_message = f"{message}" + (f" ({extra_info})" if extra_info else "")
        logger.warning(full_message)


def log_mcp_tool_execution(tool_name: str):
    """
    Simple decorator for logging MCP tool execution with basic metrics.

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

            if is_logfire_logger(logger):
                with logger.span(f"mcp_tool_{tool_name}") as span:
                    span.set_attributes({
                        'tool.name': tool_name,
                        'tool.parameters': safe_kwargs,
                        'tool.args_count': len(args),
                        'execution.start_time': start_time,
                    })

                    try:
                        result = await func(*args, **kwargs)
                        execution_time = time.time() - start_time

                        # Log successful execution
                        span.set_attributes({
                            'execution.success': True,
                            'execution.duration_seconds': execution_time,
                            'result.type': type(result).__name__,
                        })

                        # Log result summary if it's a dict
                        if isinstance(result, dict):
                            if 'success' in result:
                                span.set_attribute('result.success', result['success'])
                            if 'chunks_stored' in result:
                                span.set_attribute('result.chunks_stored', result['chunks_stored'])
                            if 'pages_crawled' in result:
                                span.set_attribute('result.pages_crawled', result['pages_crawled'])

                        log_info(f"MCP tool {tool_name} completed successfully",
                                execution_time=execution_time, result_type=type(result).__name__)

                        return result

                    except Exception as e:
                        execution_time = time.time() - start_time

                        # Log error
                        span.set_attributes({
                            'execution.success': False,
                            'execution.duration_seconds': execution_time,
                            'error.type': type(e).__name__,
                            'error.message': str(e),
                        })

                        log_error(f"MCP tool {tool_name} failed",
                                 error=str(e), error_type=type(e).__name__,
                                 execution_time=execution_time)

                        raise
            else:
                # Fallback logging
                try:
                    logger.info(f"Executing MCP tool: {tool_name}")
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    logger.info(f"MCP tool {tool_name} completed in {execution_time:.2f}s")
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(f"MCP tool {tool_name} failed after {execution_time:.2f}s: {e}")
                    raise

        return wrapper
    return decorator





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
            
            if is_logfire_logger(logger):
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
                        
                        log_info(f"Agent {agent_type} interaction completed", 
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
                        
                        log_error(f"Agent {agent_type} interaction failed", 
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
            
            if is_logfire_logger(logger):
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
                        
                        log_info(f"Database {operation} completed", 
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
                        
                        log_error(f"Database {operation} failed", 
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
    log_info("Crawling operation started", url=url, operation_type=operation_type)


def log_rag_query(query: str, match_count: int, source: Optional[str] = None):
    """
    Log RAG query operation.
    
    Args:
        query: Search query
        match_count: Number of matches requested
        source: Optional source filter
    """
    log_info("RAG query executed", 
             query_preview=query[:100], query_length=len(query),
             match_count=match_count, source=source)


def log_system_startup(service_name: str, version: str):
    """
    Log system startup information.
    
    Args:
        service_name: Name of the service starting up
        version: Version of the service
    """
    log_info("System startup", 
             service=service_name, version=version, 
             environment=os.getenv('ENVIRONMENT', 'development'))