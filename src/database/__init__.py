"""Database operations package for PostgreSQL integration."""

from .operations import (
    get_postgres_connection,
    create_postgres_pool,
    add_documents_to_postgres,
    add_code_examples_to_postgres,
    update_source_info,
    search_documents,
    search_code_examples
)

__all__ = [
    "get_postgres_connection",
    "create_postgres_pool",
    "add_documents_to_postgres", 
    "add_code_examples_to_postgres",
    "update_source_info",
    "search_documents",
    "search_code_examples"
]