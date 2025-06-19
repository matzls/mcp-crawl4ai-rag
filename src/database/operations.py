"""
Database operations for PostgreSQL integration.

This module handles all PostgreSQL database operations including connection management,
document storage, code example storage, and source metadata management.
"""
import os
import json
import time
import asyncpg
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import getpass

__all__ = [
    "get_postgres_connection",
    "create_postgres_pool", 
    "add_documents_to_postgres",
    "add_code_examples_to_postgres",
    "update_source_info",
    "search_documents",
    "search_code_examples"
]


async def get_postgres_connection() -> asyncpg.Connection:
    """
    Get a PostgreSQL connection using environment variables.
    
    Returns:
        PostgreSQL connection instance
    """
    database_url = os.getenv("DATABASE_URL")
    
    if database_url:
        return await asyncpg.connect(database_url)
    
    # Fallback to individual components
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "crawl4ai_rag")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD")
    
    if not password:
        raise ValueError("Either DATABASE_URL or POSTGRES_PASSWORD must be set in environment variables")
    
    return await asyncpg.connect(
        host=host,
        port=int(port),
        database=database,
        user=user,
        password=password
    )


async def create_postgres_pool() -> asyncpg.Pool:
    """
    Create a PostgreSQL connection pool.
    
    Returns:
        PostgreSQL connection pool
    """
    database_url = os.getenv("DATABASE_URL")
    
    if database_url:
        # Expand $(whoami) if present in DATABASE_URL
        if "$(whoami)" in database_url:
            current_user = getpass.getuser()
            database_url = database_url.replace("$(whoami)", current_user)
        return await asyncpg.create_pool(database_url, min_size=2, max_size=10)
    
    # Fallback to individual components
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "crawl4ai_rag")
    user = os.getenv("POSTGRES_USER", getpass.getuser())  # Default to current user
    password = os.getenv("POSTGRES_PASSWORD")  # None is acceptable for trust auth
    
    return await asyncpg.create_pool(
        host=host,
        port=int(port),
        database=database,
        user=user,
        password=password,
        min_size=2,
        max_size=10
    )


def _embedding_to_vector_string(embedding: List[float]) -> str:
    """
    Convert a Python list embedding to PostgreSQL vector string format.
    
    Args:
        embedding: List of floats representing the embedding
        
    Returns:
        String representation suitable for PostgreSQL vector type
    """
    return '[' + ','.join(map(str, embedding)) + ']'


async def add_documents_to_postgres(
    pool: asyncpg.Pool, 
    urls: List[str], 
    chunk_numbers: List[int],
    contents: List[str], 
    metadatas: List[Dict[str, Any]],
    embeddings: List[List[float]],
    batch_size: int = 20
) -> None:
    """
    Add documents to the PostgreSQL crawled_pages table in batches.
    Deletes existing records with the same URLs before inserting to prevent duplicates.
    
    Args:
        pool: PostgreSQL connection pool
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        embeddings: List of embeddings for each document
        batch_size: Size of each batch for insertion
    """
    
    # Get unique URLs to delete existing records
    unique_urls = list(set(urls))
    
    # Delete existing records for these URLs
    async with pool.acquire() as conn:
        try:
            if unique_urls:
                # Delete all records with matching URLs
                await conn.execute(
                    "DELETE FROM crawled_pages WHERE url = ANY($1)",
                    unique_urls
                )
        except Exception as e:
            print(f"Batch delete failed: {e}. Trying one-by-one deletion as fallback.")
            # Fallback: delete records one by one
            for url in unique_urls:
                try:
                    await conn.execute("DELETE FROM crawled_pages WHERE url = $1", url)
                except Exception as inner_e:
                    print(f"Error deleting record for URL {url}: {inner_e}")
                    # Continue with the next URL even if one fails
    
    # Process in batches to avoid memory issues
    for i in range(0, len(contents), batch_size):
        batch_end = min(i + batch_size, len(contents))
        
        # Get batch slices
        batch_urls = urls[i:batch_end]
        batch_chunk_numbers = chunk_numbers[i:batch_end]
        batch_contents = contents[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]
        batch_embeddings = embeddings[i:batch_end]
        
        batch_data = []
        for j in range(len(batch_contents)):
            # Extract metadata fields
            chunk_size = len(batch_contents[j])
            
            # Extract source_id from URL
            parsed_url = urlparse(batch_urls[j])
            source_id = parsed_url.netloc or parsed_url.path
            
            # Prepare data for insertion
            data = {
                "url": batch_urls[j],
                "chunk_number": batch_chunk_numbers[j],
                "content": batch_contents[j],
                "metadata": {
                    "chunk_size": chunk_size,
                    **batch_metadatas[j]
                },
                "source_id": source_id,
                "embedding": batch_embeddings[j]
            }
            
            batch_data.append(data)
        
        # Insert batch into PostgreSQL with retry logic
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay
        
        for retry in range(max_retries):
            try:
                async with pool.acquire() as conn:
                    # Prepare batch insert query
                    insert_query = """
                        INSERT INTO crawled_pages (url, chunk_number, content, metadata, source_id, embedding)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """
                    
                    # Insert all records in the batch
                    for data in batch_data:
                        await conn.execute(
                            insert_query,
                            data["url"],
                            data["chunk_number"],
                            data["content"],
                            json.dumps(data["metadata"]),
                            data["source_id"],
                            _embedding_to_vector_string(data["embedding"])
                        )
                # Success - break out of retry loop
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error inserting batch into PostgreSQL (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")
                    # Optionally, try inserting records one by one as a last resort
                    print("Attempting to insert records individually...")
                    successful_inserts = 0
                    async with pool.acquire() as conn:
                        for record in batch_data:
                            try:
                                await conn.execute(
                                    insert_query,
                                    record["url"],
                                    record["chunk_number"],
                                    record["content"],
                                    json.dumps(record["metadata"]),
                                    record["source_id"],
                                    embedding_to_vector_string(record["embedding"])
                                )
                                successful_inserts += 1
                            except Exception as individual_error:
                                print(f"Failed to insert individual record for URL {record['url']}: {individual_error}")
                    
                    if successful_inserts > 0:
                        print(f"Successfully inserted {successful_inserts}/{len(batch_data)} records individually")


async def add_code_examples_to_postgres(
    pool: asyncpg.Pool,
    urls: List[str],
    chunk_numbers: List[int],
    code_examples: List[str],
    summaries: List[str],
    metadatas: List[Dict[str, Any]],
    embeddings: List[List[float]],
    batch_size: int = 20
):
    """
    Add code examples to the PostgreSQL code_examples table in batches.
    
    Args:
        pool: PostgreSQL connection pool
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        code_examples: List of code example contents
        summaries: List of code example summaries
        metadatas: List of metadata dictionaries
        embeddings: List of embeddings for each code example
        batch_size: Size of each batch for insertion
    """
    
    if not urls:
        return
        
    # Delete existing records for these URLs
    unique_urls = list(set(urls))
    async with pool.acquire() as conn:
        for url in unique_urls:
            try:
                await conn.execute("DELETE FROM code_examples WHERE url = $1", url)
            except Exception as e:
                print(f"Error deleting existing code examples for {url}: {e}")
    
    # Process in batches
    total_items = len(urls)
    for i in range(0, total_items, batch_size):
        batch_end = min(i + batch_size, total_items)
        
        # Prepare batch data
        batch_data = []
        for j in range(i, batch_end):
            # Extract source_id from URL
            parsed_url = urlparse(urls[j])
            source_id = parsed_url.netloc or parsed_url.path
            
            batch_data.append({
                'url': urls[j],
                'chunk_number': chunk_numbers[j],
                'content': code_examples[j],
                'summary': summaries[j],
                'metadata': metadatas[j],
                'source_id': source_id,
                'embedding': embeddings[j]
            })
        
        # Insert batch into PostgreSQL with retry logic
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay
        
        for retry in range(max_retries):
            try:
                async with pool.acquire() as conn:
                    # Prepare batch insert query
                    insert_query = """
                        INSERT INTO code_examples (url, chunk_number, content, summary, metadata, source_id, embedding)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """
                    
                    # Insert all records in the batch
                    for data in batch_data:
                        await conn.execute(
                            insert_query,
                            data['url'],
                            data['chunk_number'],
                            data['content'],
                            data['summary'],
                            json.dumps(data['metadata']),
                            data['source_id'],
                            _embedding_to_vector_string(data['embedding'])
                        )
                # Success - break out of retry loop
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error inserting batch into PostgreSQL (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")
                    # Optionally, try inserting records one by one as a last resort
                    print("Attempting to insert records individually...")
                    successful_inserts = 0
                    async with pool.acquire() as conn:
                        for record in batch_data:
                            try:
                                await conn.execute(
                                    insert_query,
                                    record['url'],
                                    record['chunk_number'],
                                    record['content'],
                                    record['summary'],
                                    json.dumps(record['metadata']),
                                    record['source_id'],
                                    embedding_to_vector_string(record['embedding'])
                                )
                                successful_inserts += 1
                            except Exception as individual_error:
                                print(f"Failed to insert individual record for URL {record['url']}: {individual_error}")
                    
                    if successful_inserts > 0:
                        print(f"Successfully inserted {successful_inserts}/{len(batch_data)} records individually")
        print(f"Inserted batch {i//batch_size + 1} of {(total_items + batch_size - 1)//batch_size} code examples")


async def update_source_info(pool: asyncpg.Pool, source_id: str, summary: str, word_count: int):
    """
    Update or insert source information in the sources table.
    
    Args:
        pool: PostgreSQL connection pool
        source_id: The source ID (domain)
        summary: Summary of the source
        word_count: Total word count for the source
    """
    try:
        async with pool.acquire() as conn:
            # Try to update existing source
            result = await conn.execute(
                """
                UPDATE sources 
                SET summary = $2, total_word_count = $3, updated_at = NOW() 
                WHERE source_id = $1
                """,
                source_id, summary, word_count
            )
            
            # If no rows were updated, insert new source
            if result == "UPDATE 0":
                await conn.execute(
                    """
                    INSERT INTO sources (source_id, summary, total_word_count) 
                    VALUES ($1, $2, $3)
                    """,
                    source_id, summary, word_count
                )
                print(f"Created new source: {source_id}")
            else:
                print(f"Updated source: {source_id}")
                
    except Exception as e:
        print(f"Error updating source {source_id}: {e}")


async def search_documents(
    pool: asyncpg.Pool, 
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in PostgreSQL using vector similarity.
    
    Args:
        pool: PostgreSQL connection pool
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        
    Returns:
        List of matching documents
    """
    from embeddings.generator import create_embedding
    
    # Create embedding for the query
    query_embedding = create_embedding(query)
    
    # Execute the search using the match_crawled_pages function
    try:
        async with pool.acquire() as conn:
            # Prepare filter parameter
            filter_json = json.dumps(filter_metadata) if filter_metadata else '{}'
            source_filter = filter_metadata.get('source') if filter_metadata else None
            
            # Call the match_crawled_pages function
            result = await conn.fetch(
                "SELECT * FROM match_crawled_pages($1, $2, $3, $4)",
                _embedding_to_vector_string(query_embedding),
                match_count,
                filter_json,
                source_filter
            )
            
            # Convert records to dictionaries
            return [dict(record) for record in result]
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []


async def search_code_examples(
    pool: asyncpg.Pool, 
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for code examples in PostgreSQL using vector similarity.
    
    Args:
        pool: PostgreSQL connection pool
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        source_id: Optional source ID to filter results
        
    Returns:
        List of matching code examples
    """
    from embeddings.generator import create_embedding
    
    # Create a more descriptive query for better embedding match
    enhanced_query = f"Code example for {query}\n\nSummary: Example code showing {query}"
    
    # Create embedding for the enhanced query
    query_embedding = create_embedding(enhanced_query)
    
    # Execute the search using the match_code_examples function
    try:
        async with pool.acquire() as conn:
            # Prepare filter parameter
            filter_json = json.dumps(filter_metadata) if filter_metadata else '{}'
            source_filter = source_id
            
            # Call the match_code_examples function
            result = await conn.fetch(
                "SELECT * FROM match_code_examples($1, $2, $3, $4)",
                _embedding_to_vector_string(query_embedding),
                match_count,
                filter_json,
                source_filter
            )
            
            # Convert records to dictionaries
            return [dict(record) for record in result]
    except Exception as e:
        print(f"Error searching code examples: {e}")
        return []