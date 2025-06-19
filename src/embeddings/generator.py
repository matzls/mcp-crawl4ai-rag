"""
Embedding generation and processing utilities.

This module handles OpenAI API integration for text embeddings, batch processing,
contextual embeddings, and vector format conversion for PostgreSQL.
"""
import os
import time
import concurrent.futures
from typing import List, Tuple
import openai

__all__ = [
    "create_embeddings_batch",
    "create_embedding", 
    "embedding_to_vector_string",
    "generate_contextual_embedding",
    "process_chunk_with_context"
]

# Load OpenAI API key for embeddings
openai.api_key = os.getenv("OPENAI_API_KEY")


def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call.
    
    Args:
        texts: List of texts to create embeddings for
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
    
    max_retries = 3
    retry_delay = 1.0  # Start with 1 second delay
    
    for retry in range(max_retries):
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small", # Hardcoding embedding model for now, will change this later to be more dynamic
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            if retry < max_retries - 1:
                print(f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to create batch embeddings after {max_retries} attempts: {e}")
                # Try creating embeddings one by one as fallback
                print("Attempting to create embeddings individually...")
                embeddings = []
                successful_count = 0
                
                for i, text in enumerate(texts):
                    try:
                        individual_response = openai.embeddings.create(
                            model="text-embedding-3-small",
                            input=[text]
                        )
                        embeddings.append(individual_response.data[0].embedding)
                        successful_count += 1
                    except Exception as individual_error:
                        print(f"Failed to create embedding for text {i}: {individual_error}")
                        # Add zero embedding as fallback
                        embeddings.append([0.0] * 1536)
                
                print(f"Successfully created {successful_count}/{len(texts)} embeddings individually")
                return embeddings


def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using OpenAI's API.

    Args:
        text: Text to create an embedding for

    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * 1536
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return empty embedding if there's an error
        return [0.0] * 1536


def embedding_to_vector_string(embedding: List[float]) -> str:
    """
    Convert a Python list embedding to PostgreSQL vector string format.

    Args:
        embedding: List of floats representing the embedding

    Returns:
        String representation suitable for PostgreSQL vector type
    """
    return '[' + ','.join(map(str, embedding)) + ']'


def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    
    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    model_choice = os.getenv("MODEL_CHOICE")
    
    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Call the OpenAI API to generate contextual information
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        # Extract the generated context
        context = response.choices[0].message.content.strip()
        
        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"
        
        return contextual_text, True
    
    except Exception as e:
        print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
        return chunk, False


def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (url, content, full_document)
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)


def process_documents_with_contextual_embeddings(
    urls: List[str],
    contents: List[str],
    metadatas: List[dict],
    url_to_full_document: dict,
    max_workers: int = 10
) -> Tuple[List[str], List[dict]]:
    """
    Process multiple documents with contextual embeddings using parallel processing.
    
    Args:
        urls: List of URLs
        contents: List of document contents
        metadatas: List of metadata dictionaries
        url_to_full_document: Dictionary mapping URLs to full document content
        max_workers: Maximum number of worker threads
        
    Returns:
        Tuple containing:
        - List of contextual contents
        - List of updated metadata dictionaries
    """
    use_contextual_embeddings = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
    print(f"\n\nUse contextual embeddings: {use_contextual_embeddings}\n\n")
    
    if not use_contextual_embeddings:
        return contents, metadatas
    
    # Prepare arguments for parallel processing
    process_args = []
    for i, content in enumerate(contents):
        url = urls[i]
        full_document = url_to_full_document.get(url, "")
        process_args.append((url, content, full_document))
    
    # Process in parallel using ThreadPoolExecutor
    contextual_contents = []
    updated_metadatas = metadatas.copy()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and collect results
        future_to_idx = {executor.submit(process_chunk_with_context, arg): idx 
                        for idx, arg in enumerate(process_args)}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result, success = future.result()
                contextual_contents.append(result)
                if success:
                    updated_metadatas[idx]["contextual_embedding"] = True
            except Exception as e:
                print(f"Error processing chunk {idx}: {e}")
                # Use original content as fallback
                contextual_contents.append(contents[idx])
    
    # Sort results back into original order if needed
    if len(contextual_contents) != len(contents):
        print(f"Warning: Expected {len(contents)} results but got {len(contextual_contents)}")
        # Use original contents as fallback
        contextual_contents = contents
    
    return contextual_contents, updated_metadatas