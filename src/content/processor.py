"""
Content processing utilities for text analysis and summarization.

This module handles code extraction from markdown, content summarization using LLMs,
and source analysis for the Crawl4AI RAG system.
"""
import os
import re
from typing import List, Dict, Any
import openai

__all__ = [
    "extract_code_blocks",
    "generate_code_example_summary",
    "extract_source_summary"
]


def extract_code_blocks(markdown_content: str, min_length: int = 1000) -> List[Dict[str, Any]]:
    """
    Extract code blocks from markdown content along with context.
    
    Args:
        markdown_content: The markdown content to extract code blocks from
        min_length: Minimum length of code blocks to extract (default: 1000 characters)
        
    Returns:
        List of dictionaries containing code blocks and their context
    """
    code_blocks = []
    
    # Skip if content starts with triple backticks (edge case for files wrapped in backticks)
    content = markdown_content.strip()
    start_offset = 0
    if content.startswith('```'):
        # Skip the first triple backticks
        start_offset = 3
        print("Skipping initial triple backticks")
    
    # Find all occurrences of triple backticks
    backtick_positions = []
    pos = start_offset
    while True:
        pos = markdown_content.find('```', pos)
        if pos == -1:
            break
        backtick_positions.append(pos)
        pos += 3
    
    # Process pairs of backticks
    i = 0
    while i < len(backtick_positions) - 1:
        start_pos = backtick_positions[i]
        end_pos = backtick_positions[i + 1]
        
        # Extract the content between backticks
        code_section = markdown_content[start_pos+3:end_pos]
        
        # Check if there's a language specifier on the first line
        lines = code_section.split('\n', 1)
        if len(lines) > 1:
            # Check if first line is a language specifier (no spaces, common language names)
            first_line = lines[0].strip()
            if first_line and ' ' not in first_line and len(first_line) < 20:
                language = first_line
                code_content = lines[1].strip() if len(lines) > 1 else ""
            else:
                language = ""
                code_content = code_section.strip()
        else:
            language = ""
            code_content = code_section.strip()
        
        # Skip if code block is too short
        if len(code_content) < min_length:
            i += 2  # Move to next pair
            continue
        
        # Extract context before (1000 chars)
        context_start = max(0, start_pos - 1000)
        context_before = markdown_content[context_start:start_pos].strip()
        
        # Extract context after (1000 chars)
        context_end = min(len(markdown_content), end_pos + 3 + 1000)
        context_after = markdown_content[end_pos + 3:context_end].strip()
        
        code_blocks.append({
            'code': code_content,
            'language': language,
            'context_before': context_before,
            'context_after': context_after,
            'full_context': f"{context_before}\n\n{code_content}\n\n{context_after}"
        })
        
        # Move to next pair (skip the closing backtick we just processed)
        i += 2
    
    return code_blocks


def generate_code_example_summary(code: str, context_before: str, context_after: str) -> str:
    """
    Generate a summary for a code example using its surrounding context.
    
    Args:
        code: The code example
        context_before: Context before the code
        context_after: Context after the code
        
    Returns:
        A summary of what the code example demonstrates
    """
    model_choice = os.getenv("MODEL_CHOICE")
    
    # Create the prompt
    prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated.
"""
    
    try:
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise code example summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error generating code example summary: {e}")
        return "Code example for demonstration purposes."


def extract_source_summary(source_id: str, content: str, max_length: int = 500) -> str:
    """
    Extract a summary for a source from its content using an LLM.
    
    This function uses the OpenAI API to generate a concise summary of the source content.
    
    Args:
        source_id: The source ID (domain)
        content: The content to extract a summary from
        max_length: Maximum length of the summary
        
    Returns:
        A summary string
    """
    # Default summary if we can't extract anything meaningful
    default_summary = f"Content from {source_id}"
    
    if not content or len(content.strip()) == 0:
        return default_summary
    
    # Get the model choice from environment variables
    model_choice = os.getenv("MODEL_CHOICE")
    
    # Limit content length to avoid token limits
    truncated_content = content[:25000] if len(content) > 25000 else content
    
    # Create the prompt for generating the summary
    prompt = f"""<source_content>
{truncated_content}
</source_content>

The above content is from the documentation for '{source_id}'. Please provide a concise summary (3-5 sentences) that describes what this library/tool/framework is about. The summary should help understand what the library/tool/framework accomplishes and the purpose.
"""
    
    try:
        # Call the OpenAI API to generate the summary
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise library/tool/framework summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        # Extract the generated summary
        summary = response.choices[0].message.content.strip()
        
        # Ensure the summary is not too long
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
            
        return summary
    
    except Exception as e:
        print(f"Error generating summary with LLM for {source_id}: {e}. Using default summary.")
        return default_summary


def process_code_examples_batch(
    markdown_content: str,
    url: str,
    min_code_length: int = 1000
) -> List[Dict[str, Any]]:
    """
    Process a single markdown document and extract all code examples with summaries.
    
    Args:
        markdown_content: The markdown content to process
        url: The URL of the source document
        min_code_length: Minimum length for code blocks to be considered
        
    Returns:
        List of dictionaries containing processed code examples
    """
    code_blocks = extract_code_blocks(markdown_content, min_code_length)
    
    processed_examples = []
    for i, block in enumerate(code_blocks):
        # Generate summary for this code block
        summary = generate_code_example_summary(
            block['code'],
            block['context_before'], 
            block['context_after']
        )
        
        # Create metadata
        metadata = {
            'language': block['language'],
            'code_length': len(block['code']),
            'has_context': bool(block['context_before'] or block['context_after']),
            'chunk_index': i
        }
        
        processed_examples.append({
            'url': url,
            'chunk_number': i,
            'code': block['code'],
            'summary': summary,
            'metadata': metadata,
            'full_context': block['full_context']
        })
    
    return processed_examples


def analyze_content_structure(content: str) -> Dict[str, Any]:
    """
    Analyze the structure of markdown content and extract metadata.
    
    Args:
        content: The markdown content to analyze
        
    Returns:
        Dictionary containing structural analysis
    """
    # Count different elements
    header_count = len(re.findall(r'^#+\s', content, re.MULTILINE))
    code_block_count = content.count('```') // 2
    word_count = len(content.split())
    line_count = content.count('\n') + 1
    
    # Extract headers
    headers = re.findall(r'^(#+)\s+(.+)$', content, re.MULTILINE)
    header_structure = [{'level': len(h[0]), 'text': h[1]} for h in headers]
    
    # Detect content type
    content_type = 'documentation'
    if 'api' in content.lower() or 'endpoint' in content.lower():
        content_type = 'api_documentation'
    elif 'tutorial' in content.lower() or 'guide' in content.lower():
        content_type = 'tutorial'
    elif code_block_count > 10:
        content_type = 'code_heavy'
    
    return {
        'word_count': word_count,
        'line_count': line_count,
        'header_count': header_count,
        'code_block_count': code_block_count,
        'header_structure': header_structure,
        'content_type': content_type,
        'estimated_reading_time': max(1, word_count // 200)  # Approximate reading time in minutes
    }