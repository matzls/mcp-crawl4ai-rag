#!/usr/bin/env python3
"""
Database Browser for Crawl4AI RAG Database

A simple CLI tool to browse and explore the PostgreSQL database
with rich formatting and interactive queries.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncpg
from datetime import datetime

# Add the src directory to the path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# Import rich for beautiful CLI interface
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, IntPrompt
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich not available, using basic CLI interface")


class DatabaseBrowser:
    """Interactive database browser for Crawl4AI RAG database."""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.pool = None
        
    def print_message(self, message: str, style: str = ""):
        """Print message with optional styling."""
        if RICH_AVAILABLE and self.console:
            self.console.print(message, style=style)
        else:
            print(message)
    
    def print_panel(self, content: str, title: str = "", style: str = ""):
        """Print content in a panel with optional styling."""
        if RICH_AVAILABLE and self.console:
            panel = Panel(content, title=title, style=style)
            self.console.print(panel)
        else:
            print(f"\n=== {title} ===")
            print(content)
            print("=" * (len(title) + 8))
    
    def print_table(self, data: List[Dict], title: str = ""):
        """Print data in a table format."""
        if not data:
            self.print_message("No data found.", "yellow")
            return
            
        if RICH_AVAILABLE and self.console:
            table = Table(title=title)
            
            # Add columns based on first row
            for key in data[0].keys():
                table.add_column(str(key))
            
            # Add rows
            for row in data:
                table.add_row(*[str(value) if value is not None else "NULL" for value in row.values()])
            
            self.console.print(table)
        else:
            # Basic table for non-rich environments
            if data:
                headers = list(data[0].keys())
                print(f"\n{title}")
                print(" | ".join(headers))
                print("-" * (len(" | ".join(headers))))
                for row in data:
                    print(" | ".join(str(row[key]) if row[key] is not None else "NULL" for key in headers))
    
    async def connect(self):
        """Connect to the database."""
        try:
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                # Fallback to individual components
                host = os.getenv("POSTGRES_HOST", "localhost")
                port = os.getenv("POSTGRES_PORT", "5432")
                database = os.getenv("POSTGRES_DB", "crawl4ai_rag")
                user = os.getenv("POSTGRES_USER", "mg")
                password = os.getenv("POSTGRES_PASSWORD", "")
                
                database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
            
            self.pool = await asyncpg.create_pool(database_url, min_size=1, max_size=5)
            self.print_message("✅ Connected to database successfully!", "green")
            return True
            
        except Exception as e:
            self.print_message(f"❌ Failed to connect to database: {e}", "red")
            return False
    
    async def close(self):
        """Close database connection."""
        if self.pool:
            await self.pool.close()
    
    async def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute a query and return results."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    async def show_database_overview(self):
        """Show database overview with table statistics."""
        self.print_panel("Database Overview", "📊 Crawl4AI RAG Database", "blue")
        
        # Table statistics
        queries = {
            "Sources": "SELECT COUNT(*) as count, SUM(total_word_count) as total_words FROM sources",
            "Crawled Pages": "SELECT COUNT(*) as count, COUNT(DISTINCT source_id) as unique_sources FROM crawled_pages",
            "Code Examples": "SELECT COUNT(*) as count, COUNT(DISTINCT source_id) as unique_sources FROM code_examples"
        }
        
        stats_data = []
        for table_name, query in queries.items():
            try:
                result = await self.execute_query(query)
                if result:
                    row = result[0]
                    if table_name == "Sources":
                        stats_data.append({
                            "Table": table_name,
                            "Count": row['count'],
                            "Extra Info": f"{row['total_words'] or 0:,} total words"
                        })
                    else:
                        stats_data.append({
                            "Table": table_name,
                            "Count": row['count'],
                            "Extra Info": f"{row['unique_sources']} unique sources"
                        })
            except Exception as e:
                stats_data.append({
                    "Table": table_name,
                    "Count": "Error",
                    "Extra Info": str(e)
                })
        
        self.print_table(stats_data, "📈 Table Statistics")
    
    async def browse_sources(self):
        """Browse available sources."""
        query = """
        SELECT source_id, summary, total_word_count, 
               created_at::date as created_date,
               (SELECT COUNT(*) FROM crawled_pages WHERE crawled_pages.source_id = sources.source_id) as page_chunks,
               (SELECT COUNT(*) FROM code_examples WHERE code_examples.source_id = sources.source_id) as code_chunks
        FROM sources 
        ORDER BY created_at DESC
        """
        
        try:
            results = await self.execute_query(query)
            self.print_table(results, "📚 Available Sources")
            
            if results:
                self.print_message(f"\n💡 Found {len(results)} sources with content", "cyan")
        except Exception as e:
            self.print_message(f"❌ Error browsing sources: {e}", "red")
    
    async def browse_recent_content(self, limit: int = 10):
        """Browse recent crawled content."""
        query = """
        SELECT url, chunk_number, 
               LEFT(content, 100) as content_preview,
               source_id, created_at::timestamp(0) as created_at
        FROM crawled_pages 
        ORDER BY created_at DESC 
        LIMIT $1
        """
        
        try:
            results = await self.execute_query(query, (limit,))
            self.print_table(results, f"📄 Recent Content (Last {limit})")
        except Exception as e:
            self.print_message(f"❌ Error browsing recent content: {e}", "red")
    
    async def search_content(self, search_term: str, limit: int = 10):
        """Search for content containing specific terms."""
        query = """
        SELECT url, chunk_number,
               LEFT(content, 150) as content_preview,
               source_id, created_at::date as created_date
        FROM crawled_pages 
        WHERE content ILIKE $1
        ORDER BY created_at DESC 
        LIMIT $2
        """
        
        try:
            search_pattern = f"%{search_term}%"
            results = await self.execute_query(query, (search_pattern, limit))
            
            if results:
                self.print_table(results, f"🔍 Search Results for '{search_term}'")
                self.print_message(f"\n💡 Found {len(results)} matches", "cyan")
            else:
                self.print_message(f"No content found containing '{search_term}'", "yellow")
                
        except Exception as e:
            self.print_message(f"❌ Error searching content: {e}", "red")
    
    async def browse_code_examples(self, limit: int = 10):
        """Browse code examples."""
        query = """
        SELECT url, LEFT(summary, 80) as summary_preview,
               source_id, created_at::date as created_date
        FROM code_examples 
        ORDER BY created_at DESC 
        LIMIT $1
        """
        
        try:
            results = await self.execute_query(query, (limit,))
            if results:
                self.print_table(results, f"💻 Code Examples (Last {limit})")
            else:
                self.print_message("No code examples found. Enable USE_AGENTIC_RAG=true to extract code examples.", "yellow")
        except Exception as e:
            self.print_message(f"❌ Error browsing code examples: {e}", "red")
    
    async def custom_query(self):
        """Execute a custom SQL query."""
        self.print_message("\n💡 Enter your SQL query (or 'back' to return):", "cyan")
        self.print_message("Example: SELECT COUNT(*) FROM crawled_pages WHERE source_id = 'example.com'", "dim")
        
        if RICH_AVAILABLE:
            query = Prompt.ask("SQL")
        else:
            query = input("SQL: ")
        
        if query.lower().strip() == 'back':
            return
        
        try:
            results = await self.execute_query(query)
            if results:
                self.print_table(results, "🔧 Custom Query Results")
            else:
                self.print_message("Query executed successfully (no results returned)", "green")
        except Exception as e:
            self.print_message(f"❌ Query error: {e}", "red")
    
    def show_menu(self):
        """Show the main menu."""
        menu_text = """
🗄️  Database Browser Menu

1. Database Overview
2. Browse Sources
3. Browse Recent Content
4. Search Content
5. Browse Code Examples
6. Custom SQL Query
7. Exit
        """
        self.print_panel(menu_text.strip(), "Main Menu", "blue")
    
    async def run(self):
        """Main browser loop."""
        self.print_message("🚀 Starting Database Browser...", "blue")
        
        if not await self.connect():
            return 1
        
        try:
            while True:
                self.show_menu()
                
                if RICH_AVAILABLE:
                    choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4", "5", "6", "7"])
                else:
                    choice = input("\nChoose an option (1-7): ").strip()
                
                if choice == "1":
                    await self.show_database_overview()
                elif choice == "2":
                    await self.browse_sources()
                elif choice == "3":
                    if RICH_AVAILABLE:
                        limit = IntPrompt.ask("Number of recent items to show", default=10)
                    else:
                        limit = int(input("Number of recent items to show (default 10): ") or "10")
                    await self.browse_recent_content(limit)
                elif choice == "4":
                    if RICH_AVAILABLE:
                        search_term = Prompt.ask("Enter search term")
                    else:
                        search_term = input("Enter search term: ")
                    await self.search_content(search_term)
                elif choice == "5":
                    if RICH_AVAILABLE:
                        limit = IntPrompt.ask("Number of code examples to show", default=10)
                    else:
                        limit = int(input("Number of code examples to show (default 10): ") or "10")
                    await self.browse_code_examples(limit)
                elif choice == "6":
                    await self.custom_query()
                elif choice == "7":
                    self.print_message("👋 Goodbye!", "green")
                    break
                else:
                    self.print_message("Invalid choice. Please try again.", "red")
                
                # Pause before showing menu again
                if RICH_AVAILABLE:
                    Prompt.ask("\nPress Enter to continue")
                else:
                    input("\nPress Enter to continue...")
        
        finally:
            await self.close()
        
        return 0


async def main():
    """Main entry point."""
    browser = DatabaseBrowser()
    return await browser.run()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
