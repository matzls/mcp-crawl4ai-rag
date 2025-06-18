#!/usr/bin/env python3
"""
CLI Chat Interface for Crawl4AI Unified Agent

Interactive command-line interface that provides a natural chat experience
with the intelligent orchestrator agent that can crawl, search, and synthesize
web content using all 5 MCP tools.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional

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
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich not available, using basic CLI interface")

from pydantic_agent.unified_agent import (
    create_unified_agent,
    run_unified_agent,
    UnifiedAgentDependencies,
    setup_logfire_instrumentation
)
from logging_config import log_error


class ChatInterface:
    """Interactive chat interface for the unified agent."""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.agent = None
        self.dependencies = None
        self.conversation_history = []
        
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
    
    def print_table(self, data: list, headers: list, title: str = ""):
        """Print data in a table format."""
        if RICH_AVAILABLE and self.console:
            table = Table(title=title)
            for header in headers:
                table.add_column(header)
            for row in data:
                table.add_row(*[str(item) for item in row])
            self.console.print(table)
        else:
            print(f"\n{title}")
            print(" | ".join(headers))
            print("-" * (len(" | ".join(headers))))
            for row in data:
                print(" | ".join(str(item) for item in row))
    
    async def initialize(self):
        """Initialize the agent and dependencies."""
        self.print_message("🔧 Initializing Crawl4AI Unified Agent...", "blue")
        
        try:
            # Check prerequisites
            if not os.getenv("OPENAI_API_KEY"):
                self.print_message("⚠️  Warning: OPENAI_API_KEY not found in environment", "yellow")
                return False
            
            # Create dependencies
            self.dependencies = UnifiedAgentDependencies(
                mcp_server_url="http://localhost:8051/sse",
                default_max_depth=3,
                default_max_concurrent=10,
                default_match_count=5,
                enable_hybrid_search=True,
                enable_code_search=True
            )
            
            # Create unified agent
            self.agent = create_unified_agent(self.dependencies.mcp_server_url)
            
            self.print_message("✅ Agent initialized successfully!", "green")
            return True
            
        except Exception as e:
            self.print_message(f"❌ Failed to initialize agent: {e}", "red")
            return False
    
    def show_welcome(self):
        """Display welcome message and instructions."""
        welcome_text = """
🤖 Crawl4AI Unified Agent - Interactive Chat Interface

I'm an intelligent research assistant that can:
• 🕷️  Crawl websites, sitemaps, and documentation
• 🔍 Search through previously crawled content  
• 💻 Find specific code examples and implementations
• 📚 Synthesize information from multiple sources
• 🎯 Answer questions with cited sources

I intelligently choose from 5 specialized tools based on your needs:
- smart_crawl_url, crawl_single_page (for gathering new content)
- get_available_sources (for discovering what's indexed)  
- perform_rag_query, search_code_examples (for searching existing content)
"""
        
        self.print_panel(welcome_text.strip(), "Welcome", "blue")
        
        self.print_message("\n💡 Example queries:", "cyan")
        examples = [
            "Research Python async programming from the official docs",
            "What do you know about machine learning?", 
            "Show me code examples for error handling",
            "Crawl https://fastapi.tiangolo.com and summarize the features",
            "Find information about database connections"
        ]
        
        for i, example in enumerate(examples, 1):
            self.print_message(f"   {i}. {example}", "dim")
        
        self.print_message("\n🎮 Commands:", "cyan")
        commands = [
            ("help", "Show this help message"),
            ("sources", "List available content sources"),
            ("history", "Show conversation history"),
            ("clear", "Clear conversation history"),
            ("quit/exit", "Exit the chat interface")
        ]
        
        for cmd, desc in commands:
            self.print_message(f"   {cmd:10} - {desc}", "dim")
    
    async def handle_command(self, user_input: str) -> bool:
        """Handle special commands. Returns True if command was handled."""
        command = user_input.lower().strip()
        
        if command in ['quit', 'exit', 'q']:
            self.print_message("👋 Thanks for using Crawl4AI! Goodbye!", "green")
            return True
            
        elif command == 'help':
            self.show_welcome()
            return True
            
        elif command == 'sources':
            await self.show_available_sources()
            return True
            
        elif command == 'history':
            self.show_conversation_history()
            return True
            
        elif command == 'clear':
            self.conversation_history.clear()
            self.print_message("🧹 Conversation history cleared!", "green")
            return True
            
        return False
    
    async def show_available_sources(self):
        """Query and display available content sources."""
        self.print_message("🔍 Checking available content sources...", "blue")
        
        try:
            # Use the agent to check sources
            result = await run_unified_agent(
                self.agent,
                "What content sources do you have available? Use get_available_sources to check.",
                self.dependencies
            )
            
            if result.workflow_success and result.content_searched:
                sources_data = result.content_searched.get('sources', [])
                if sources_data:
                    table_data = []
                    for source in sources_data:
                        table_data.append([
                            source.get('source_id', 'Unknown'),
                            str(source.get('total_words', 0)),
                            source.get('summary', 'No summary')[:60] + "..."
                        ])
                    
                    self.print_table(
                        table_data,
                        ["Source", "Words", "Summary"],
                        "📚 Available Content Sources"
                    )
                else:
                    self.print_message("📭 No content sources found. Try crawling some websites first!", "yellow")
            else:
                self.print_message("❌ Failed to retrieve sources", "red")
                
        except Exception as e:
            self.print_message(f"❌ Error checking sources: {e}", "red")
    
    def show_conversation_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            self.print_message("📝 No conversation history yet.", "yellow")
            return
        
        self.print_message("📜 Conversation History:", "cyan")
        for i, (query, response_summary) in enumerate(self.conversation_history, 1):
            self.print_message(f"\n{i}. User: {query}", "white")
            self.print_message(f"   Assistant: {response_summary}", "dim")
    
    async def process_query(self, user_query: str):
        """Process user query with the unified agent."""
        # Show processing indicator
        if RICH_AVAILABLE and self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("🤖 Processing your query...", total=None)
                
                try:
                    result = await run_unified_agent(
                        self.agent,
                        user_query, 
                        self.dependencies
                    )
                except Exception as e:
                    progress.stop()
                    self.print_message(f"❌ Error processing query: {e}", "red")
                    return
        else:
            self.print_message("🤖 Processing your query...", "blue")
            try:
                result = await run_unified_agent(
                    self.agent,
                    user_query,
                    self.dependencies
                )
            except Exception as e:
                self.print_message(f"❌ Error processing query: {e}", "red")
                return
        
        # Display results
        self.display_agent_response(result)
        
        # Add to conversation history
        response_summary = result.primary_findings[:100] + "..." if len(result.primary_findings) > 100 else result.primary_findings
        self.conversation_history.append((user_query, response_summary))
    
    def display_agent_response(self, result):
        """Display the agent's response in a formatted way."""
        # Main response
        self.print_panel(result.primary_findings, "🤖 Response", "green")
        
        # Workflow details
        if result.steps_executed:
            workflow_text = f"Strategy: {result.workflow_strategy}\n"
            workflow_text += f"Steps: {len(result.steps_executed)} | "
            workflow_text += f"Time: {result.total_execution_time_seconds:.2f}s | "
            workflow_text += f"Confidence: {result.confidence_score:.0%}"
            
            self.print_message(f"\n📊 {workflow_text}", "cyan")
        
        # Supporting evidence
        if result.supporting_evidence:
            evidence_text = "\n".join(f"• {evidence}" for evidence in result.supporting_evidence[:3])
            if len(result.supporting_evidence) > 3:
                evidence_text += f"\n... and {len(result.supporting_evidence) - 3} more"
            
            self.print_panel(evidence_text, "📝 Supporting Evidence", "blue")
        
        # Sources accessed
        if result.sources_accessed:
            sources_text = ", ".join(result.sources_accessed[:5])
            if len(result.sources_accessed) > 5:
                sources_text += f" and {len(result.sources_accessed) - 5} more"
            
            self.print_message(f"🔗 Sources: {sources_text}", "dim")
        
        # Recommendations
        if result.recommendations:
            rec_text = "\n".join(f"• {rec}" for rec in result.recommendations[:3])
            self.print_panel(rec_text, "💡 Recommendations", "yellow")
        
        # Warnings for partial failures
        if result.partial_failures:
            failure_text = "\n".join(f"• {failure}" for failure in result.partial_failures)
            self.print_panel(failure_text, "⚠️  Partial Failures", "orange")
    
    async def run(self):
        """Main chat loop."""
        # Initialize
        if not await self.initialize():
            return 1
        
        # Show welcome
        self.show_welcome()
        
        # Chat loop
        self.print_message("\n💬 Start chatting! (type 'help' for commands, 'quit' to exit)\n", "green")
        
        while True:
            try:
                # Get user input
                if RICH_AVAILABLE:
                    user_input = Prompt.ask("[bold blue]You[/bold blue]").strip()
                else:
                    user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if await self.handle_command(user_input):
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    continue
                
                # Process query
                await self.process_query(user_input)
                
            except KeyboardInterrupt:
                self.print_message("\n👋 Goodbye! Thanks for using Crawl4AI!", "green")
                break
            except Exception as e:
                self.print_message(f"❌ Unexpected error: {e}", "red")
                log_error("CLI chat error", error=str(e), error_type=type(e).__name__)
        
        return 0


async def main():
    """Main entry point."""
    # Setup Logfire instrumentation first
    logfire_enabled = setup_logfire_instrumentation()

    # Check if MCP server is likely running
    if RICH_AVAILABLE:
        console = Console()
        console.print("🚀 Starting Crawl4AI CLI Chat Interface", style="bold green")
        if logfire_enabled:
            console.print("📊 Logfire observability enabled", style="green")
            console.print("🔗 Dashboard: https://logfire-eu.pydantic.dev/matzls/crawl4ai-agent", style="blue")
    else:
        print("🚀 Starting Crawl4AI CLI Chat Interface")
        if logfire_enabled:
            print("📊 Logfire observability enabled")
            print("🔗 Dashboard: https://logfire-eu.pydantic.dev/matzls/crawl4ai-agent")

    # Basic checks
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
        print("   Please set it with: export OPENAI_API_KEY=your_key_here")

    print("💡 Make sure the MCP server is running: ./start_mcp_server.sh")
    print()

    # Run chat interface
    chat = ChatInterface()
    return await chat.run()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)