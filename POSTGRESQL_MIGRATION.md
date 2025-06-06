# PostgreSQL Migration Summary

The Crawl4AI RAG MCP Server has been successfully migrated from Supabase to local PostgreSQL. Here's what changed and how to set it up.

## What Changed

### Dependencies
- **Removed**: `supabase==2.15.1`
- **Added**: `asyncpg==0.30.0`

### Core Changes
1. **Connection Management**: Replaced Supabase client with PostgreSQL connection pool using asyncpg
2. **Database Operations**: All CRUD operations now use direct PostgreSQL queries
3. **Vector Search**: Still uses the same PostgreSQL functions (`match_crawled_pages`, `match_code_examples`)
4. **Schema**: No changes to database schema - same tables and functions

### Function Updates
- `get_supabase_client()` → `create_postgres_pool()`
- `add_documents_to_supabase()` → `add_documents_to_postgres()`
- `add_code_examples_to_supabase()` → `add_code_examples_to_postgres()`
- All search functions updated to use asyncpg

## Setup Instructions

### 1. Install PostgreSQL with pgvector

**Option A: Using Homebrew (macOS)**
```bash
brew install postgresql@16
brew install pgvector
brew services start postgresql@16
```

**Option B: Using Docker**
```bash
docker run --name crawl4ai-postgres -e POSTGRES_PASSWORD=mypassword -e POSTGRES_DB=crawl4ai_rag -p 5432:5432 -d pgvector/pgvector:pg16
```

### 2. Create Database and Schema

```bash
# Connect to PostgreSQL
psql -h localhost -U postgres -d crawl4ai_rag

# Run the schema creation
\i crawled_pages.sql
```

Or using a database client, execute the contents of `crawled_pages.sql`.

### 3. Update Dependencies

```bash
uv pip install -e .
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and update:

```bash
# Option 1: Use DATABASE_URL (recommended)
DATABASE_URL=postgresql://postgres:mypassword@localhost:5432/crawl4ai_rag

# Option 2: Individual components
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=crawl4ai_rag
POSTGRES_USER=postgres
POSTGRES_PASSWORD=mypassword

# Keep existing OpenAI and other settings
OPENAI_API_KEY=your_openai_api_key
MODEL_CHOICE=gpt-4o-mini
# ... other settings
```

### 5. Run the Server

```bash
uv run src/crawl4ai_mcp.py
```

## Database UI Tools

Now that you're using local PostgreSQL, you can inspect your database with:

### 1. pgAdmin
- Web-based PostgreSQL administration
- Download from: https://www.pgadmin.org/

### 2. DBeaver
- Free universal database tool
- Download from: https://dbeaver.io/

### 3. TablePlus (macOS/Windows)
- Modern database client
- Download from: https://tableplus.com/

### 4. Command Line
```bash
# Connect via psql
psql -h localhost -U postgres -d crawl4ai_rag

# View tables
\dt

# Query data
SELECT source_id, summary FROM sources;
SELECT COUNT(*) FROM crawled_pages;
```

## Benefits of This Migration

1. **No External Dependencies**: Everything runs locally
2. **Better Performance**: Direct database access without API overhead
3. **Full Control**: Complete control over database configuration and tuning
4. **Cost Savings**: No monthly Supabase costs
5. **Privacy**: All data stays on your machine
6. **Database Tools**: Access to full ecosystem of PostgreSQL tools

## Troubleshooting

### Connection Issues
- Ensure PostgreSQL is running: `brew services list` or `docker ps`
- Check connection string format
- Verify database exists: `psql -l`

### Schema Issues
- Re-run `crawled_pages.sql` if tables are missing
- Ensure pgvector extension is installed: `SELECT * FROM pg_extension WHERE extname = 'vector';`

### Performance
- Consider adding more indexes for large datasets
- Tune PostgreSQL configuration for your hardware
- Monitor connection pool usage

The migration maintains 100% compatibility with existing functionality while providing the benefits of local database control.