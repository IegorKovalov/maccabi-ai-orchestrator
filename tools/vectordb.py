"""
Vector Database Tool for Maccabi AI Orchestrator
Handles document loading, chunking, embedding, and PostgreSQL/pgvector storage.
"""

import os
import re
from pathlib import Path
from typing import Optional

import google.generativeai as genai
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Embedding model - Gemini's text-embedding-004 supports 768 dimensions
EMBEDDING_MODEL = "models/text-embedding-004"
EMBEDDING_DIMENSION = 768

# Chunking configuration
CHUNK_SIZE = 500  # characters for fixed-size chunking
CHUNK_OVERLAP = 50


# =============================================================================
# DATABASE SETUP
# =============================================================================

def get_db_connection():
    """Create a connection to PostgreSQL database."""
    return psycopg2.connect(DATABASE_URL)


def init_database():
    """Initialize the database with pgvector extension and documents table."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create documents table
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector({EMBEDDING_DIMENSION}),
                source_file VARCHAR(255),
                chunk_index INTEGER,
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create index for faster similarity search
        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_embedding_idx 
            ON documents USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        
        conn.commit()
        print("✅ Database initialized successfully")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Database initialization failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()


def get_document_count() -> int:
    """Get the number of documents in the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT COUNT(*) FROM documents;")
        count = cur.fetchone()[0]
        return count
    except:
        return 0
    finally:
        cur.close()
        conn.close()


def get_indexed_files() -> list[str]:
    """Get list of unique source files already indexed."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT DISTINCT source_file FROM documents;")
        files = [row[0] for row in cur.fetchall()]
        return files
    except:
        return []
    finally:
        cur.close()
        conn.close()


# =============================================================================
# DOCUMENT LOADING
# =============================================================================

def load_document(file_path: str) -> str:
    """Load a document from file."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return content


def load_all_documents(directory: str) -> list[dict]:
    """Load all markdown documents from a directory."""
    documents = []
    dir_path = Path(directory)
    
    for file_path in dir_path.glob("*.md"):
        content = load_document(str(file_path))
        documents.append({
            "content": content,
            "source_file": file_path.name
        })
        print(f"📄 Loaded: {file_path.name}")
    
    return documents


# =============================================================================
# CHUNKING STRATEGIES
# =============================================================================

def chunk_by_sentences(text: str, sentences_per_chunk: int = 5) -> list[str]:
    """
    Split text into chunks based on sentences.
    Better for Hebrew text as it preserves semantic meaning.
    """
    # Hebrew and English sentence endings
    sentence_pattern = r'(?<=[.!?。])\s+'
    sentences = re.split(sentence_pattern, text)
    
    # Remove empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        if chunk:
            chunks.append(chunk)
    
    return chunks


def chunk_by_paragraphs(text: str, min_length: int = 50) -> list[str]:
    """
    Split text into chunks based on paragraphs.
    Ideal for structured documents with clear sections.
    Skips short paragraphs (headers) that are less than min_length characters.
    """
    # Split on double newlines (paragraph breaks)
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Remove empty paragraphs, strip whitespace, and filter out short headers
    chunks = []
    for p in paragraphs:
        p = p.strip()
        if p and len(p) >= min_length:
            chunks.append(p)
    
    return chunks


def chunk_fixed_size(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into fixed-size chunks with overlap.
    Simple but effective baseline strategy.
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start = end - overlap
    
    return chunks


def chunk_document(text: str, strategy: str = "sentences") -> list[str]:
    """
    Chunk a document using the specified strategy.
    
    Args:
        text: Document text to chunk
        strategy: One of "sentences", "paragraphs", or "fixed"
    
    Returns:
        List of text chunks
    """
    strategies = {
        "sentences": chunk_by_sentences,
        "paragraphs": chunk_by_paragraphs,
        "fixed": chunk_fixed_size
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Use one of {list(strategies.keys())}")
    
    return strategies[strategy](text)


# =============================================================================
# EMBEDDINGS
# =============================================================================

def get_embedding(text: str) -> list[float]:
    """Generate embedding for a single text using Gemini."""
    genai.configure(api_key=GEMINI_API_KEY)
    
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document"
    )
    
    return result["embedding"]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts."""
    genai.configure(api_key=GEMINI_API_KEY)
    
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(result["embedding"])
    
    return embeddings


# =============================================================================
# STORAGE
# =============================================================================

def store_chunks(chunks: list[str], embeddings: list[list[float]], source_file: str):
    """Store chunks and their embeddings in PostgreSQL."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Prepare data for batch insert
        data = [
            (chunk, embedding, source_file, idx)
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        
        execute_values(
            cur,
            """
            INSERT INTO documents (content, embedding, source_file, chunk_index)
            VALUES %s
            """,
            data,
            template="(%s, %s::vector, %s, %s)"
        )
        
        conn.commit()
        print(f"✅ Stored {len(chunks)} chunks from {source_file}")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Failed to store chunks: {e}")
        raise
    finally:
        cur.close()
        conn.close()


def clear_documents():
    """Clear all documents from the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("TRUNCATE TABLE documents RESTART IDENTITY;")
        conn.commit()
        print("🗑️ Cleared all documents from database")
    finally:
        cur.close()
        conn.close()


# =============================================================================
# RETRIEVAL (for RAG Agent)
# =============================================================================

def extract_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from Hebrew query."""
    stop_words = {'מה', 'איך', 'מתי', 'למה', 'האם', 'של', 'את', 'על', 'עם', 'או', 'אם', 'כי', 'אני', 'הוא', 'היא', 'אנחנו', 'הם', 'לי', 'לו', 'לה', 'לנו', 'להם', 'זה', 'זו', 'אלה', 'כל', 'יש', 'אין', 'רק', 'גם', 'עוד', 'כבר', 'עכשיו', 'היום', 'אפשר', 'צריך', 'רוצה'}
    words = query.split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords


def search_similar(query: str, top_k: int = 5) -> list[dict]:
    """
    Hybrid search: combines semantic similarity with keyword matching.
    
    Args:
        query: Search query text
        top_k: Number of results to return
    
    Returns:
        List of dicts with content, source_file, and similarity score
    """
    query_embedding = get_embedding(query)
    keywords = extract_keywords(query)
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Get more results for reranking
        cur.execute(
            """
            SELECT 
                id,
                content,
                source_file,
                1 - (embedding <=> %s::vector) as similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (query_embedding, query_embedding, top_k * 3)
        )
        
        results = []
        for row in cur.fetchall():
            content = row[1]
            semantic_score = row[3]
            
            # Keyword boost
            keyword_boost = 0
            for kw in keywords:
                if kw in content:
                    keyword_boost += 0.05
            
            combined_score = semantic_score + keyword_boost
            
            results.append({
                "content": content,
                "source_file": row[2],
                "similarity": combined_score
            })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
        
    finally:
        cur.close()
        conn.close()


def search_keyword(query: str, top_k: int = 5) -> list[dict]:
    """
    Pure keyword search - useful for specific terms like phone numbers.
    
    Args:
        query: Search query text (will search for exact substring)
        top_k: Number of results to return
    
    Returns:
        List of dicts with content and source_file
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute(
            """
            SELECT content, source_file
            FROM documents
            WHERE content ILIKE %s
            LIMIT %s;
            """,
            (f'%{query}%', top_k)
        )
        
        results = []
        for row in cur.fetchall():
            results.append({
                "content": row[0],
                "source_file": row[1],
                "similarity": 1.0  # Exact match
            })
        
        return results
        
    finally:
        cur.close()
        conn.close()
    
    try:
        # Cosine similarity search
        cur.execute(
            """
            SELECT 
                content,
                source_file,
                1 - (embedding <=> %s::vector) as similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (query_embedding, query_embedding, top_k)
        )
        
        results = []
        for row in cur.fetchall():
            results.append({
                "content": row[0],
                "source_file": row[1],
                "similarity": row[2]
            })
        
        return results
        
    finally:
        cur.close()
        conn.close()


# =============================================================================
# MAIN INDEXING FUNCTION
# =============================================================================

def index_documents(
    documents_dir: str,
    chunking_strategy: str = "sentences",
    clear_existing: bool = False,
    force: bool = False
):
    """
    Main function to index all documents in a directory.
    
    Args:
        documents_dir: Path to directory containing markdown files
        chunking_strategy: "sentences", "paragraphs", or "fixed"
        clear_existing: Whether to clear existing documents first
        force: Force re-indexing even if documents exist
    """
    print(f"\n{'='*60}")
    print("🚀 Starting document indexing")
    print(f"{'='*60}")
    print(f"📁 Directory: {documents_dir}")
    print(f"📝 Chunking strategy: {chunking_strategy}")
    print()
    
    # Initialize database
    init_database()
    
    # Check if documents already exist
    existing_count = get_document_count()
    
    if existing_count > 0 and not clear_existing and not force:
        indexed_files = get_indexed_files()
        print(f"📊 Database already has {existing_count} chunks from {len(indexed_files)} files:")
        for f in indexed_files:
            print(f"   • {f}")
        print(f"\n⏭️  Skipping indexing. Use --clear to re-index or --force to add more.")
        print(f"{'='*60}\n")
        return
    
    # Clear existing if requested
    if clear_existing:
        clear_documents()
    
    # Load documents
    documents = load_all_documents(documents_dir)
    print(f"\n📚 Loaded {len(documents)} documents\n")
    
    # Process each document
    total_chunks = 0
    for doc in documents:
        print(f"Processing: {doc['source_file']}")
        
        # Chunk the document
        chunks = chunk_document(doc["content"], strategy=chunking_strategy)
        print(f"  → {len(chunks)} chunks")
        
        # Generate embeddings
        print(f"  → Generating embeddings...")
        embeddings = get_embeddings_batch(chunks)
        
        # Store in database
        store_chunks(chunks, embeddings, doc["source_file"])
        total_chunks += len(chunks)
    
    print(f"\n{'='*60}")
    print(f"✅ Indexing complete! Total chunks: {total_chunks}")
    print(f"{'='*60}\n")


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index documents for RAG")
    parser.add_argument(
        "--dir",
        default="data/medical_docs",
        help="Directory containing documents"
    )
    parser.add_argument(
        "--strategy",
        choices=["sentences", "paragraphs", "fixed"],
        default="sentences",
        help="Chunking strategy"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing documents before indexing"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force indexing even if documents exist (adds to existing)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show database status only (no indexing)"
    )
    parser.add_argument(
        "--test-search",
        type=str,
        help="Test search with a query (no indexing)"
    )
    
    args = parser.parse_args()
    
    # Status check only
    if args.status:
        init_database()
        count = get_document_count()
        files = get_indexed_files()
        print(f"\n📊 Database status:")
        print(f"   Total chunks: {count}")
        print(f"   Indexed files: {len(files)}")
        for f in files:
            print(f"   • {f}")
        print()
    # Test search only (no indexing)
    elif args.test_search:
        print(f"\n🔍 Searching: '{args.test_search}'")
        results = search_similar(args.test_search, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} (similarity: {result['similarity']:.4f}) ---")
            print(f"Source: {result['source_file']}")
            print(f"Content: {result['content'][:200]}...")
    # Run indexing
    else:
        index_documents(
            documents_dir=args.dir,
            chunking_strategy=args.strategy,
            clear_existing=args.clear,
            force=args.force
        )