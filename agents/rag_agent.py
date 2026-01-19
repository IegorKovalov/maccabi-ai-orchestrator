"""
RAG Agent for Maccabi AI Orchestrator
Retrieves relevant documents and generates answers using Claude.
"""

import os
from typing import Any

import anthropic
from dotenv import load_dotenv

from tools.vectordb import search_similar

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


# =============================================================================
# RAG AGENT
# =============================================================================

def create_rag_prompt(query: str, context_docs: list[dict]) -> str:
    """Create a prompt with retrieved context for Claude."""
    
    # Format retrieved documents
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        context_parts.append(f"""
מסמך {i} (מקור: {doc['source_file']}, רלוונטיות: {doc['similarity']:.0%}):
{doc['content']}
""")
    
    context_text = "\n---\n".join(context_parts)
    
    prompt = f"""אתה עוזר וירטואלי של מכבי שירותי בריאות. עליך לענות על שאלות המבוטחים בעברית, בצורה מקצועית ואדיבה.

השתמש במידע הבא מתוך מסמכי מכבי כדי לענות על השאלה:

{context_text}

---

שאלת המבוטח: {query}

הנחיות:
1. ענה בעברית בלבד
2. התבסס רק על המידע שסופק למעלה
3. אם המידע לא מספיק לתשובה מלאה, ציין זאת בכנות
4. היה תמציתי אך מקיף
5. אם רלוונטי, הפנה את המבוטח למוקד *3555 לפרטים נוספים

תשובה:"""

    return prompt


def rag_query(query: str, top_k: int = 10) -> dict[str, Any]:
    """
    Execute a RAG query: retrieve documents and generate answer.
    
    Args:
        query: User's question in Hebrew
        top_k: Number of documents to retrieve
    
    Returns:
        Dict with answer, sources, and metadata
    """
    # Step 1: Retrieve relevant documents
    retrieved_docs = search_similar(query, top_k=top_k)
    
    if not retrieved_docs:
        return {
            "answer": "מצטער, לא מצאתי מידע רלוונטי לשאלתך. אנא פנה למוקד מכבי *3555.",
            "sources": [],
            "query": query
        }
    
    # Step 2: Create prompt with context
    prompt = create_rag_prompt(query, retrieved_docs)
    
    # Step 3: Generate answer with Claude
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    answer = response.content[0].text
    
    # Step 4: Return structured response
    return {
        "answer": answer,
        "sources": [
            {
                "file": doc["source_file"],
                "similarity": doc["similarity"],
                "snippet": doc["content"][:200] + "..."
            }
            for doc in retrieved_docs
        ],
        "query": query,
        "model": MODEL,
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens
    }


# =============================================================================
# LANGGRAPH NODE FUNCTION
# =============================================================================

def rag_agent_node(state: dict) -> dict:
    """
    LangGraph node function for RAG agent.
    
    Expected state:
        - query: str (user's question)
    
    Returns updated state with:
        - rag_response: dict (answer + sources)
    """
    query = state.get("query", "")
    
    if not query:
        return {
            **state,
            "rag_response": {
                "answer": "לא התקבלה שאלה.",
                "sources": [],
                "query": ""
            }
        }
    
    result = rag_query(query)
    
    return {
        **state,
        "rag_response": result
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================

def interactive_mode():
    """Run interactive Q&A session."""
    print("\n" + "=" * 60)
    print("🏥 מכבי AI - מערכת שאלות ותשובות")
    print("=" * 60)
    print("הקלד שאלה בעברית (או 'exit' ליציאה)\n")
    
    while True:
        query = input("❓ שאלה: ").strip()
        
        if query.lower() in ['exit', 'quit', 'יציאה']:
            print("\n👋 להתראות!")
            break
        
        if not query:
            continue
        
        print("\n🔍 מחפש מידע רלוונטי...")
        result = rag_query(query)
        
        print("\n" + "-" * 40)
        print("💬 תשובה:")
        print(result["answer"])
        print("-" * 40)
        
        print("\n📚 מקורות:")
        for src in result["sources"]:
            print(f"  • {src['file']} ({src['similarity']:.0%})")
        
        print(f"\n📊 טוקנים: {result['tokens_used']}")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Maccabi RAG Agent")
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to answer"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive Q&A session"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.query:
        result = rag_query(args.query)
        print(f"\n💬 תשובה:\n{result['answer']}")
        print(f"\n📚 מקורות: {[s['file'] for s in result['sources']]}")
    else:
        # Default: interactive mode
        interactive_mode()