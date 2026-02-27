"""
RAG Pipeline Functions
Simple functional approach with hybrid retrieval (semantic + keyword)
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple

from dotenv import load_dotenv
load_dotenv()

import pymupdf4llm
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from prompts import get_prompt
from sensor_filter import should_include_sensors, get_sensor_context, get_critical_alerts, is_environment_query
from db_utils import get_latest_sensor_reading


# =============================================================================
# SHARED LLM INSTANCES (created once, reused across calls)
# =============================================================================

_embedding_model = None
_llm = None


def get_embedding_model():
    """Get or create the shared embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe"))
    return _embedding_model


def get_llm(model: str = None):
    """Get or create the shared LLM instance."""
    global _llm
    if model is None:
        model = os.getenv("OLLAMA_MODEL", "smollm2:1.7b")
    if _llm is None or _llm.model != model:
        _llm = OllamaLLM(
            model=model,
            temperature=0.2,      # low — factual advice needs consistency
            num_predict=400,      # tight budget — forces concise answers
            repeat_penalty=1.1,   # prevents small-model repetition loops
            top_k=40,
            top_p=0.9,
        )
    return _llm


# =============================================================================
# DOCUMENT LOADING
# =============================================================================

def load_documents(folder_path: str) -> List:
    """
    Load all .txt and .pdf files from a folder.
    
    Args:
        folder_path: Path to folder with knowledge base documents
    
    Returns:
        List of LangChain document objects
    """
    documents = []
    
    print(f"Loading documents from {folder_path}...")
    
    # Load TXT files
    for file_path in Path(folder_path).glob("*.txt"):
        loader = TextLoader(str(file_path), encoding='utf-8')
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file_path.name
        documents.extend(docs)
        print(f"  ✓ Loaded {file_path.name}")
    
    # Load PDF files
    for file_path in Path(folder_path).glob("*.pdf"):
        md_text = pymupdf4llm.to_markdown(str(file_path))
        doc = Document(
            page_content=md_text,
            metadata={"source": file_path.name}
        )
        documents.append(doc)
        print(f"  ✓ Loaded {file_path.name} (as Markdown)")
    
    print(f"Total documents loaded: {len(documents)}")
    return documents


def clean_text(text: str) -> str:
    """
    Clean text to improve embedding quality.
    
    Args:
        text: Raw text string
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove null bytes
    text = text.replace("\x00", "")
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    # Remove very long repeated characters (OCR errors)
    text = re.sub(r"(.)\1{10,}", r"\1", text)
    
    return text


def split_documents(documents: List, chunk_size: int = 800, chunk_overlap: int = 120) -> List:
    """
    Split documents into smaller chunks for embedding.
    
    Args:
        documents: List of documents
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of document chunks
    """
    print(f"Splitting documents into chunks (size={chunk_size})...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    
    # Clean each chunk and drop fragments too short to be useful
    chunks = [chunk for chunk in chunks if len(clean_text(chunk.page_content)) >= 50]
    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)

    print(f"Created {len(chunks)} chunks")
    return chunks


# =============================================================================
# VECTOR STORE (SEMANTIC RETRIEVAL)
# =============================================================================

def _docs_fingerprint(folder_path: str) -> str:
    """
    Build a hash fingerprint of the knowledge-base folder.
    Changes when files are added, removed, or modified.
    """
    entries = []
    for p in sorted(Path(folder_path).glob("*")):
        if p.suffix.lower() in (".txt", ".pdf"):
            entries.append(f"{p.name}:{p.stat().st_mtime_ns}")
    raw = "|".join(entries)
    return hashlib.md5(raw.encode()).hexdigest()


def _needs_rebuild(persist_dir: str, folder_path: str) -> bool:
    """
    Check if the vector DB needs to be rebuilt.
    Compares the stored fingerprint with the current docs folder.
    """
    fp_file = os.path.join(persist_dir, "_docs_fingerprint.json")
    if not os.path.exists(fp_file):
        return True
    with open(fp_file, "r") as f:
        stored = json.load(f).get("fingerprint", "")
    return stored != _docs_fingerprint(folder_path)


def _save_fingerprint(persist_dir: str, folder_path: str):
    """Save the current docs fingerprint alongside the Chroma DB."""
    fp_file = os.path.join(persist_dir, "_docs_fingerprint.json")
    with open(fp_file, "w") as f:
        json.dump({"fingerprint": _docs_fingerprint(folder_path)}, f)


def build_vector_store(
    chunks: List,
    persist_dir: str = "chroma_db",
    folder_path: str = "test_docs",
) -> Chroma:
    """
    Build or load Chroma vector database.
    Automatically rebuilds when knowledge-base files change.

    Args:
        chunks: Document chunks to embed
        persist_dir: Directory to store the database
        folder_path: Knowledge-base folder (used for change detection)

    Returns:
        Chroma vector store
    """
    embedding_model = get_embedding_model()

    # Reuse existing DB only if docs haven't changed
    if os.path.exists(persist_dir) and not _needs_rebuild(persist_dir, folder_path):
        print(f"Loading existing vector database from {persist_dir} (docs unchanged)")
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model,
        )
        return vectordb

    # Docs changed (or first run) — rebuild from scratch
    if os.path.exists(persist_dir):
        print("Knowledge base changed — rebuilding vector database...")
        import shutil
        shutil.rmtree(persist_dir)
    else:
        print("Building new vector database...")

    print(f"  Embedding {len(chunks)} chunks...")

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
    )

    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        vectordb.add_texts(texts=batch_texts, metadatas=batch_meta)
        print(f"  Progress: {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")

    print("\n✓ Vector database built successfully")
    _save_fingerprint(persist_dir, folder_path)
    return vectordb


def semantic_search(vectordb: Chroma, query: str, k: int = 3) -> List:
    """
    Perform semantic search using Chroma.
    
    Args:
        vectordb: Chroma vector database
        query: Search query
        k: Number of results to return
    
    Returns:
        List of relevant documents
    """
    retriever = vectordb.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance for diversity
        search_kwargs={"k": k, "fetch_k": k*3}
    )
    
    results = retriever.invoke(query)
    return results


# =============================================================================
# KEYWORD RETRIEVAL (BM25)
# =============================================================================

def build_bm25_retriever(chunks: List, k: int = 4) -> BM25Retriever:
    """
    Build BM25 keyword-based retriever.
    
    Args:
        chunks: Document chunks
        k: Number of results to return
    
    Returns:
        BM25Retriever
    """
    print("Building BM25 keyword retriever...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k
    print("✓ BM25 retriever ready")
    return bm25_retriever


# =============================================================================
# HYBRID RETRIEVAL (SEMANTIC + KEYWORD)
# =============================================================================

def hybrid_search(
    vectordb: Chroma,
    bm25_retriever: BM25Retriever,
    query: str,
    k: int = 3
) -> List:
    """
    Perform hybrid search combining semantic (Chroma) and keyword (BM25).
    
    Args:
        vectordb: Chroma vector database
        bm25_retriever: BM25 retriever
        query: Search query
        k: Number of results to return
    
    Returns:
        List of relevant documents
    """
    # Create semantic retriever
    semantic_retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k*3}
    )

    # Sync BM25 k to match the requested k
    bm25_retriever.k = k

    # Combine both retrievers
    # 60% weight to semantic, 40% to keyword
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )
    
    results = ensemble_retriever.invoke(query)
    return results[:k]  # Limit to k results


# =============================================================================
# LLM RESPONSE GENERATION
# =============================================================================

def format_context(documents: List, max_chars: int = 3000) -> str:
    """
    Format retrieved documents into a context string for the LLM.
    Stops adding chunks once max_chars is reached to protect token budget.
    At ~4 chars per token, 3000 chars ≈ 750 tokens — keeps input short for small LLMs.

    Args:
        documents: List of retrieved documents
        max_chars:  Hard ceiling on total context length in characters

    Returns:
        Formatted context string, capped at max_chars
    """
    context_parts = []
    total_chars = 0

    for i, doc in enumerate(documents, 1):
        content = doc.page_content
        entry = content.strip() + "\n"

        if total_chars + len(entry) > max_chars:
            print(f"  ⚠ Token budget reached — skipping chunk {i} onwards")
            break

        context_parts.append(entry)
        total_chars += len(entry)

    return "\n\n".join(context_parts)


def generate_response(prompt: str, model: str = None) -> str:
    """
    Generate response using LLM.

    Args:
        prompt: Complete prompt with context and query
        model: Ollama model to use

    Returns:
        Generated response
    """
    if model is None:
        model = os.getenv("OLLAMA_MODEL", "smollm2:1.7b")
    llm = get_llm(model)
    response = llm.invoke(prompt)
    return response


# =============================================================================
# CONVERSATION HISTORY FORMATTING
# =============================================================================

def format_conversation_history(history: list, max_exchanges: int = 2) -> str:
    """
    Format the last N conversation exchanges into a short context prefix.

    Keeps only the most recent exchanges to stay within the small model's
    token budget (~200 tokens max for 2 exchanges).

    Args:
        history: List of {"role": "user"|"assistant", "content": "..."} dicts
        max_exchanges: Maximum number of user+assistant pairs to include

    Returns:
        Formatted string like:
        "Previous conversation:
        User: ...
        Assistant: ...
        ---"
        Or empty string if no history.
    """
    if not history:
        return ""

    # Each "exchange" is one user turn + one assistant turn = 2 items
    # Take only the last max_exchanges*2 messages
    recent = history[-(max_exchanges * 2):]

    lines = ["Previous conversation:"]
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()
        # Truncate very long messages to protect token budget
        if len(content) > 200:
            content = content[:200] + "..."
        lines.append(f"{role}: {content}")
    lines.append("---")

    return "\n".join(lines)


# =============================================================================
# COMPLETE RAG PIPELINE
# =============================================================================

def answer_query(
    query: str,
    vectordb: Chroma,
    bm25_retriever: BM25Retriever,
    use_sensors: bool = True,
    use_hybrid: bool = True,
    k: int = 4,
    history: list = None,
) -> Dict:
    """
    Complete RAG pipeline: retrieve relevant chunks, build prompt, generate answer.

    Args:
        query:           User's question
        vectordb:        Chroma vector database
        bm25_retriever:  BM25 keyword retriever
        use_sensors:     Whether to check and include live sensor data
        use_hybrid:      Whether to use hybrid retrieval (vs semantic only)
        k:               Number of chunks to retrieve
        history:         Recent conversation history for context

    Returns:
        Dictionary with answer, sources, and metadata
    """

    # Step 1: Get sensor data if enabled
    sensor_data = None
    sensor_context = None
    has_critical = False

    if use_sensors:
        sensor_data = get_latest_sensor_reading()
        if sensor_data and should_include_sensors(query, sensor_data):
            sensor_context = get_sensor_context(sensor_data)
            critical_alerts = get_critical_alerts(sensor_data)
            # Emergency mode only when critical AND user is asking about their coop
            has_critical = len(critical_alerts) > 0 and is_environment_query(query)

    # Step 2: Retrieve relevant documents
    if use_hybrid:
        documents = hybrid_search(vectordb, bm25_retriever, query, k=k)
        retrieval_method = "hybrid"
    else:
        documents = semantic_search(vectordb, query, k=k)
        retrieval_method = "semantic"

    # Step 3: Build prompt
    context = format_context(documents)
    history_prefix = format_conversation_history(history or [])

    prompt = get_prompt(
        query=query,
        context=context,
        sensor_context=sensor_context,
        has_critical=has_critical,
    )

    if history_prefix:
        prompt = history_prefix + "\n\n" + prompt

    # Step 4: Generate response
    response = generate_response(prompt)

    return {
        "query": query,
        "answer": response,
        "sources": [doc.metadata.get("source", "Unknown") for doc in documents],
        "documents": documents,
        "sensor_included": sensor_context is not None,
        "sensor_data": sensor_data,
        "sensor_context": sensor_context,
        "has_critical": has_critical,
        "retrieval_method": retrieval_method,
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("RAG PIPELINE TEST")
    print("="*70)
    
    # Load and prepare knowledge base
    docs = load_documents("test_docs")
    chunks = split_documents(docs, chunk_size=800)
    
    # Build retrievers
    vectordb = build_vector_store(chunks)
    bm25_retriever = build_bm25_retriever(chunks)
    
    # Test queries
    test_queries = [
        "What temperature is too hot for chickens?",
        "How often do chickens lay eggs?",
        "My chickens are panting heavily"
    ]
    
    for query in test_queries:
        print("\n" + "="*70)
        print(f"Query: {query}")
        print("="*70)
        
        result = answer_query(query, vectordb, bm25_retriever, use_hybrid=True)
        
        print(f"\nSensor included: {result['sensor_included']}")
        if result['sensor_context']:
            print(f"Sensor context:\n{result['sensor_context']}")
        
        print(f"\nSources: {', '.join(result['sources'])}")
        print(f"\nAnswer:\n{result['answer']}")
