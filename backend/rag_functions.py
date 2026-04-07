"""
RAG Pipeline Functions
Simple functional approach with hybrid retrieval (semantic + keyword)
"""

import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

import pymupdf4llm
pymupdf4llm.use_layout(False)  # disable broken layout path in pymupdf4llm 0.3.4 + PyMuPDF 1.27.2
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from backend.prompts import get_prompt
from backend.sensor_filter import (
    should_include_sensors, get_sensor_context, get_critical_alerts,
    is_environment_query, is_reading_stale, llm_route_sensors,
)
from backend.db_utils import (
    get_latest_sensor_reading,
    get_chore_log_in_range,
    get_automation_windows_in_range,
)


def build_task_context(days_back: int = 14) -> str:
    """
    Build an optional task-context block the user can opt into from the chat UI.
    Includes recent chore entries (formatted via each chore's llm_template) and
    active automation windows. Returns "" if nothing relevant.
    """
    from datetime import date, timedelta
    today = date.today()
    start = today - timedelta(days=days_back)
    lines: List[str] = []

    try:
        chores = get_chore_log_in_range(start, today)
    except Exception:
        chores = []
    if chores:
        lines.append("Recent chores:")
        for c in chores[:10]:
            entry_date = c.get("entry_date")
            days_ago = (today - entry_date).days if entry_date else 0
            items = ", ".join(c.get("checked_items") or []) or "—"
            tmpl = c.get("llm_template") or "{label} ({days_ago}d ago): {items}"
            try:
                line = tmpl.format(
                    days_ago=days_ago,
                    items=items,
                    label=c.get("label", ""),
                )
            except Exception:
                line = f"{c.get('label','chore')} ({days_ago}d ago)"
            lines.append(f"- {line}")

    try:
        windows = get_automation_windows_in_range(today, today)
    except Exception:
        windows = []
    if windows:
        lines.append("Active automation today:")
        for w in windows:
            st = w.get("start_time")
            et = w.get("end_time")
            span = f"{st}–{et}" if st and et else "all day"
            lines.append(f"- {w.get('task')}: {span}")

    return "\n".join(lines)


# =============================================================================
# SHARED LLM INSTANCES (created once, reused across calls)
# =============================================================================

_embedding_model = None
_embedding_model_id = None
_llm = None
_llm_model_id = None


def get_embedding_model(model: str = None):
    """Get or create the shared embedding model instance."""
    global _embedding_model, _embedding_model_id
    if model is None:
        model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe")
    if _embedding_model is None or _embedding_model_id != model:
        if model == "text-embedding-3-small":
            from langchain_openai import OpenAIEmbeddings
            _embedding_model = OpenAIEmbeddings(model=model)
        else:
            _embedding_model = OllamaEmbeddings(model=model)
        _embedding_model_id = model
    return _embedding_model


def get_llm(model: str = None):
    """Get or create the shared LLM instance.

    Supports local Ollama models and OpenRouter models (prefix: 'openrouter/').
    Example: get_llm("openrouter/meta-llama/llama-3.1-8b-instruct:free")
    """
    global _llm, _llm_model_id
    if model is None:
        model = os.getenv("OLLAMA_MODEL", "smollm2:1.7b")
    if _llm is None or _llm_model_id != model:
        if model.startswith("openrouter/"):
            from langchain_openai import ChatOpenAI
            _llm = ChatOpenAI(
                model=model.removeprefix("openrouter/"),
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                temperature=0.2,
                max_tokens=600,
            )
        else:
            _llm = OllamaLLM(
                model=model,
                temperature=0.2,      # low — factual advice needs consistency
                num_predict=600,
                repeat_penalty=1.1,   # prevents small-model repetition loops
                top_k=40,
                top_p=0.9,
            )
        _llm_model_id = model
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
        print(f"  [OK] Loaded {file_path.name}")
    
    # Load PDF files
    for file_path in Path(folder_path).glob("*.pdf"):
        md_text = pymupdf4llm.to_markdown(str(file_path))
        doc = Document(
            page_content=md_text,
            metadata={"source": file_path.name}
        )
        documents.append(doc)
        print(f"  [OK] Loaded {file_path.name} (as Markdown)")
    
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


def split_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 120) -> List:
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
    Uses file content hash (not mtime) so it works correctly inside Docker on Windows.
    """
    hasher = hashlib.md5()
    for p in sorted(Path(folder_path).glob("*")):
        if p.suffix.lower() in (".txt", ".pdf"):
            hasher.update(p.name.encode())
            hasher.update(p.read_bytes())
    return hasher.hexdigest()


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
    embedding_model: str = None,
) -> Chroma:
    """
    Build or load Chroma vector database.
    Automatically rebuilds when knowledge-base files change.

    Args:
        chunks: Document chunks to embed
        persist_dir: Directory to store the database
        folder_path: Knowledge-base folder (used for change detection)
        embedding_model: Embedding model name override (None = use env var default)

    Returns:
        Chroma vector store
    """
    embedding_model = get_embedding_model(embedding_model)

    # Reuse existing DB only if docs haven't changed
    chroma_exists = os.path.exists(persist_dir)
    needs_rebuild = _needs_rebuild(persist_dir, folder_path) if chroma_exists else True
    print(f"[RAG] chroma_db exists={chroma_exists}, needs_rebuild={needs_rebuild}, persist_dir={os.path.abspath(persist_dir)}")
    if chroma_exists and not needs_rebuild:
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
        for item in os.listdir(persist_dir):
            item_path = os.path.join(persist_dir, item)
            shutil.rmtree(item_path) if os.path.isdir(item_path) else os.remove(item_path)
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

    print("\n[OK] Vector database built successfully")
    _save_fingerprint(persist_dir, folder_path)
    return vectordb


def semantic_search(vectordb: Chroma, query: str, k: int = 3, search_type: str = "mmr") -> List:
    """
    Perform semantic search using Chroma.

    Args:
        vectordb: Chroma vector database
        query: Search query
        k: Number of results to return
        search_type: "mmr" (diverse) or "similarity" (cosine, faster)

    Returns:
        List of relevant documents
    """
    if search_type == "mmr":
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3}
        )
    else:
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
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
    print("[OK] BM25 retriever ready")
    return bm25_retriever


# =============================================================================
# HYBRID RETRIEVAL (SEMANTIC + KEYWORD)
# =============================================================================

_cached_ensemble = None
_cached_ensemble_key = None  # (k, search_type, weights_tuple)


def hybrid_search(
    vectordb: Chroma,
    bm25_retriever: BM25Retriever,
    query: str,
    k: int = 3,
    weights: list = None,
    search_type: str = "mmr",
) -> List:
    """
    Perform hybrid search combining semantic (Chroma) and keyword (BM25).
    Caches the EnsembleRetriever; rebuilds when k, weights, or search_type change.

    Args:
        vectordb: Chroma vector database
        bm25_retriever: BM25 retriever
        query: Search query
        k: Number of results to return
        weights: [semantic_weight, bm25_weight], default [0.7, 0.3] (Phase 1 sweep winner)
        search_type: "mmr" or "similarity" for the semantic retriever

    Returns:
        List of relevant documents
    """
    global _cached_ensemble, _cached_ensemble_key

    if weights is None:
        weights = [0.7, 0.3]

    cache_key = (k, search_type, tuple(weights))
    if _cached_ensemble is None or _cached_ensemble_key != cache_key:
        if search_type == "mmr":
            semantic_retriever = vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": k * 3}
            )
        else:
            semantic_retriever = vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        bm25_retriever.k = k

        # EnsembleRetriever weights order: [semantic, bm25]
        _cached_ensemble = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=weights
        )
        _cached_ensemble_key = cache_key

    results = _cached_ensemble.invoke(query)
    return results[:k]


# =============================================================================
# LLM RESPONSE GENERATION
# =============================================================================

def format_context(documents: List, max_chars: int = 8000) -> str:
    """
    Format retrieved documents into a context string for the LLM.
    Stops adding chunks once max_chars is reached to protect token budget.
    At ~4 chars per token, 8000 chars ≈ 2000 tokens — sufficient for OpenRouter models.

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
    # ChatOpenAI (OpenRouter) returns an AIMessage; OllamaLLM returns a plain string
    text = response.content if hasattr(response, "content") else response
    # Strip <think>...</think> blocks (Qwen3 and similar reasoning models)
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


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
    sensor_data: Dict = None,
    search_type: str = "mmr",
    weights: list = None,
    llm_model: str = None,
    include_task_context: bool = False,
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
        sensor_data:     Pre-fetched sensor reading (skips DB call if provided)
        search_type:     "mmr" or "similarity" for vector retrieval
        weights:         [semantic_weight, bm25_weight] for hybrid mode (default [0.6, 0.4])
        llm_model:       LLM model override (None = use env var default)

    Returns:
        Dictionary with answer, sources, and metadata
    """

    t_start = time.time()

    # Step 1: Get sensor data if enabled
    sensor_context = None
    has_critical = False
    routing_mode = None
    routing_decision = "disabled"

    if use_sensors:
        if sensor_data is None:
            sensor_data = get_latest_sensor_reading()
        if sensor_data:
            stale = is_reading_stale(sensor_data)
            # Critical alerts only fire on fresh data (don't page on days-old readings)
            critical_alerts = [] if stale else get_critical_alerts(sensor_data)
            has_critical = len(critical_alerts) > 0 and is_environment_query(query)

            # Routing: decide whether sensor context belongs in this query's prompt
            routing_mode = os.getenv("SENSOR_ROUTING_MODE", "llm")
            if routing_mode == "llm":
                include = llm_route_sensors(query, sensor_data, model=llm_model)
            else:
                include = should_include_sensors(query, sensor_data)

            routing_decision = "include" if include else "exclude"
            if include:
                sensor_context = get_sensor_context(sensor_data)

    # Step 2: Retrieve relevant documents
    if use_hybrid:
        documents = hybrid_search(
            vectordb, bm25_retriever, query, k=k, weights=weights, search_type=search_type
        )
        retrieval_method = "hybrid"
    else:
        documents = semantic_search(vectordb, query, k=k, search_type=search_type)
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
    prompt_template = "emergency" if has_critical and sensor_context else "standard"

    if include_task_context:
        task_block = build_task_context()
        if task_block:
            prompt = f"[Recent coop activity — user-requested]\n{task_block}\n\n" + prompt

    if history_prefix:
        prompt = history_prefix + "\n\n" + prompt

    # Step 4: Generate response
    response = generate_response(prompt, model=llm_model)

    response_time_ms = int((time.time() - t_start) * 1000)

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
        "routing_mode": routing_mode,
        "routing_decision": routing_decision,
        "prompt_template": prompt_template,
        "response_time_ms": response_time_ms,
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
    chunks = split_documents(docs, chunk_size=1000)
    
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
