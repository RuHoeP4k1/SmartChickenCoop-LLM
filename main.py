import os
import json
from pathlib import Path  

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_ollama.llms import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

import gradio as gr
import re



model_name = "qwen3:4b-instruct"

#RAG_pipeline
#what to install
# Install langchain & embeddings support: pip install langchain langchain-community langchain-text-splitters
# Install your local vector database: pip install chromadb  # easiest + local
# For PDFs, text, etc: pip install pymupdf python-docx tiktoken

#Creating a folder(later) and loading documents


def clean_text(text: str) -> str:
    if not text:
        return ""

    # Remove null bytes and bad unicode
    text = text.replace("\x00", "").replace("\u0000", "")

    # Normalize line breaks / spacing
    text = text.replace("\r", " ").replace("\n", " ")

    # Collapse multiple spaces
    text = " ".join(text.split())

    # Remove extremely long repeated characters (OCR noise)
    text = re.sub(r"(.)\1{10,}", r"\1", text)

    # Hard cap: never allow extremely long text blobs
    if len(text) > 3000:
        text = text[:3000]

    return text


def load_docs(folder_path: str):
    documents = []

    # TXT files
    for file_path in Path(folder_path).glob("*.txt"):
        loader = TextLoader(str(file_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file_path.name
            doc.page_content = clean_text(doc.page_content)
        documents.extend(docs)

    # PDF files
    for file_path in Path(folder_path).glob("*.pdf"):
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file_path.name
            doc.page_content = clean_text(doc.page_content)
        documents.extend(docs)

    return documents


#Sensor data uit json file lezen

def load_sensor_data(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

#Splitting these documents

def split_docs(documents, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    # Enforce a strict limit to avoid embedding crashes
    for c in chunks:
        if len(c.page_content) > 2500:
            c.page_content = c.page_content[:2500]

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


#Creating embeddings for these chunks and storing in vector database

def vector_store(chunks, persist_dir="chroma_db"):
    """Create a Chroma vector store with safe batched embeddings for Windows/Ollama."""

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    # Try loading existing DB
    if os.path.exists(persist_dir):
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )

    print("No existing DB found — creating new embeddings...")
    print(f"Total chunks: {len(chunks)}")

    # Extract text
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # Batch embedding
    batch_size = 10 
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = embedding_model.embed_documents(batch)
        all_embeddings.extend(emb)

    # Create vector DB manually
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )

    vectordb.add_texts(texts=texts, metadatas=metadatas, embeddings=all_embeddings)
    vectordb._persist()

    print("Embedding finished successfully.")
    return vectordb


# let's start building our basic pipeline 

def build_QA_pipeline(vector_store):
    """
    Build a modern English-only RAG pipeline using LCEL.
    """

    # 1. Retriever - fetch relevant document chunks
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,          # number of final retrieved chunks
            "fetch_k": 15,   # number of candidates to consider first
            "lambda_mult": 0.95   # relevance vs diversity balance
        }
    )

    # 2. LLM - Qwen 4B via Ollama
    llm = OllamaLLM(
        model="qwen2.5:1.5b-instruct",
        temperature=0.4,
        num_predict=300
    )

    # 3. RAG Prompt (promt how the system should behave and what user input and contect is available)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful AI assistant. That help users with questions about their chickens. "
         "Use ONLY the provided document excerpts to answer the question. "
         "answer must have a be suggest clear and actionale advice to the user. do not make eternally long lists. "
         "When making list of observed issues and actions use new lines and bullet points to increase the readablility "
         "If the answer cannot be found in the documents, say so. "),
        ("human",
         "Question: {input}\n\n"
         "Use the following documents:\n\n"
         "{context}")
    ])

    # 4. Stuffing chain (LLM fills answer using retrieved docs)
    combine_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    # 5. Wrap into a full Retrieval-Augmented Generation chain
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_chain
    )

    return rag_chain
def list_sensor_files(folder="Data/sensors"):
    """
    Returns a list of all .json sensor dataset filenames
    inside the Data/sensors/ folder.
    """
    return [f.name for f in Path(folder).glob("*.json")]

def load_sensor_file(file_name, folder="Data/sensors"):
    """
    Load a specific sensor JSON file by filename.
    """
    full_path = Path(folder) / file_name
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def interpret_sensor_data(sensor_data: dict) -> str:
    """
    Convert raw sensor JSON into a clean English summary
    with severity icons (✔️ OK, ⚠️ Warning, ❗ Critical).
    """

    messages = []

    # ===== HEAT STRESS =====
    hsi = sensor_data.get("heat_stress_index", "unknown").lower()
    if hsi in ["warning", "moderate"]:
        messages.append(f"⚠️ Heat stress level: WARNING")
    elif hsi in ["critical", "high"]:
        messages.append(f"❗ Heat stress level: CRITICAL")
    elif hsi == "normal":
        messages.append("✔️ Heat stress: Normal")
    else:
        messages.append(f"Heat stress index: {hsi}")

    # ===== TEMPERATURE =====
    temp = sensor_data.get("temperature_c", None)
    if temp is not None:
        if temp > 28:
            messages.append(f"⚠️ High temperature: {temp} °C")
        elif temp < 15:
            messages.append(f"⚠️ Low temperature: {temp} °C")
        else:
            messages.append(f"✔️ Temperature: {temp} °C")

    # ===== HUMIDITY =====
    hum = sensor_data.get("humidity_pct", None)
    if hum is not None:
        if hum > 70:
            messages.append(f"⚠️ High humidity: {hum}%")
        elif hum < 30:
            messages.append(f"⚠️ Low humidity: {hum}%")
        else:
            messages.append(f"✔️ Humidity: {hum}%")

    # ===== FEEDER =====
    feed = sensor_data.get("feeder_status", "unknown").lower()
    if feed in ["full", "ok", "okay", "normal"]:
        messages.append("✔️ Feeder: Full")
    elif feed in ["low", "needs refill", "empty"]:
        messages.append(f"⚠️ Feeder: {feed.capitalize()}")
    else:
        messages.append(f"Feeder status: {feed}")

    # ===== WATER =====
    water = sensor_data.get("waterer_status", "unknown").lower()
    if water in ["full", "ok", "okay", "normal"]:
        messages.append("✔️ Waterer: Full")
    elif water in ["low", "needs refill", "empty"]:
        messages.append(f"⚠️ Waterer: {water.capitalize()}")
    else:
        messages.append(f"Waterer status: {water}")

    return "\n".join(messages)



def build_query_with_data(sensor_summary: str, user_question: str) -> str:
    """
    Build the final RAG query using a clean,
    interpreted summary of the sensor data.
    """
    return (
        "Use the following real-time sensor information from the chicken coop "
        "together with the retrieved documents to answer the question.\n\n"
        f"Sensor status:\n{sensor_summary}\n\n"
        f"Question:\n{user_question}"
    )



"""
def answer_from_dataset(folder_path: str, query: str):
    
    # 1. Load & split docs
    documents = load_docs(folder_path)
    chunks = split_docs(documents)

    # 2. Build / load vector store
    vector_store = create_vector_store(chunks)

    # 3. Build QA pipeline
    qa_pipeline = build_QA_pipeline(vector_store)

    # 4. Ask question
    result = qa_pipeline.invoke({"query": query})

    # 5. Print nicely
    print(f"\n=== Vraag ===\n{query}\n")
    print("=== Antwoord ===")
    print(result["result"])

    print("\n=== Bronnen ===")
    for i, doc in enumerate(result["source_documents"], start=1):
        print(f"\nBron {i}:")
        print("Bestand:", doc.metadata.get("source"))
        print("Tekstfragment:", doc.page_content[:500], "...")

    return result
"""

def answer_with_realtime_data(folder_path: str, json_path: str, user_question: str):
    # 1. Load documents and create vector store
    documents = load_docs(folder_path)
    chunks = split_docs(documents)
    print("Number of chunks:", len(chunks))
    vs = vector_store(chunks)    
    qa_pipeline = build_QA_pipeline(vs)

    # 2. Load sensor data
    sensor_data = load_sensor_data(json_path)

    # 3. Build enriched query with sensor data included
    sensor_summary = interpret_sensor_data(sensor_data)
    query = build_query_with_data(sensor_summary, user_question)


    # 4. Run RAG pipeline
    result = qa_pipeline.invoke({"input": query})  

    print("\n=== Query sent to RAG ===")
    print(query)

    print("\n=== Answer ===")
    print(result["answer"])            # FIXED

    print("\n=== Sources ===")
    for i, doc in enumerate(result["context"], start=1):   # FIXED
        print(f"\nSource {i}:")
        print("File:", doc.metadata.get("source"))

    return result


if __name__ == "__main__":
    folder = "test_docs"
    sensor_json = "data/sensor_data.json"
    query = "Waar wordt machine learning toegepast?"  # simpele test prompt

    answer_with_realtime_data(folder, sensor_json, query)

