import ollama
import json
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
model_name = "qwen3:4b-instruct"
'''prompt = "how do i make a cake"

response = ollama.generate(
    model=model_name,
    prompt=prompt,
    options={
        "temperature": 0.7,
        "top_p": 0.9,
        "num_predict": 500
    }
)

print(response.response)'''

#RAG_pipeline
#what to install
# Install langchain & embeddings support: pip install langchain langchain-community langchain-text-splitters
# Install your local vector database: pip install chromadb  # easiest + local
# For PDFs, text, etc: pip install pymupdf python-docx tiktoken

#Creating a folder(later) and loading documents

def load_docs(folder_path):
    txt_loader = DirectoryLoader(folder_path, glob="*.txt", loader_cls=TextLoader)
    #pdf_loader = DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyMuDFLoader)
    documents = txt_loader.load() #+ pdf_loader.load()
    return documents

#Sensor data uit json file lezen

def load_sensor_data(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


#Splitting these documents

def split_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks

#Creating embeddings for these chunks and storing in vector database

def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b") #ollama pull qwen3-embedding:0.6b
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    return vector_store

# let's start building our basic pipeline 

def build_QA_pipeline(vector_store):

    retriever = vector_store.as_retriever(
        search_type="mmr",  #Maximal Marginal Relevance zoekalgoritme
        search_kwargs={
            "k": 3,  #hoeveel bronnen uiteindelijk worden gebruikt
            "fetch_k": 15,  #hoeveel kandidaten eerst ophalen
            "lambda_mult": 0.95  #balans: relevant vs. divers
        }
    )

    llm = OllamaLLM(
        model="qwen3:4b-instruct",
        temperature=0.7,
        num_predict=500
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

def build_query_with_data(sensor_data: dict, user_question: str) -> str:
    """
    Bouwt een query waarin de realtime data
    expliciet verwerkt zit.
    """
    hsi = sensor_data.get("heat_stress_index", "onbekend")
    temp = sensor_data.get("temperature_c", "onbekend")
    hum = sensor_data.get("humidity_pct", "onbekend")

    query = (
        f"De huidige sensordata voor het kippenhok is als volgt:\n"
        f"- Heat Stress Index (HSI): {hsi}\n"
        f"- Temperatuur: {temp} °C\n"
        f"- Luchtvochtigheid: {hum} %\n\n"
        f"Gebruik deze actuele waarden samen met de kennis uit de documenten "
        f"om de volgende vraag te beantwoorden:\n"
        f"{user_question}"
    )
    return query

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
    # 1. Laad documenten en maak de vector store
    documents = load_docs(folder_path)
    chunks = split_docs(documents)
    vector_store = create_vector_store(chunks)
    qa_pipeline = build_QA_pipeline(vector_store)

    # 2. Laad sensordata
    sensor_data = load_sensor_data(json_path)

    # 3. Bouw query met HSI erin
    query = build_query_with_data(sensor_data, user_question)

    # 4. Vraag aan RAG-pipeline
    result = qa_pipeline.invoke({"query": query})

    print("\n=== Query die naar de RAG ging ===")
    print(query)

    print("\n=== Antwoord ===")
    print(result["result"])

    print("\n=== Bronnen ===")
    for i, doc in enumerate(result["source_documents"], start=1):
        print(f"\nBron {i}:")
        print("Bestand:", doc.metadata.get("source"))
        print("Tekstfragment:\n")
        print(doc.page_content[:800], "...\n")

    return result


if __name__ == "__main__":
    folder = "test_docs"
    sensor_json = "data/sensor_data.json"
    query = "Waar wordt machine learning toegepast?"  # simpele test prompt

    answer_with_realtime_data(folder, sensor_json, query)

