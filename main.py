import ollama
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

#Splitting these documents

def split_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks

#Creating embeddings for these chunks and storing in vector database

def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    return vector_store

# let's start building our basic pipeline 

def build_QA_pipeline(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

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

#Putting it all together in a simple RAG pipeline
documents = load_docs("test_docs")
chunks = split_docs(documents)
vector_store = create_vector_store(chunks)
qa_pipeline = build_QA_pipeline(vector_store)
query = "Waar wordt machine learning toegepast?"
result = qa_pipeline.invoke({"query": query})
print("Answer:", result['result'])
print("/nSources:", result['source_documents'])
