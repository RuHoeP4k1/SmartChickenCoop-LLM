import ollama
from langchain.document_loaders import DirectoryLoader, TextLoader, PyMuDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

model_name = "qwen3:4b-instruct"
prompt = "how do i make a cake"

response = ollama.generate(
    model=model_name,
    prompt=prompt,
    options={
        "temperature": 0.7,
        "top_p": 0.9,
        "num_predict": 500
    }
)

print(response.response)

#RAG_pipeline
#what to install
# Install langchain & embeddings support: pip install langchain langchain-community langchain-text-splitters
# Install your local vector database: pip install chromadb  # easiest + local
# For PDFs, text, etc: pip install pymupdf python-docx tiktoken

#Creating a folder(later) and loading documents

def load_docs(folder_path):
    txt_loader = DirectoryLoader(folder_path, glob="*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyMuDFLoader)
    documents = txt_loader.load() + pdf_loader.load()
    return documents

#Splitting these documents

def split_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks

#Creating embeddings for these chunks and storing in vector database

def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(model_name="qwen3:4b-instruct")
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    return vector_store

# let's start building our basic pipeline 

def build_QA_pipeline(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 1}) #k is the number of relevant documents that are retrieved (how much context will the LLM receive)
    qa_chain = RetrievalQA.from_chain_type(
        llm=ollama.Ollama(model_name="qwen3:4b-instruct"),
        chain_type="stuff", #stuff = all chunks are put together and sent to the LLM ; other options are: map_reduce (chuks are summarized), refine(stepwise), map_rerank(model ranks chunks based on relevance)
        retriever=retriever, 
        return_source_documents=True
    )
    return qa_chain

#Putting it all together
documents = load_docs("test_docs")
chunks = split_docs(documents)
vector_store = create_vector_store(chunks)
qa_pipeline = build_QA_pipeline(vector_store)
query = "What is Ollama?"
result = qa_pipeline.run(query)
print("Answer:", result['result'])
print("/nSources:", result['source_documents'])
