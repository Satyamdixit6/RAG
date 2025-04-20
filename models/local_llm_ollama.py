# Assumes Ollama server is running with a model like 'llama2' pulled
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA

loader = TextLoader("path/to/your/document.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="llama2")
vectorstore = FAISS.from_documents(docs, embeddings)

llm = Ollama(model="llama2")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)

query = "What is the document about?"
result = qa_chain.invoke(query)
print(result['result'])