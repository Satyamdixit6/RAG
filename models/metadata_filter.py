import os
import chromadb
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
persist_directory = './my_chroma_db_metadata'
client = chromadb.PersistentClient(path=persist_directory)
embeddings = OpenAIEmbeddings()

collection = client.get_or_create_collection("docs_with_metadata", embedding_function=chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-ada-002"))

docs_with_meta = [
    Document(page_content="Langchain is great for building LLM apps.", metadata={"source": "blog", "year": 2023}),
    Document(page_content="RAG combines retrieval and generation.", metadata={"source": "paper", "year": 2024}),
    Document(page_content="Vector stores like Chroma are useful.", metadata={"source": "blog", "year": 2024})
]

ids = [f"doc_{i}" for i in range(len(docs_with_meta))]
texts = [doc.page_content for doc in docs_with_meta]
metadatas = [doc.metadata for doc in docs_with_meta]

if not collection.get(ids=ids)['ids']:
     collection.add(
         documents=texts,
         metadatas=metadatas,
         ids=ids
     )

vectorstore = Chroma(
    client=client,
    collection_name="docs_with_metadata",
    embedding_function=embeddings,
)

llm = ChatOpenAI()
retriever = vectorstore.as_retriever(search_kwargs={'filter': {'year': 2024}})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever
)

query = "What was discussed in 2024?"
result = qa_chain.invoke(query)
print(result['result'])