import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
persist_directory = './my_chroma_db'

loader = TextLoader("path/to/another/document.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)
vectorstore.persist()


llm = ChatOpenAI()
persisted_vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=persisted_vectorstore.as_retriever()
)

query = "What details are provided about project X?"
result = qa_chain.invoke(query)
print(result['result'])