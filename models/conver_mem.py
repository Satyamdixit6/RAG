import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

loader = TextLoader("path/to/your/document.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
llm = ChatOpenAI()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

query1 = "What is Langchain?"
result1 = qa_chain.invoke({"question": query1})
print("Q1:", result1['answer'])

query2 = "How does it relate to vector stores?"
result2 = qa_chain.invoke({"question": query2})
print("Q2:", result2['answer'])