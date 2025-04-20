import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

documents = SimpleDirectoryReader("your_data_directory").load_data()

llm = OpenAI(model="gpt-3.5-turbo")
embed_model = OpenAIEmbedding()

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What is the core idea presented?")
print(response)