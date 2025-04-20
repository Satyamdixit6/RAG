import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

documents = SimpleDirectoryReader("your_data_directory").load_data()

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

llm = OpenAI(model="gpt-3.5-turbo")
embed_model = OpenAIEmbedding()
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=node_parser)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ]
)

response = query_engine.query("What context surrounds the mention of 'specific keyword'?")
print(response)