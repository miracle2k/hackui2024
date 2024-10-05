import json
import sys
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core import Document
from prettyprinter import pprint
import csv

import logging
import sys

from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import llama_index.core
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.openai import OpenAI


llama_index.core.set_global_handler("simple")

with open("data/lerandom.art - Artists.csv", 'r') as file:
    csv_reader = csv.DictReader(file, delimiter=';')
    data = list(csv_reader)

#prompt = input("Please type your query: ")
prompt = "what artist name sounds like a child's toy"

documents = [Document(text=json.dumps(item)) for item in data]


debug_handler = LlamaDebugHandler()
callback_manager = CallbackManager([debug_handler])

model = "openai"
llm_to_use = None

if model == "mistral":
    mistral_llm = MistralAI(api_key="nugxXB49bvVfX0jz49VKPGFopJUWTTuO", model="mistral-medium")
    llm_to_use = mistral_llm
elif model == "openai":
    openai_llm = OpenAI(model="gpt-4o")
    print(openai_llm.model)
    llm_to_use = openai_llm
else:
    raise ValueError("Model must be either mistral or openai")


### INDEX ############################################################## 
if os.path.exists("./stored_index"):
    print('loading index')
    storage_context = StorageContext.from_defaults(persist_dir="./stored_index")
    index = load_index_from_storage(storage_context)
else:
    print('creating index')
    index = VectorStoreIndex.from_documents(documents, callback_manager=callback_manager, llm=llm_to_use)    
    # Store the index on disk
    index.storage_context.persist(persist_dir="./stored_index")


### SEARCH ############################################################## 

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer(llm=llm_to_use)

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

response = query_engine.query("Artists who started using AI before 2020")

# print("\nLLM Inputs:")
# for event in debug_handler.get_llm_inputs_outputs():
#     print(event)