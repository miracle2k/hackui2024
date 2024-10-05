import json
import sys
import os
import csv
import httpx
import prettyprinter
import typer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core import Document
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import llama_index.core
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.openai import OpenAI
from llama_index.core import SummaryIndex
from llama_index.readers.web import SimpleWebPageReader
import os
from bs4 import BeautifulSoup

from hackathon_genart.artblocks import extract_artblocks_bio, get_artblocks_artist_bio, get_artblocks_artist_index, get_artblocks_script_tags


app = typer.Typer()


llama_index.core.set_global_handler("simple")


def get_llm(model: str):
    if model == "mistral":
        return MistralAI(api_key="nugxXB49bvVfX0jz49VKPGFopJUWTTuO", model="mistral-medium")
    elif model == "openai":
        return OpenAI(model="gpt-4o")
    else:
        raise ValueError("Model must be either mistral or openai")


@app.command()
def index_artblocks(model: str = typer.Option("openai", help="Model to use: mistral or openai")):
    http_client = httpx.Client()

    overview_page = "https://www.artblocks.io/curated/artists"
    response = http_client.get(overview_page)
    artists = get_artblocks_artist_index(response.text)
    
    artist_bios = {}

    def persists():
        # Store all the artist bios in a file in the data folder
        data_folder = "data"
        os.makedirs(data_folder, exist_ok=True)    
        bio_file_path = os.path.join(data_folder, "artist_bios.json")    
        print(f"Saving {len(artist_bios)} artist bios to {bio_file_path}")    
        with open(bio_file_path, "w") as f:
            json.dump(artist_bios, f, indent=2)    

    for artist in artists:
        print("indexing artist", artist)

        response = http_client.get(f"https://www.artblocks.io/curated/artists/{artist['slug']}")
        text = response.text

        bio, addreess = get_artblocks_artist_bio(text)
        artist_bios[addreess] = {
            "bio": bio,
            "slug": artist["slug"],
            "name": artist["name"]
        }

        persists()

    print(f"Artist bios saved successfully")
    



@app.command()
def build(model: str = typer.Option("openai", help="Model to use: mistral or openai")):
    llm = get_llm(model)
    
    with open("data/lerandom.art - Artists.csv", 'r') as file:
        csv_reader = csv.DictReader(file, delimiter=';')
        data = list(csv_reader)

    documents = [Document(text=json.dumps(item)) for item in data]

    debug_handler = LlamaDebugHandler()
    callback_manager = CallbackManager([debug_handler])

    print('creating index')
    index = VectorStoreIndex.from_documents(documents, callback_manager=callback_manager, llm=llm)    
    # Store the index on disk
    index.storage_context.persist(persist_dir="./stored_index")
    print('Index built and stored successfully')


@app.command()
def search(query: str, model: str = typer.Option("openai", help="Model to use: mistral or openai")):
    llm = get_llm(model)

    if not os.path.exists("./stored_index"):
        print('Index not found. Please build the index first using the "build-index" command.')
        return

    print('loading index')
    storage_context = StorageContext.from_defaults(persist_dir="./stored_index")
    index = load_index_from_storage(storage_context)

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(llm=llm)

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    )

    response = query_engine.query(query)
    print(response)


if __name__ == "__main__":
    app()