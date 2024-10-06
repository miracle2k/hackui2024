import hashlib
import itertools
import json
import sys
import os
import csv
import time
from typing import List
import httpx
import prettyprinter
import tqdm
import typer
from llama_index.core.schema import ImageNode
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
import rich
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.multi_modal_llms.openai import OpenAIMultiModal


from hackathon_genart.artblocks import extract_artblocks_bio, fetch_artblocks_collection_data, fetch_artblocks_collection_description, get_artblocks_artist_bio, get_artblocks_artist_index, get_artblocks_script_tags, load_artblocks_data
from hackathon_genart.lerandom import load_lerandom_data
from hackathon_genart.minhash import MinHashLSH
from hackathon_genart.objkt import get_objkt_collections_for_artist_address, get_objkt_tokens_for_artist_address


app = typer.Typer()

data_folder = "data"


def get_llm(model: str):
    if model == "mistral":
        return MistralAI(api_key="nugxXB49bvVfX0jz49VKPGFopJUWTTuO", model="mistral-medium")
    elif model == "openai":
        return OpenAI(model="gpt-4o")
    else:
        raise ValueError("Model must be either mistral or openai")


def load_all_artists():
    processed_lerandom_data = load_lerandom_data()
    print(f"lerandom.art: {len(processed_lerandom_data)} artists")
    
    processed_artblocks_data = load_artblocks_data()
    print(f"artblocks.io: {len(processed_artblocks_data)} artists")

    return list(itertools.chain(processed_lerandom_data, processed_artblocks_data))


@app.command()
def dedup_objkt_tokens():
    """Deduplicate objkt tokens using MinHashLSH."""
    data_folder = "data"
    tokens_file = os.path.join(data_folder, "objkt_tokens.json")

    # Load existing tokens
    with open(tokens_file, "r") as f:
        tokens = json.load(f)
    
    print(f"Loaded {len(tokens)} objkt tokens")

    # Initialize MinHashLSH
    lsh = MinHashLSH(num_hashes=128, bands=8)
    
    # Deduplicate tokens
    unique_tokens = []    
    for token in tqdm.tqdm(tokens, desc="Processing tokens", unit="token"):
        if not token['description'] or lsh.has_similar_document(token['description']):
            continue
        lsh.add_document(f"{token['token_id']}/{token['fa_contract']}", token['description'])
        unique_tokens.append(token)

    print(f"Deduplicated to {len(unique_tokens)} unique tokens")

    # Save deduplicated tokens
    output_file = os.path.join(data_folder, "objkt_tokens_deduped.json")
    with open(output_file, "w") as f:
        json.dump(unique_tokens, f, indent=2)

    print(f"Saved deduplicated tokens to {output_file}")


@app.command()
def fetch_images():
    """Fetch all images from the tokens of artblocks and objekt.
    """
    images = []

    # # load all objekt tokens
    # data_folder = "data"
    # with open(os.path.join(data_folder, "objkt_tokens_deduped.json"), "r") as f:
    #     tokens = json.load(f)
    # print(f"Loading {len(tokens)} objekt tokens")
    # for token in tokens:        
    #     images.append(token['thumbnail_uri'])

    # load all artblocks tokens    
    with open(os.path.join(data_folder, "artblocks_collections.json"), "r") as f:
        collections = json.load(f)
    print(f"Loading {len(collections)} artblocks collections")
    for collection in collections.values():
        for image in collection["images"]:
            images.append(image)

    image_dir = os.path.join(data_folder, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    # loop through and download all these images
    for image in tqdm.tqdm(images):
        if not image:
            continue
        filename = hashlib.md5(image.encode()).hexdigest()
        file_path = os.path.join(image_dir, filename)
        if os.path.exists(file_path):
            continue

        
        retries = 0
        max_retries = 3
        while retries < max_retries:
            try:
                content = httpx.get(image, timeout=10).content
                break
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                retries += 1
                if retries == max_retries:
                    print(f"Failed to fetch image after {max_retries} attempts: {image}")
                    continue
                time.sleep(1)  # Wait for 1 second before retrying
        else:
            continue  # Skip this image if all retries failed
        
        # Save the image        
        file_path = os.path.join(image_dir, filename)
        with open(file_path, "wb") as f:
            f.write(content)
        print(f"Saved image: {file_path}")


@app.command()
def describe_images(model: str = typer.Option("openai", help="Model to use: mistral or openai")):
    """Describe all images in the data/images directory using an LLM."""

    def persist():
        os.makedirs(data_folder, exist_ok=True)
        with open(os.path.join(data_folder, "image_descriptions.json"), "w") as f:
            json.dump(image_descriptions, f, indent=2)

    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4o-mini", max_new_tokens=5500
    )

    image_descriptions = {}
    image_dir = os.path.join(data_folder, "images")

    if not os.path.exists(image_dir):
        print(f"Image directory not found at {image_dir}. Please run 'fetch_images' command first.")
        return

    for filename in tqdm.tqdm(os.listdir(image_dir), desc="Processing images"):        
        image_path = os.path.join(image_dir, filename)

        image_documents = [ImageNode(image_path=image_path)]
        
        response_1 = openai_mm_llm.complete(
            prompt="""
Can you please describe the artwork as succinctly as possible by discussing the following:
- color (dominant colors, palette, contrast)
- composition (balance, symmetry, focal points, ...)
- subject matter (abstract, representational, conceptual, ...)
- artistic style (abstract, surreal, impressionist, pixel art, 3D, glitch, ASCII, ...)
- recognised objects (what objects can you identify in the piece? look for human faces, bodies, cats, dogs, fruits, ...)
- orientation (horizontal, vertical, square)
- form (round, square, flow field, ...)
- overall feeling, emotional impact
Your reply has to be a succinct as possible and should only mention specific terms if you are sure they apply to the artwork.

Make sure you quantify your adjectives as much as possible, for example "somewhat vibrant", "very vibrant" etc.

Use a free-flowing text reply, do not use enumeration.
            """,
            image_documents=image_documents,
        )
        
        image_descriptions[filename] = response_1.text

        persist()

    # Save the descriptions to a JSON file
    output_file = os.path.join(data_folder, "image_descriptions.json")
    with open(output_file, "w") as f:
        json.dump(image_descriptions, f, indent=2)

    print(f"Saved {len(image_descriptions)} image descriptions to {output_file}")


@app.command()
def fetch_artblocks_collections(model: str = typer.Option("openai", help="Model to use: mistral or openai")):
    """Fetch all artblocks collections.
    """
    llm = get_llm(model)

    def persist():
        os.makedirs(data_folder, exist_ok=True)
        with open(os.path.join(data_folder, "artblocks_collections.json"), "w") as f:
            json.dump(artblocks_collections, f, indent=2)


    artblocks_collections = {}
    page_size = None
    offset = 0
    while True:
        print(f"Fetching artblocks collections with offset {offset}")
        collections = fetch_artblocks_collection_data(offset)
        #print(collections)
        for collection in collections:
            name, description, creator, images = fetch_artblocks_collection_description(collection["collectionId"])
            artblocks_collections[collection["collectionId"]] = {
                "name": name,
                "description": description,
                "creator": creator,
                "images": images,
            }

        if not page_size:
            page_size = len(collections)
        if len(collections) < page_size:
            break
        offset += page_size
        
        persist()

    print(f"Fetched {len(collections)} artblocks collections")
    

@app.command()
def index_objkt(model: str = typer.Option("openai", help="Model to use: mistral or openai")):
    """Fetch all objekt colletions for our known artists.
    """
    artists = load_all_artists()
    all_tokens = []
    all_collections = []

    for artist in artists:
        tezos_addresses = [addr for addr in artist.addresses if addr.startswith('tz')]
        if not tezos_addresses:
            continue
        
        print(f"Processing artist: {artist.name}")        
        for address in tezos_addresses:
            print(f"  Fetching tokens / collections for Tezos address: {address}")
            tokens = get_objkt_tokens_for_artist_address(address)
            if tokens:
                print(f"    Found {len(tokens)} tokens")
                all_tokens.extend(tokens)

            collections = get_objkt_collections_for_artist_address(address)
            if collections:
                print(f"    Found {len(collections)} collections")
                all_collections.extend(collections)

            import time; time.sleep(0.5)
    
    # Save all tokens to a JSON file in the data folder
    os.makedirs(data_folder, exist_ok=True)
    tokens_file_path = os.path.join(data_folder, "objkt_tokens.json")
    
    print(f"Saving {len(all_tokens)} tokens to {tokens_file_path}")
    
    with open(tokens_file_path, "w") as f:
        json.dump(all_tokens, f, indent=2)

    collections_file_path = os.path.join(data_folder, "objkt_collections.json")
    print(f"Saving {len(all_collections)} collections to {collections_file_path}")
    with open(collections_file_path, "w") as f:
        json.dump(all_collections, f, indent=2)


@app.command()
def build_artist_index(model: str = typer.Option("openai", help="Model to use: mistral or openai")):
    """Build artist index.
    """
    llm = get_llm(model)
    
    artists = load_all_artists()
    documents = [Document(text=item.bio) for item in artists]

    debug_handler = LlamaDebugHandler()
    callback_manager = CallbackManager([debug_handler])

    print('creating index')
    index = VectorStoreIndex.from_documents(documents, callback_manager=callback_manager, llm=llm)    
    # Store the index on disk
    index.storage_context.persist(persist_dir="./stored_artist_index")
    print('Index built and stored successfully')


def load_objkt_collections():
    ################ objekt 
    data_folder = "data"
    collections_file_path = os.path.join(data_folder, "objkt_collections.json")
    
    if not os.path.exists(collections_file_path):
        print(f"Collections file not found at {collections_file_path}. Please run 'index_objkt' command first.")
        return
    
    with open(collections_file_path, "r") as f:
        collections = json.load(f)

    print(f"Loading {len(collections)} objkt collections")
    
    documents = []
    
    for collection in collections:
        doc_text = f"Name: {collection['name']}\nDescription: {collection['description']}"
        documents.append(Document(text=doc_text))
    return documents


def load_artblocks_collections():
    ################ artblocks 
    data_folder = "data"
    collections_file_path = os.path.join(data_folder, "artblocks_collections.json")
    
    if not os.path.exists(collections_file_path):
        print(f"Collections file not found at {collections_file_path}. Please run 'fetch_artblocks_collections' command first.")
        return
    
    with open(collections_file_path, "r") as f:
        collections = json.load(f)

    print(f"Loading {len(collections)} artblocks collections")
    
    documents = []
    for collection in collections.values():
        doc_text = f"Name: {collection['name']}\nDescription: {collection['description']}"
        documents.append(Document(text=doc_text))

    return documents


@app.command()
def build_collection_index(model: str = typer.Option("openai", help="Model to use: mistral or openai")):
    """Build collection index."""
    llm = get_llm(model)
    
    documents = load_objkt_collections()
    documents.extend(load_artblocks_collections())

    debug_handler = LlamaDebugHandler()
    callback_manager = CallbackManager([debug_handler])

    print('Creating collection index')
    index = VectorStoreIndex.from_documents(documents, callback_manager=callback_manager, llm=llm)    
    # Store the index on disk
    index.storage_context.persist(persist_dir="./stored_collection_index")
    print('Collection index built and stored successfully')


@app.command()
def build_token_index(model: str = typer.Option("openai", help="Model to use: mistral or openai")):
    """Build token index."""
    from hackathon_genart.minhash import MinHashLSH
    
    llm = get_llm(model)
    
    data_folder = "data"
    tokens_file_path = os.path.join(data_folder, "objkt_tokens.json")
    
    if not os.path.exists(tokens_file_path):
        print(f"Token file not found at {tokens_file_path}. Please run 'index_objkt' command first.")
        return
    
    with open(tokens_file_path, "r") as f:
        tokens = json.load(f)

    print(f"Loading {len(tokens)} tokens")
    
    lsh = MinHashLSH(num_hashes=128, bands=8)
    documents = []
    
    from tqdm import tqdm
    for token in tqdm(tokens, desc="Processing tokens", unit="token"):
        description = token['description']
        doc_id = token['token_id'] + "/" + token['fa_contract']
        if description:
            if lsh.has_similar_document(description):
                continue
            lsh.add_document(doc_id, description)
        documents.append(Document(text=f"Name: {token['name']}\nDescription: {description}\nToken ID: {doc_id}"))

    print(f"Created {len(documents)} unique documents")

    debug_handler = LlamaDebugHandler()
    callback_manager = CallbackManager([debug_handler])

    print('Creating token index')
    index = VectorStoreIndex.from_documents(documents, callback_manager=callback_manager, llm=llm)    
    # Store the index on disk
    index.storage_context.persist(persist_dir="./stored_token_index")
    print('Token index built and stored successfully')

    # # Store the LSH index
    # with open("./stored_token_lsh_index.pkl", "wb") as f:
    #     pickle.dump(lsh, f)
    # print('LSH index stored successfully')


"""
artist who creates red collections



"""


@app.command()
def search(query: str, model: str = typer.Option("openai", help="Model to use: mistral or openai"), synthesize: bool = typer.Option(False, help="Use response synthesizer mode"),
           dataset: str = typer.Option(..., help="Dataset to search: artist, collection, or token")):
    llm = get_llm(model)    

    print(f'loading {dataset} index')
    if dataset == 'artist':
        filename = "./stored_artist_index"
    elif dataset == 'collection':
        filename = "./stored_collection_index"
    elif dataset == 'token':
        filename = "./stored_token_index"
    else:
        raise ValueError("Dataset must be either artist, collection, or token")
    
    if not os.path.exists(filename):
        print(f'Index not found at {filename}. Please build the index first using the "build-{dataset}-index" command.')
        return

    storage_context = StorageContext.from_defaults(persist_dir=filename)
    index = load_index_from_storage(storage_context)

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )

    cohere_rerank = CohereRerank(api_key="9le3nZvIS8VVT36L2ot7YY7wHfxKV8GTOgZ2hQj7", top_n=6)

    if synthesize:
        # configure response synthesizer
        response_synthesizer = get_response_synthesizer(llm=llm)

        # assemble query engine with synthesizer
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.7),
                #cohere_rerank
            ],
        )

        print()
        response = query_engine.query(query)

        for i, node in enumerate(response.source_nodes):
            print(f"Node {i + 1}:")
            print(f"Content: {node.get_content()}")
            print(f"Score: {node.get_score()}")
            print("---")

        print(response)
    else:
        # retrieve nodes
        nodes = retriever.retrieve(query)

        print("BEFORE -----------------------")
        for node in nodes:
            print(f"Content: {node.get_content()}")
            print(f"Score: {node.get_score()}")
            print("---")

        print("              ")
        print("              ")
        print("              ")
        print("              ")


        nodes = cohere_rerank.postprocess_nodes(nodes, query_str=query)
        print("AFTER -------------------------")
        for node in nodes:
            print(f"Content: {node.get_content()}")
            print(f"Score: {node.get_score()}")
            print("---")

        return

        # ask LLM for each retrieval result
        for i, node in enumerate(nodes):
            content = node.get_content()
            prompt = f"A user searching an artist database for '{query}, might they specifically be looking for the following artist? \n\nContent: {content}\n\nPlease answer with 'Yes' or 'No' and provide a brief explanation."

            rich.print(f"[green]Node {i + 1}:[/green]")

            llm_response = llm.complete(prompt)
            if "No" in llm_response.text:
                rich.print(f"[dim]LLM Evaluation: {llm_response.text.replace('No', '[red]No[/red]')}[/dim]")
                rich.print(f"[dim]Content: {content}[/dim]")
                rich.print(f"[dim]Score: {node.get_score()}[/dim]")
                pass
            else:
                rich.print(f"LLM Evaluation: {llm_response.text.replace('Yes', '[green]Yes[/green]')}")
                rich.print(f"Content: {content}")
                rich.print(f"Score: {node.get_score()}")
            
            print("---")
            print("")


if __name__ == "__main__":
    app()