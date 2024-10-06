import httpx


def get_objkt_tokens_for_artist_address(artist_address: str):
    query = """
        query MyQuery {
            token_creator(
                offset: 100,
                where: {creator_address: {_eq: "%(artist_address)s"}}
            ) {
                    token {
                    name
                    fa_contract
                    token_id,
                    description,
                    dimensions,
                    thumbnail_uri
                }
            }
        }
    """ % {"artist_address": artist_address}

    client = httpx.Client()
    response = client.post("https://data.objkt.com/v3/graphql", json={"query": query})
    response.raise_for_status()

    tokens = response.json()['data']['token_creator']
    if len(tokens) == 0:
        return []
    
    result = []
    for token in tokens:
        result.append({
            "token_id": token['token']['token_id'],
            "name": token['token']['name'],
            "description": token['token']['description'],
            "fa_contract": token['token']['fa_contract'],
            #"dimensions": token['token']['dimensions'],
            "thumbnail_uri": token['token']['thumbnail_uri'],
        })
    return result
    

def get_objkt_collections_for_artist_address(artist_address: str):
    query = """
    query MyQuery {
        fa(
            limit: 100
            where: {creator_address: {_eq: "%(artist_address)s"}}
        ) {
            name
            creator_address
            description
        }
    }
    """ % {"artist_address": artist_address}

    client = httpx.Client()
    response = client.post("https://data.objkt.com/v3/graphql", json={"query": query})
    response.raise_for_status()

    collections = response.json().get('data', {}).get('fa', [])
    if not collections:
        return []
    
    result = []
    for collection in collections:
        result.append({
            "name": collection['name'],
            "creator_address": collection['creator_address'],
            "description": collection['description'],
        })
    return result


#@app.command()
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