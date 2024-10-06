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
