import json
import os
import re
import typer
from bs4 import BeautifulSoup
import httpx
from prettyprinter import pprint

from hackathon_genart.ttypes import ArtistData


def extract_artblocks_bio(file_content):
    # Find the relevant JSON object containing the bio
    pattern = r'"individualArtistData":\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
    match = re.search(pattern, file_content)
    
    if not match:
        return "Bio not found"
    
    json_str = "{" + match.group(0) + "}"
    
    try:
        # Parse the JSON data
        data = json.loads(json_str)
        
        # Navigate through the nested structure to find the bio
        bio = data['individualArtistData']['artistEditorialPages']['data'][0]['attributes']['bio']
        return bio
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return f"Error extracting bio: {str(e)}"
    

def get_artblocks_script_tags(text):
    # Parse the HTML content
    soup = BeautifulSoup(text, 'html.parser')
    
    # Find all script tags
    script_tags = soup.find_all('script')
    
    # Extract and print the JSON from script tags starting with self.__next_f.push
    artists = []

    for i, tag in enumerate(script_tags, 1):
        if tag.string and tag.string.strip().startswith('self.__next_f.push'):
            # Extract the JSON part
            json_start = tag.string.find('{')
            json_end = tag.string.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_content = tag.string[json_start:json_end]
                unescaped_json = json_content.encode().decode('unicode_escape')
                try:
                    decoded_json = json.loads(unescaped_json)
                    yield decoded_json                    

                except json.JSONDecodeError as e:
                    continue


def get_artblocks_artist_bio(text):
    for decoded_json in get_artblocks_script_tags(text):
        if 'children' in decoded_json and isinstance(decoded_json['children'], list) and len(decoded_json['children']) > 0 and isinstance(decoded_json['children'][0], list) and len(decoded_json['children'][0]) > 3 and isinstance(decoded_json['children'][0][3], dict) and decoded_json['children'][0][3].get('pageName') == 'Artist':
            if len(decoded_json['children'][1]) < 4:
                continue

            #pprint(decoded_json)

            bio = decoded_json['children'][1][3]["individualArtistData"]["artistEditorialPages"]["data"][0]["attributes"]["bio"]
            userProfileAddress = decoded_json['children'][1][3]["individualArtistData"]["artistEditorialPages"]["data"][0]["attributes"]["userProfileAddress"]
    return bio, userProfileAddress
                

def get_artblocks_artist_index(text):
    artists = []
    for decoded_json in get_artblocks_script_tags(text):
        if ('children' in decoded_json and 
          isinstance(decoded_json['children'], list) and 
          len(decoded_json['children']) > 0 and 
          isinstance(decoded_json['children'][0], list) and 
          len(decoded_json['children'][0]) > 3 and 
          isinstance(decoded_json['children'][0][3], dict) and 
          decoded_json['children'][0][3].get('pageName') == 'Artists grid'):
          if len(decoded_json['children'][1]) < 4:
              continue
          artist_data = decoded_json['children'][1][3]['artistData']
          #prettyprinter.pprint(artist_data)

          for artist in artist_data['artistEditorialPages']['data']:
              attributes = artist['attributes']
              artists.append({
                  'name': attributes['artistName'],
                  'slug': attributes['slug'],
              })                            

    return artists


def load_artblocks_data():
    with open("data/artist_bios.json", 'r') as file:
        artist_bios = json.load(file)
        
        processed_artblocks_data = []
        for address, bio in artist_bios.items():
            processed_artblocks_data.append(ArtistData(
                name=bio["name"],
                bio=bio["bio"],
                addresses=[address]
            ))

    return processed_artblocks_data


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


def fetch_artblocks_collection_data(offset: int = 0):
    url = "https://api.reservoir.tools/search/collections/v2"
    params = {
        "community": "artblocks",
        "offset": offset
    }
    headers = {
        "Accept": "*/*",
        "X-Api-Key": "1ad08f8b-99e2-5317-87d7-e7675997299b"
    }

    try:
        with httpx.Client() as client:
            response = client.get(url, params=params, headers=headers)
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
    except httpx.RequestError as e:
        print(f"An error occurred while requesting: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    
    return result["collections"]


def fetch_artblocks_collection_description(collection_id):
    url = f"https://api.reservoir.tools/tokens/v7?collection={collection_id}&limit=1"
    headers = {
        "Accept": "*/*",
        "X-Api-Key": "1ad08f8b-99e2-5317-87d7-e7675997299b"
    }

    try:
        with httpx.Client() as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result["tokens"][0]["token"]["collection"]["name"], result["tokens"][0]["token"]["description"], result["tokens"][0]["token"]["collection"]["creator"]
    except:
        raise
