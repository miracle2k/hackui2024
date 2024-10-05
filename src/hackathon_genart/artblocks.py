import json
import re

from bs4 import BeautifulSoup
from prettyprinter import pprint


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