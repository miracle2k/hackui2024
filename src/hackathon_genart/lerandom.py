import csv
from llama_index.core import Document
from hackathon_genart.ttypes import ArtistData


def load_lerandom_data():
    with open("data/lerandom.art - Artists.csv", 'r') as file:
        csv_reader = csv.DictReader(file, delimiter=';')
        lerandom_data = list(csv_reader)

        processed_lerandom_data = []
        for item in lerandom_data:
            processed_item = {
                **item,
                "Addresses": [],
            }            
            for i in range(1, 4):
                address = item.get(f"Ethereum Address {i}")
                if address and address.strip() and address.lower() != "null":
                    processed_item["Addresses"].append(address.strip().lower())        
            for i in range(1, 4):
                address = item.get(f"Tezos Address {i}")
                if address and address.strip() and address.lower() != "null":
                    processed_item["Addresses"].append(address.strip())
            processed_lerandom_data.append(ArtistData(
                name=item["Name"], 
                alias=item["Also Known As"], 
                bio=item["Biography"],
                addresses=processed_item["Addresses"]
            ))

    return processed_lerandom_data


def load_additional_artist_context():
    with open("data/Additional Context - Artists (1).csv", 'r') as file:
        csv_reader = csv.DictReader(file, delimiter=',')
        lerandom_data = list(csv_reader)

        processed_lerandom_data = []
        for item in lerandom_data:
            processed_item = {
                **item,
            }            

            processed_lerandom_data.append(ArtistData(
                name=item["Artist"], 
                bio=item["Article Text"],
                addresses=[processed_item["Artist Address"]]
            ))

    return processed_lerandom_data


def load_additional_collection_context():
    documents = []

    with open("data/Additional Context - Collections.csv", 'r') as file:
        csv_reader = csv.DictReader(file)
        collection_data = list(csv_reader)

        for item in collection_data:
            processed_item = {
                "Artist": item["Artist"],
                "Collection": item["Collection"],
                "Blockchain": item["Blockchain"],
                "ArtistAddress": item["Artist Address"],
                "Description": item["Article Text"],
                "SourceURL": item["Source URL"]
            }

            
            doc_text = f"Name: {processed_item['Collection']}\nArtist: {processed_item['Artist']}\nDescription: {processed_item['Description']}"
            documents.append(Document(text=doc_text))

    return documents
