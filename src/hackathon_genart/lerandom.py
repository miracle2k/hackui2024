import csv

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