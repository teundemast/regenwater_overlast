import pandas as pd
import json
import requests
import random
import time
from postcode_finder import find_all_postcode6, find_random_housenumber
import rdconverter

def get_lat_lng(houses_postcode, headers):
    try:
        rand_pand = random.choice(houses_postcode["_embedded"]["zoekresultaten"])
        zoekidentificatie = rand_pand["identificatie"]
        response = requests.get(f'https://api.bag.acceptatie.kadaster.nl/lvbag/individuelebevragingen/v2/adressen?zoekresultaatIdentificatie={zoekidentificatie}', headers=headers, timeout=10)
        response = response.json()
        time.sleep(0.2)
        pandidentificatie = response["_embedded"]["adressen"][0]["pandIdentificaties"][0]
        response = requests.get(f'https://api.bag.acceptatie.kadaster.nl/lvbag/individuelebevragingen/v2/verblijfsobjecten?pandIdentificatie={pandidentificatie}', headers=headers, timeout=10)
        time.sleep(0.2)
        coordinates = response.json()["_embedded"]["verblijfsobjecten"][0]["verblijfsobject"]["geometrie"]["punt"]["coordinates"]
        x = coordinates[0]
        y = coordinates[1]
        lat = rdconverter.RD2lat(x, y)
        lng = rdconverter.RD2lng(x, y)
        return((lat, lng))
    except Exception as ex:
        print(ex)
        return None 

def get_random_house_sub(postcode, postcodes_df, headers):
    try:
        random_hn = find_random_housenumber(postcodes_df, postcode)
    except Exception as ex:
        # print(f"Out of range: {postcode}")
        # print(ex)
        return None 
    try:
        houses_postcode = requests.get(f"https://api.bag.acceptatie.kadaster.nl/lvbag/individuelebevragingen/v2/adressen/zoek?zoek={random_hn}&pageSize=100", headers=headers, timeout=10).json()
        time.sleep(0.2)
    except Exception as ex:
        # print(ex)
        try:
            houses_postcode = requests.get(f"https://api.bag.acceptatie.kadaster.nl/lvbag/individuelebevragingen/v2/adressen/zoek?zoek={random_hn}&pageSize=100", headers=headers, timeout=10).json()
            time.sleep(0.2)
        except Exception as ex:
            # print(ex)
            return None
    return houses_postcode
    

def get_random_house(postcode: str, resolution: str, postcodes_df: pd.DataFrame, original_postcode=None, tries=0, ensurance=False):

    # Fill in api key:
    headers = {"X-Api-Key": "APIKEY",
               "Accept-Crs": "epsg:xxxx"}

    if resolution == "postcode6":
        postcode_without_spaces = postcode.replace(" ", "")
        houses_postcode = get_random_house_sub(postcode_without_spaces, postcodes_df, headers)
        if houses_postcode is None:
            houses_postcode = get_random_house_sub(postcode_without_spaces, postcodes_df, headers)
        if houses_postcode is None:
            if original_postcode is not None and tries < 2:
                tries += 1 
                return get_random_house(original_postcode, "postcode4", postcodes_df, original_postcode, tries)
            else:
                if tries >= 2:
                    return None 
                try: 
                    houses_postcode = requests.get(f"https://api.bag.acceptatie.kadaster.nl/lvbag/individuelebevragingen/v2/adressen/zoek?zoek={postcode_without_spaces}&pageSize=100", headers=headers, timeout=10).json()
                    embedded = houses_postcode["_embedded"]["zoekresultaten"]
                except Exception as ex:
                    try: 
                        houses_postcode = requests.get(f"https://api.bag.acceptatie.kadaster.nl/lvbag/individuelebevragingen/v2/adressen/zoek?zoek={postcode}&pageSize=100", headers=headers, timeout=10).json()
                        embedded = houses_postcode["_embedded"]["zoekresultaten"]
                    except Exception:                       
                        return None 
            
        output = get_lat_lng(houses_postcode, headers)
        if output is not None:
            return output 
        time.sleep(0.2)
        if original_postcode is not None and tries < 2:
            if original_postcode is not None and tries < 2:
                tries += 1 
                return get_random_house(original_postcode, "postcode4", postcodes_df, original_postcode, tries)
            else:
                try: 
                    output = get_lat_lng(houses_postcode, headers)
                except Exception as ex:
                    return None 

    if resolution == "postcode4":
        if ensurance:
            postcode4 = postcode 
        else:
            postcode4 = postcode[:-3]
        try:
            possible_postcode6 = find_all_postcode6("all_postcode6.txt", postcode4)
        except KeyError:
            try: 
                possible_postcode6 = find_all_postcode6("all_postcode6.txt", postcode)
            except KeyError:
                if original_postcode is not None and tries < 2:
                    tries += 1 
                    return get_random_house(original_postcode, "postcode4", postcodes_df, original_postcode, tries)
                else:
                    return None 
            
        postcode6 = random.choice(possible_postcode6)

        return get_random_house(postcode6, "postcode6", postcodes_df, postcode, tries)


def change_resolution(dataset: list, out_file_name: str, resolution="postcode6"):
    data = []
    missing = []
    postcodes_df = pd.read_csv("src/data/pc6hnr20210801_gwb.csv", sep=";")
    n = len(dataset)
    for i in dataset:
        n -= 1
        if n % 5 == 0:
            print(n)
        new_item = {}
        new_item["date"] = i["date"]
        for x in i["google_results"]:
            for y in x["address_components"]:
                if "postal_code" in y["types"]:
                    postal_code = y['long_name']
                    if len(postal_code) == 4:
                        resolution="postcode4"
                    else:
                        resolution="postcode6"

        output = get_random_house(postal_code, resolution, postcodes_df)
        if output is None:
            print(postal_code)
            print("FUCK")
            missing.append(i)
            continue

        new_item["lat"] = output[0]
        new_item["lng"] = output[1]
        data.append(new_item)
        
    out_file = open(f"{out_file_name}missing.json", "w")
    json.dump(missing, out_file, indent=6)
    out_file.close()
    
    out_file = open(f"{out_file_name}.json", "w")
    json.dump(data, out_file, indent=6)
    out_file.close()


def load_data(file):
    with open(file, 'r') as file:
        data = json.load(file)
        return data

if __name__ == "__main__":
    dataset = load_data('src/data/merged_precise_coord.json')
    # dataset = dataset[:10]
    # change_resolution(dataset, "postcode6_number1", "postcode6")
    # change_resolution(dataset, "postcode6_number2", "postcode6")
    change_resolution(dataset, "postcode6_number3", "postcode6")
    change_resolution(dataset, "postcode6_number4", "postcode6")
    change_resolution(dataset, "postcode6_number5", "postcode6")



    
    # print(get_random_house("2324MC", "postcode4"))
