from maps_api import MapsApi

import os
import json

def load_data(file):
    with open(file, 'r') as file:
        data = json.load(file)
        return data

key = os.environ['maps_key']
api = MapsApi(key=key)

google_results = api.get_coordinate("Tuinderijpad 7, Leiden, Netherlands")
print(google_results)