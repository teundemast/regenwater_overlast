import json 

def load_data(file):
    with open(file, 'r') as file:
        data = json.load(file)
        return data['data']