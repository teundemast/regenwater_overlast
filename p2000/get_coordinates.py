from maps_api import MapsApi

import os
import json


def load_data(file):
    with open(file, 'r') as file:
        data = json.load(file)
        return data


def print_data():
    data = load_data('parsed.json')
    for d in data:
        if len(d['address'].split(' ')) <= 1:
            print(d['message'], "-----", d['address'], "----- ")


def print_json(data):
    to_print = ''
    to_print += '['
    for d in data:
        to_print += json.dumps(d) + ",\n"

    to_print = to_print[0:-2]
    to_print += ']\n'
    print(to_print)


def main():
    key = os.environ['maps_key']
    api = MapsApi(key=key)
    data = load_data('parsed.json')['data']
    data_coord = []
    for d in data:
        google_results = None
        try:
            google_results = api.get_coordinate(d['address'] + ", Netherlands")
        except Exception as e:
            try:
                google_results = api.get_coordinate(
                    d['address'] + ", Netherlands")
            except Exception as e:
                try:
                    google_results = api.get_coordinate(
                        d['address'] + ", Netherlands")
                except Exception as e:
                    pass

        if google_results and google_results['status'] == 'OK':
            d['google_results'] = google_results['results']
            data_coord.append(json.dumps(d))
        # else:
        #     print("ERROR")
        #     print(json.dumps(d) + ",")
            
    out_file = open("coord.json", "w")
    data = {"data": data_coord}
    json.dump(data, out_file, indent=6)
    out_file.close()



if __name__ == '__main__':
    # print_data()
    main()
