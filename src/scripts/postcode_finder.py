import pandas as pd
import json 
import random
 
# Download first all postcode6 and housenumbers: https://www.cbs.nl/nl-nl/maatwerk/2021/36/buurt-wijk-en-gemeente-2021-voor-postcode-huisnummer
def create_postcode6_file(output_file:str):
    data = {}

    postcode_df = pd.read_csv("src/data/pc6hnr20210801_gwb.csv", sep=";")

    all_postcode6 = set(postcode_df["PC6"].tolist())
    n = len(all_postcode6)

    for postcode6 in all_postcode6:
        n -= 1
        postcode4 = postcode6[:4]
        if n % 1000 == 0:
            print(n)
            print(postcode4)
        if postcode4 in data:
            data[postcode4].append(postcode6)
        else:
            data[postcode4] = [postcode6]


    with open(f'src/data/{output_file}.txt', 'w') as convert_file:
        convert_file.write(json.dumps(data))
        
def find_all_postcode6(postcode_file:str, postcode4:str):
    with open(f"src/data/{postcode_file}", 'r') as f:
        postcodes = json.loads(f.read())
    return postcodes[postcode4]

def find_random_housenumber(postcode_df:pd.DataFrame, postcode6:str):
    df = postcode_df[postcode_df["PC6"] == postcode6]
    housenumbers = df["Huisnummer"].tolist()
    random_housenumber = random.choice(housenumbers)
    
    return f"{random_housenumber}, {postcode6}"
    
if __name__ == '__main__':
    postcode_df = pd.read_csv("src/data/pc6hnr20210801_gwb.csv", sep=";")
    # print(find_maxnumber_postcode6("all_postcode6.txt"))
    print(find_random_housenumber(postcode_df, "2324MC"))
    # find_all_postcode6("all_postcode6.txt", '2324')
