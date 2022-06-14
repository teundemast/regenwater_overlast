from change_resolution import get_random_house
import pandas as pd 
import json


# File Names
ensurance_data = "src/data/postcodes.csv"
postcodes_data = "src/data/pc6hnr20210801_gwb.csv"

# Create dataframes
postcodes_df = pd.read_csv(postcodes_data, sep=";")
df_ensurance = pd.read_csv(ensurance_data)

def transform(row):
    
    postcode4 = str(row["Postcode4"])

    number_notifications = row["Aantal meldingen"]    
    
    for i in range(number_notifications):
        new_object = {}
        new_object['date'] = row["Datum"]
        
        # Progress tracking: 
        global n 
        n -= 1
        if n % 5 == 0:
            print(n)
            
        output = get_random_house(postcode4, "postcode4", postcodes_df, ensurance=True)
        if output is None:
            print(postcode4)
            missing.append(row["Datum"])
            continue

        new_object["lat"] = output[0]
        new_object["lng"] = output[1]
        transformed_data.append(new_object)
numbers_missing = []
for i in range(5):
    output_file = f"src/data/ensurance_transformed{i}.csv"
    #Sample data
    df_ensurance = df_ensurance.sample(7000)
    
    # Initialize some variables 
    transformed_data = []
    missing = []
    n = len(df_ensurance.index)
    df_ensurance.apply(lambda row: transform(row), axis=1)
    # Output transformed data
    with open(output_file, "w") as f:
        json.dump(transformed_data, output_file, indent=6)
    numbers_missing.append(len(missing))




    

