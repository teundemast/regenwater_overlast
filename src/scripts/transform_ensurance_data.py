from change_resolution import get_random_house
import pandas as pd 
import json


# File Names
ensurance_data = "src/data/postcodes.csv"
postcodes_data = "src/data/pc6hnr20210801_gwb.csv"

# Create dataframes
postcodes_df = pd.read_csv(postcodes_data, sep=";")
df_ensurance = pd.read_csv(ensurance_data)

def import_wet_days(path_neerslag):
    wet_days = []
    for year in range(2017,2022):
        file = path_neerslag + f"wet_days{year}.txt"
        with open(file, "r") as f:
            days = f.read().splitlines()
        wet_days.extend(days)
    transformed_wet_days = []
    for day in wet_days:
        daylist = day.split("-")
        daylist[0], daylist[1], daylist[2] = daylist[2], daylist[1], daylist[0]
        transformed_wet_days.append("/".join(daylist))
    return transformed_wet_days
def transform(row):

    postcode4 = str(row["Postcode4"])

    new_object = {}
    new_object['date'] = row["date"]

    # Progress tracking:
    global n
    n -= 1
    if n % 5 == 0:
        print(n)

    output = get_random_house(postcode4, "postcode4", postcodes_df, ensurance=True)
    if output is None:
        print(postcode4)
        return None

    new_object["lat"] = output[0]
    new_object["lng"] = output[1]
    transformed_data.append(new_object)


output_file = f"src/data/ensurance/ensurance_transformed6.json"
#Sample data
df_ensurance = df_ensurance.sample(17000)
print(df_ensurance)
# Initialize some variables
transformed_data = []
missing = []
n = len(df_ensurance.index)
df_ensurance.apply(lambda row: transform(row), axis=1)
# Output transformed data
with open(output_file, "w") as f:
    json.dump(transformed_data, f, indent=6)





    

