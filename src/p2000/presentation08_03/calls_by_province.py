import pandas as pd
import json 


# # Step 1
# def load_data(file):
#     with open(file, 'r') as file:
#         data = json.load(file)
#         return data['data']
    
# data = load_data("parsed.json")

# dict = {
#     "province": [],
#     "call": []
# }

# for call in data:
#     dict["province"].append(call["province"])
#     dict["call"].append(1)
    
# df = pd.DataFrame(dict)
# df.to_csv("calls_by_province.csv", index = False)

dict ={
    "province": [],
    "n_calls": []
}

df = pd.read_csv("calls_by_province.csv")

grouped_df = df.groupby(["province"])
for province, group in grouped_df:
    dict["province"].append(province)
    dict["n_calls"].append(group["call"].sum())
    
df = pd.DataFrame(data = dict)

df.to_csv("calls_by_province_parsed.csv", index = False)