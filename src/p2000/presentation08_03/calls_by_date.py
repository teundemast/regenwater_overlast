import json 
import pandas as pd

# # Step 1
# def load_data(file):
#     with open(file, 'r') as file:
#         data = json.load(file)
#         return data['data']
    
# data = load_data("parsed.json")

# dict = {
#     "date": [],
#     "call": []
# }

# for call in data:
#     dict["date"].append(call["date"])
#     dict["call"].append(1)
    
# df = pd.DataFrame(dict)
# df.to_csv("calls_by_date.csv", index = False)

dict ={
    "date": [],
    "n_calls": []
}

df = pd.read_csv("calls_by_date.csv")
df["date"] = pd.to_datetime(df["date"])

grouped_df = df.groupby(pd.Grouper(key='date', freq='D'))
for day, group in grouped_df:
    dict["date"].append(day)
    dict["n_calls"].append(group["call"].sum())
    
df = pd.DataFrame(data = dict)

df.to_csv("calls_by_date_parsed.csv")