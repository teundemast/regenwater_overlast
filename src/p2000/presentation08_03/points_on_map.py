import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

months_not_to_analyze = [(2021, 3)]

map_nl = plt.imread("p2000/map_nederland.png")

# Helper functions

def load_data(file):
    with open(file, 'r') as file:
        data = json.load(file)
        return data['data']

# Transforming parsed.json in a useful dataframe

data = load_data("parsed.json")

dict_df = {
    "Date": [],
    "longitude": [],
    "latitude": []
}

for call in data:
    dict_df["Date"].append(call["date"])
    dict_df["longitude"].append((call["longitude"]))
    dict_df["latitude"].append((call["latitude"]))

df = pd.DataFrame(dict_df)

# Defining boundaries of map

df = df[df["longitude"] != 0.0] # Remove zero values from dataframe
df = df[df["latitude"] != 0.0]

BBox = (df.longitude.min(),   df.longitude.max(),
        df.latitude.min(), df.latitude.max())

# Grouping data by month to get a plot for each month

df['Date'] = pd.to_datetime(df.Date)

grouped_df = df.groupby(pd.Grouper(key='Date', freq='Y'))

for year_datetime, year_group in grouped_df:
    grouped_year = year_group.groupby(pd.Grouper(key='Date', freq='M'))
    for month_datetime, month_group in grouped_year:
        year = month_group.iloc[0]["Date"].year
        month = month_group.iloc[0]["Date"].month
        year_month = tuple((year, month))
        if tuple((year, month)) not in months_not_to_analyze:
            fig, ax = plt.subplots(figsize = (10,20))
            ax.scatter(month_group.longitude, month_group.latitude, zorder=1, alpha= 0.2, c='b', s=10)
            ax.set_title(f'Plotting p2000 calls on map for {year}-{month}')
            ax.set_xlim(BBox[0],BBox[1])
            ax.set_ylim(BBox[2],BBox[3])
            ax.imshow(map_nl, zorder=0, extent = BBox, aspect= 'equal')
            plt.savefig(f"{year}_{month}", bbox_inches="tight")
