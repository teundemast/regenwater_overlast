import pandas as pd
import numpy as np 
import json
from datetime import datetime, timedelta

strfformat = "%Y-%m-%d %H:%M:%S"

def get_data(data, a):
    d = data
    # print(d)
    if len(d) > 0 and d[0]['geometry']['location_type'] != 'APPROXIMATE':
        return data[0]['geometry']['location'][a]
    else:
        print("fakka")
        print(d)
        return np.nan
data = []
df1 = pd.read_json('src/data/merged_precise_coord.json')
df1 = df1.dropna()
df1['lat'] = df1['google_results'].apply(get_data, args=('lat',))
df1['lng'] = df1['google_results'].apply(get_data, args=('lng',))
df1 = df1[['lat','lng','date']]
for index, row in df1.iterrows():
    object = {"date": row['date'].strftime(strfformat), "lat": row["lat"], "lng": row["lng"]}
    data.append(object)
print(data)
with open('src/data/merged_precise_coord_transformed.json', 'w') as f:
    json.dump(data,f,indent=6)