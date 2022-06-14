import pandas as pd 
import numpy as np 
import os

df = pd.read_pickle("/local/s2656566/wateroverlast/regenwater_overlast/src/data/pkls/precise.pkl").reset_index()
is_dslab = os.getenv('DS_LAB', None)
df = df.dropna()

listofarr = []
column = "layers"

def normalize(row):
        height = row[column]
        nans = height > 1000
        height[nans] = np.nan
        height = (height-np.nanmean(height))/np.nanstd(height)
        height[np.isnan(height)] = 3
        return height

def reshape(arr):
        result = np.reshape(arr[column], (20,20))
        result = result.flatten()
        dfArr = pd.DataFrame(result)
        dfArr = dfArr.transpose()
        arr = arr.to_frame()
        arr = arr.drop("layers")
        arr = arr.transpose()
        arr = arr.reset_index()
        dfArr = dfArr.reset_index()
        arr = arr.join(dfArr, lsuffix="l")
        listofarr.append(arr)

df[column] = df.apply(normalize, axis=1)
df[column] = df.apply(reshape, axis=1)

concat_df = pd.concat(listofarr)

df = concat_df.dropna(axis="columns", how="all")
df = df.reset_index(drop=True)
rain_p2000= df.drop(columns=['level_0', 'indexl', 'index'])
number_rows = len(rain_p2000.index)
print(number_rows)
number_per_frame = int(number_rows/10)

for i in range(10):
    path = f"src/test_frames/frame_{i}.csv"
    if i == 9:
        test_frame = rain_p2000
    else:
        test_frame = rain_p2000.sample(number_per_frame)
    test_frame.to_csv(path)
    rain_p2000 = rain_p2000.drop(test_frame.index)
    