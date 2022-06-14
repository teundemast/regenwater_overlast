import pandas as pd 
import numpy as np 
import sys
import os
from numpy import savetxt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
import pandas as pd
import numpy as np

df = pd.read_pickle("/local/s2656566/wateroverlast/regenwater_overlast/src/data/pkls/postcode4/postcode4_number1.pkl").reset_index()
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