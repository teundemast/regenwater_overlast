import pandas as pd 
import random
import numpy as np 
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn import metrics

resultFolder ="/local/s2656566/wateroverlast/regenwater_overlast/results/result_texts/"
resultFile = open (resultFolder +"result_precise_bouwjaar.txt", "w+") 

column = 'layers'
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
        
listofarr = []
path = "precise_bouwjaar.pkl"
df = pd.read_pickle(f"/local/s2656566/wateroverlast/regenwater_overlast/src/data/pkls/{path}").reset_index()
df = df.dropna()
df = df[["target", "layers", "bouwjaar", "past3hours", "date"]]
df[column] = df.apply(normalize, axis=1)
df[column] = df.apply(reshape, axis=1)

concat_df = pd.concat(listofarr)

df = concat_df.dropna(axis="columns", how="all")
df = df.reset_index(drop=True)
rain_p2000= df.drop(columns=['indexl', 'index'])

a_tenth = len(rain_p2000.index) / 10
accuracyResult = []
precisionResult = []
recallResult = []
totalConfusion = [[0,0],[0,0]]
for i in range(10):
    test_frame = rain_p2000.sample(a_tenth)
    dates_test_frame = test_frame["date"].tolist()
    print(len(rain_p2000.index))
    training_frame = rain_p2000[~rain_p2000["date"].isin(dates_test_frame)]
    print(len(training_frame.index))
    training_frame = training_frame.sample(1700)

    training_frame = training_frame.drop(columns=["date"])
    test_frame = test_frame.drop(columns=["date"])

    training_labels = np.asarray(training_frame['target'])
    training_labels = training_labels.astype('int')
    training_features = np.asarray(training_frame.drop(columns=['target']))
    training_features = training_features.astype('float')
    test_labels = np.asarray(test_frame['target'])
    test_labels = test_labels.astype('int')
    test_features = np.asarray(test_frame.drop(columns=['target']))
    test_features = test_features.astype('float')

    feature_list = list(training_frame.drop(columns=['target']).columns)
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
    # print(training_labels)
    rf.fit(training_features, training_labels)

    label_prediction = rf.predict(test_features)  
    confusion = confusion_matrix(test_labels,label_prediction)
    totalConfusion[0][0] += confusion[0][0]
    totalConfusion[0][1] += confusion[0][1]
    totalConfusion[1][0] += confusion[1][0]
    totalConfusion[1][1] += confusion[1][1]

    accuracyResult.append(metrics.accuracy_score(test_labels, label_prediction))
    precisionResult.append(precision_score(test_labels, label_prediction))
    recallResult.append(recall_score(test_labels, label_prediction))

    resultFile.write("Fold "+i+"\n")
    resultFile.write(str(confusion)+'\n')
    resultFile.write("Accuracy: "+str(metrics.accuracy_score(test_labels, label_prediction))+"\n")
    resultFile.write("Precision: "+str(precision_score(test_labels, label_prediction))+"\n")
    resultFile.write("Recall: "+str(recall_score(test_labels, label_prediction)) + "\n\n")
resultFile.write("\nAverage accuracy: "+str(np.average(accuracyResult))+"\n")
resultFile.write("Average precision: "+str(np.average(precisionResult))+"\n")
resultFile.write("Average recall: "+ str(np.average(recallResult))+"\n")
resultFile.write("Total Confusion matrix: \n["+str(totalConfusion[0][0])+","+ str(totalConfusion[0][1])+"] \n"+"["+str(totalConfusion[1][0])+","+ str(totalConfusion[1][1])+"] \n")
resultFile.close()