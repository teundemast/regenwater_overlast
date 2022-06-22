import pandas as pd 
import random
import numpy as np 
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn import metrics
from sklearn.tree import export_graphviz
from subprocess import call

resultFolder ="/local/s2656566/wateroverlast/regenwater_overlast/results/result_texts/"
resultFile = open (resultFolder +"result_precise_bouwjaar_enkel.txt", "w+") 

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
        


accuracyResult = []
precisionResult = []
recallResult = []
totalConfusion = [[0,0],[0,0]]
for i in range(10):
    listofarr = []
    path = "precise_bouwjaar.pkl"
    df = pd.read_pickle(f"/local/s2656566/wateroverlast/regenwater_overlast/src/data/pkls/{path}").reset_index()
    df = df.dropna()
    df = df[["target", "bouwjaar", "date"]]
    # df = df[["target", "layers", "bouwjaar", "past3hours", "date"]]
    # df[column] = df.apply(normalize, axis=1)
    # df[column] = df.apply(reshape, axis=1)

    # concat_df = pd.concat(listofarr)

    # df = concat_df.dropna(axis="columns", how="all")
    # df = df.reset_index(drop=True)
    # df= df.drop(columns=['indexl', 'index'])

    a_tenth = int(len(df.index) / 10)
    test_frame = df.sample(a_tenth)
    # print(test_frame)
    dates_test_frame = test_frame["date"].tolist()
    # print(len(df.index))
    training_frame = df[~df["date"].isin(dates_test_frame)]
    # print(len(training_frame.index))
    training_frame = training_frame.sample(1700)
    # print(training_frame)
    
    training_frame = training_frame.drop(columns=["date"])
    test_frame = test_frame.drop(columns=["date"])

    training_labels = np.asarray(training_frame['target'])
    training_labels = training_labels.astype('int')
    list_training_features = list(training_frame.drop(columns=['target']).columns)
    training_features = np.asarray(training_frame.drop(columns=['target']))
    training_features = training_features.astype('float')
    test_labels = np.asarray(test_frame['target'])
    test_labels = test_labels.astype('int')
    test_features = np.asarray(test_frame.drop(columns=['target']))
    test_features = test_features.astype('float')

    rf = RandomForestClassifier(n_estimators = 1000, random_state = i)
    # print(training_labels)
    print(training_features)
    rf.fit(training_features, training_labels)
    estimator = rf.estimators_[5]
    export_graphviz(estimator, out_file='tree.dot', 
                feature_names = list_training_features,
                class_names = ['0', '1'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    print(test_features)
    label_prediction = rf.predict(test_features)  
    confusion = confusion_matrix(test_labels,label_prediction)
    totalConfusion[0][0] += confusion[0][0]
    totalConfusion[0][1] += confusion[0][1]
    totalConfusion[1][0] += confusion[1][0]
    totalConfusion[1][1] += confusion[1][1]

    accuracyResult.append(metrics.accuracy_score(test_labels, label_prediction))
    precisionResult.append(precision_score(test_labels, label_prediction))
    recallResult.append(recall_score(test_labels, label_prediction))

    resultFile.write("Fold "+str(i)+"\n")
    resultFile.write(str(confusion)+'\n')
    resultFile.write("Accuracy: "+str(metrics.accuracy_score(test_labels, label_prediction))+"\n")
    resultFile.write("Precision: "+str(precision_score(test_labels, label_prediction))+"\n")
    resultFile.write("Recall: "+str(recall_score(test_labels, label_prediction)) + "\n\n")
resultFile.write("\nAverage accuracy: "+str(np.average(accuracyResult))+"\n")
resultFile.write("Average precision: "+str(np.average(precisionResult))+"\n")
resultFile.write("Average recall: "+ str(np.average(recallResult))+"\n")
resultFile.write("Total Confusion matrix: \n["+str(totalConfusion[0][0])+","+ str(totalConfusion[0][1])+"] \n"+"["+str(totalConfusion[1][0])+","+ str(totalConfusion[1][1])+"] \n")
resultFile.close()