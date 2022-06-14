import pandas as pd 
import numpy as np 
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

df = pd.read_pickle("/local/s2656566/wateroverlast/regenwater_overlast/src/data/pkls/precise.pkl").reset_index()
resultFolder ="/local/s2656566/wateroverlast/regenwater_overlast/results/result_texts/" 
resultFile = open (resultFolder +"result_precise_rain_height.txt", "w+") 
df = df.dropna()

# if only rain 
# dict = {
#     "rain" : [],
#     "target": []
# }
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

accuracyResult = []
precisionResult = []
recallResult = []
totalConfusion = [[0,0],[0,0]]
for i in range(10):
    test_frame = pd.read_csv(f"src/test_frames/frame_{i}.csv", index_col=0)
    dates_test_frame = test_frame["date"].tolist()
    training_frame = rain_p2000[~rain_p2000["date"].isin(dates_test_frame)]
    
    training_frame = training_frame.sample(1700)
    
    training_frame = training_frame.drop(columns=["date"])
    test_frame = test_frame.drop(columns=["date"])
    
    training_labels = np.asarray(training_frame['target'])
    training_features = np.asarray(training_frame.drop(columns=['target']))
    test_labels = np.asarray(test_frame['target'])
    test_features = np.asarray(test_frame.drop(columns=['target']))
    
    feature_list = list(training_frame.drop(columns=['target']).columns)
    print(feature_list)    
    
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
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
    
    resultFile.write("Fold "+str(i)+"\n")
    resultFile.write(str(confusion)+'\n')
    resultFile.write("Accuracy: "+str(metrics.accuracy_score(test_labels, label_prediction))+"\n")
    resultFile.write("Precision: "+str(precision_score(test_labels, label_prediction))+"\n")
    resultFile.write("Recall: "+str(recall_score(test_labels, label_prediction)) + "\n\n")
resultFile.write("\nAverage accuracy: "+str(np.average(accuracyResult))+"\n")
resultFile.write("Average Precision: "+str(np.average(precisionResult))+"\n")
resultFile.write("Average Recall: "+ str(np.average(recallResult))+"\n")
resultFile.write("Total Confusion matrix: \n["+str(totalConfusion[0][0])+","+ str(totalConfusion[0][1])+"] \n"+"["+str(totalConfusion[1][0])+","+ str(totalConfusion[1][1])+"] \n")
resultFile.close()
    
# labels = np.asarray(rain_p2000['target'])
   
# features = rain_p2000.drop(columns=['target'])
    
# # Saving feature names for later use
# feature_list = list(features.columns)
# print(feature_list)    
# features_with_date = np.asarray(features)
# features_used = np.delete(features_with_date, 1, 1)
# print(features_used)
# # print(labels.shape)
# # #k-fold cross validation
# skf = StratifiedKFold(n_splits=10)
# accuracyResult = []
# precisionResult = []
# recallResult = []
# totalConfusion = [[0,0],[0,0]]
# labels = label_encoder.fit_transform(labels)
# n = 0
# for train_index, test_index in skf.split(features_with_date,labels):
#     print("Train: ", train_index, " Test: ", test_index)
#     train_features, test_features = features_with_date[train_index], features_with_date[test_index]
    
#     train_labels, test_labels = labels[train_index], labels[test_index]
#     test_feature
#     n += 1
#     print(len(train_index))
     
#     rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)        
#     rf.fit(train_features, train_labels)
#     label_prediction = rf.predict(test_features)

#     confusion = confusion_matrix(test_labels,label_prediction)
#     totalConfusion[0][0] += confusion[0][0]
#     totalConfusion[0][1] += confusion[0][1]
#     totalConfusion[1][0] += confusion[1][0]
#     totalConfusion[1][1] += confusion[1][1]

#     accuracyResult.append(metrics.accuracy_score(test_labels, label_prediction))
#     precisionResult.append(precision_score(test_labels, label_prediction))
#     recallResult.append(recall_score(test_labels, label_prediction))

#     resultFile.write("Fold "+str(treeNumber)+"\n")
#     resultFile.write(str(confusion)+'\n')
#     resultFile.write("Accuracy: "+str(metrics.accuracy_score(test_labels, label_prediction))+"\n")
#     resultFile.write("Precision: "+str(precision_score(test_labels, label_prediction))+"\n")
#     resultFile.write("Recall: "+str(recall_score(test_labels, label_prediction)) + "\n\n")

# resultFile.write("\nAverage accuracy: "+str(np.average(accuracyResult))+"\n")
# resultFile.write("Average Precision: "+str(np.average(precisionResult))+"\n")
# resultFile.write("Average Recall: "+ str(np.average(recallResult))+"\n")
# resultFile.write("Total Confusion matrix: \n["+str(totalConfusion[0][0])+","+ str(totalConfusion[0][1])+"] \n"+"["+str(totalConfusion[1][0])+","+ str(totalConfusion[1][1])+"] \n")
# resultFile.close()
   
# # if __name__ == '__main__':
# #     randomForest()
        
