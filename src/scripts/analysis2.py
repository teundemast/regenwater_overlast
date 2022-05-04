import pandas as pd 
import numpy as np 
import sys
import os
from numpy import savetxt
import matplotlib.pyplot as plt
#sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
# from autosklearn.experimental.askl2 import AutoSklearn2Classifier
import pandas as pd
import numpy as np

df = pd.read_pickle("/local/s2656566/wateroverlast/regenwater_overlast/src/data/dataset_depression.pkl").reset_index()
is_dslab = os.getenv('DS_LAB', None)


dict = {
    "rain" : [],
    "target": []
}
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
        result = result.flatten()
        dfArr = pd.DataFrame(result)
        dfArr = dfArr.transpose()
        arr = arr.to_frame()
        arr = arr.drop("layers")
        arr = arr.transpose()
        arr = arr.reset_index()
        dfArr = dfArr.reset_index()
        arr = arr.join(dfArr, lsuffix="l")
        print(arr)
        listofarr.append(arr)

df[column] = df.apply(normalize, axis=1)
df[column] = df.apply(reshape, axis=1)


concat_df = pd.concat(listofarr)
print(concat_df) 

df = concat_df.dropna(axis="columns", how="all")
df = df.dropna(thresh=10)
df = df.reset_index(drop=True)
print(df.head())

resultFolder ="/local/s2656566/wateroverlast/regenwater_overlast/src/" 
resultFile = open (resultFolder +"resultdepressie.txt", "w+")
     #load data
rain_p2000 = df 
print("data loaded")



rain_p2000= rain_p2000.drop(columns=['lat', 'lng','index', 'indexl','level_0', 'date'])
    

rain_p2000 = rain_p2000.dropna()
    
labels = np.asarray(rain_p2000['target'])
   
features = rain_p2000.drop(columns=['target'])
    
# Saving feature names for later use
feature_list = list(features.columns)
    
features = np.asarray(features)
print(labels.shape)
#k-fold cross validation
skf = StratifiedKFold(n_splits=10)
treeNumber = 0
accuracyResult = []
precisionResult = []
recallResult = []
totalConfusion = [[0,0],[0,0]]
labels = label_encoder.fit_transform(labels)
for train_index, test_index in skf.split(features,labels):
    print("Train: ", train_index, " Test: ", test_index)
    train_features, test_features = features[train_index], features[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]

    print(test_features[0]) 
#         #train and test the decision tree
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)        
#         rf = AutoSklearn2Classifier(time_left_for_this_task=240*60, per_run_time_limit=7*60)
    rf.fit(train_features, train_labels)
    label_prediction = rf.predict(test_features)

    
    
    
#         #output performance subtree
#         #errors = abs(label_prediction - test_labels)
#         #print('Mean Absolute Error:', round(np.mean(errors), 2))
#         # Pull out one tree from the forest
    confusion = confusion_matrix(test_labels,label_prediction)
    totalConfusion[0][0] += confusion[0][0]
    totalConfusion[0][1] += confusion[0][1]
    totalConfusion[1][0] += confusion[1][0]
    totalConfusion[1][1] += confusion[1][1]

    accuracyResult.append(metrics.accuracy_score(test_labels, label_prediction))
    precisionResult.append(precision_score(test_labels, label_prediction))
    recallResult.append(recall_score(test_labels, label_prediction))

    resultFile.write("Fold "+str(treeNumber)+"\n")
    resultFile.write(str(confusion)+'\n')
    resultFile.write("Accuracy: "+str(metrics.accuracy_score(test_labels, label_prediction))+"\n")
    resultFile.write("Precision: "+str(precision_score(test_labels, label_prediction))+"\n")
    resultFile.write("Recall: "+str(recall_score(test_labels, label_prediction)) + "\n\n")
        
#         # tree = rf.estimators_[4]# Import tools needed for visualization
#         # from sklearn.tree import export_graphviz
#         # import pydot# Pull out one tree from the forest
#         # tree = rf.estimators_[5]# Export the image to a dot file
#         # outputFile = resultFolder+"tree"+str(treeNumber)+".dot"
#         # export_graphviz(tree, out_file = outputFile, feature_names = feature_list, rounded = True, precision = 1)# Use dot file to create a graph
#         # #(graph, ) = pydot.graph_from_dot_file('tree.dot')# Write graph to a png file
#         # #graph.write_png('tree.png')
#         treeNumber+=1
        
#     #output cross validation performance
#     #all_accuracies = cross_val_score(estimator=rf, X=features, y=labels, cv=10)

#     fig, ax = plt.subplots()
#     data = [accuracyResult, precisionResult, recallResult]
#     xlabels = ["Accuracy", "Precision", "Recall"]
#     ax.boxplot(data)
#     ax.set_xticklabels(xlabels)
#     ax.set_ylim(0,1)

resultFile.write("\nAverage accuracy: "+str(np.average(accuracyResult))+"\n")
resultFile.write("Average Precision: "+str(np.average(precisionResult))+"\n")
resultFile.write("Average Recall: "+ str(np.average(recallResult))+"\n")
resultFile.write("Total Confusion matrix: \n["+str(totalConfusion[0][0])+","+ str(totalConfusion[0][1])+"] \n"+"["+str(totalConfusion[1][0])+","+ str(totalConfusion[1][1])+"] \n")
resultFile.close()
#     #print(cross_val_score(estimator=rf, X=features, y=labels, cv=skf, scoring="accuracy"))
#     plt.savefig(resultFolder+"boxplotMeasures.png")
#     plt.show()
   
# if __name__ == '__main__':
#     randomForest()
        
