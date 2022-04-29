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
# from autosklearn.experimental.askl2 import AutoSklearn2Classifier
import pandas as pd
import numpy as np

df = pd.read_pickle("/local/s2656566/wateroverlast/regenwater_overlast/dataset.pkl").reset_index()
df = df.head()

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
        result = np.reshape(arr[column], (400,400,1))
        result = result[:, :, 0]
        result = arr[column].flatten()
        dfArr = pd.DataFrame(arr)
        dfArr = dfArr.transpose()
        listofarr.append(dfArr)

df[column] = df.apply(normalize, axis=1)
df[column] = df.apply(reshape, axis=1)

dfOfArr = listofarr[0]
for i in range(1,len(listofarr)):
    dfOfArr = dfOfArr.append(listofarr[i], ignore_index=True)
    
df = df.drop(column)
df = df.join(dfOfArr)

df = df.dropna(axis="columns", how="all")
df = df.dropna(thresh=10)
df = df.reset_index(drop=True)
print(df.head())

# for index, row in df.iterrows():
#         if slak:
#                 arr = row[column].flatten()
#                 dfArr = pd.DataFrame(arr)
#                 dfArr = dfArr.transpose()
#                 print(dfArr.head())
#                 slak = False

# def randomForest(folder='/local/s2656566/regen_project/', inputFile='50-500.pkl', resultFolder = '/home/s2656566/'):
#     resultFile = open (resultFolder+"resultautosklearn.txt", "w+")
    
#     #load data
#     rain_p2000 = pd.read_pickle(folder + inputFile)
#     print("data loaded")

#     print(rain_p2000.dtypes)

#     rain_p2000= rain_p2000.drop(columns=['lat', 'lng', 'prec12', 'prec_sums', 'date'])
    
#     rain_p2000 = np.asarray(df[column].values.tolist())

#     rain_p2000 = rain_p2000.dropna()
    
#     labels = np.array(rain_p2000['target'])
    
#     features = rain_p2000.drop(columns=['target'])
    
#     # Saving feature names for later use
#     feature_list = list(features.columns)
    
#     features = np.array(features)
    
#     #k-fold cross validation
#     skf = StratifiedKFold(n_splits=10)
#     treeNumber = 0
#     accuracyResult = []
#     precisionResult = []
#     recallResult = []
#     totalConfusion = [[0,0],[0,0]]
#     for train_index, test_index in skf.split(features, labels):
#         print("Train: ", train_index, " Test: ", test_index)
#         train_features, test_features = features[train_index], features[test_index]
#         train_labels, test_labels = labels[train_index], labels[test_index]

#         #print(test_features[0]) 
#         #train and test the decision tree
#         # rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)        
#         rf = AutoSklearn2Classifier(time_left_for_this_task=240*60, per_run_time_limit=7*60)
#         rf.fit(train_features, train_labels)
#         label_prediction = rf.predict(test_features)

#         autosklResults = pd.DataFrame(rf.cv_results_)
#         autosklResults.to_csv(resultFolder+ "autosklearn"+str(treeNumber)+".csv")
#         print(autosklResults)
#         #output performance subtree
#         #errors = abs(label_prediction - test_labels)
#         #print('Mean Absolute Error:', round(np.mean(errors), 2))
#         # Pull out one tree from the forest
#         confusion = confusion_matrix(test_labels,label_prediction)
#         totalConfusion[0][0] += confusion[0][0]
#         totalConfusion[0][1] += confusion[0][1]
#         totalConfusion[1][0] += confusion[1][0]
#         totalConfusion[1][1] += confusion[1][1]

#         accuracyResult.append(metrics.accuracy_score(test_labels, label_prediction))
#         precisionResult.append(precision_score(test_labels, label_prediction))
#         recallResult.append(recall_score(test_labels, label_prediction))

#         resultFile.write("Fold "+str(treeNumber)+"\n")
#         resultFile.write(str(confusion)+'\n')
#         resultFile.write("Accuracy: "+str(metrics.accuracy_score(test_labels, label_prediction))+"\n")
#         resultFile.write("Precision: "+str(precision_score(test_labels, label_prediction))+"\n")
#         resultFile.write("Recall: "+str(recall_score(test_labels, label_prediction)) + "\n\n")
        
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

#     resultFile.write("\nAverage accuracy: "+str(np.average(accuracyResult))+"\n")
#     resultFile.write("Average Precision: "+str(np.average(precisionResult))+"\n")
#     resultFile.write("Average Recall: "+ str(np.average(recallResult))+"\n")
#     resultFile.write("Total Confusion matrix: \n["+str(totalConfusion[0][0])+","+ str(totalConfusion[0][1])+"] \n"+"["+str(totalConfusion[1][0])+","+ str(totalConfusion[1][1])+"] \n")
#     resultFile.close()
#     #print(cross_val_score(estimator=rf, X=features, y=labels, cv=skf, scoring="accuracy"))
#     plt.savefig(resultFolder+"boxplotMeasures.png")
#     plt.show()
   
# if __name__ == '__main__':
#     randomForest()
        