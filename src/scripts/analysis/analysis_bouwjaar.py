import pandas as pd 
import random
import numpy as np 
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn import metrics
from matplotlib import pyplot as plt


resultFolder ="/local/s2656566/wateroverlast/regenwater_overlast/results/result_texts/"
resultFile = open(resultFolder +"result_precise_bouwjaar_wl_bouwjaar.txt", "w+") 

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

listofarr = []
path = "precise_bouwjaar.pkl"
df = pd.read_pickle(f"/local/s2656566/wateroverlast/regenwater_overlast/src/data/pkls/{path}").reset_index()
df = df.dropna()
# df = df[["target", "bouwjaar", "date"]]
df = df[["target", "layers", "past3hours", "bouwjaar"]]
df[column] = df.apply(normalize, axis=1)
df[column] = df.apply(reshape, axis=1)

concat_df = pd.concat(listofarr)

df = concat_df.dropna(axis="columns", how="all")
df = df.reset_index(drop=True)
df= df.drop(columns=['indexl', 'index'])

df["target"] = df["target"].astype(int)
    
labels = np.array(df['target'])

#set features and convert to numpy array
#with height: features= rainTweets_eq.drop(columns=['radarX', 'radarY', 'date', 'text','tiffile', 'height','labels'])
#features= rainTweets_eq.drop(columns=['labels'])
features= df.drop(columns=['target'])
#features = rainTweets_eq[['rain']]

# Saving feature names for later use
feature_list = list(features.columns)

features = np.array(features)

#k-fold cross validation
skf = StratifiedKFold(n_splits=10)
mape = []
treeNumber = 0
accuracyResult = []
precisionResult = []
recallResult = []
totalConfusion = [[0,0],[0,0]]
for train_index, test_index in skf.split(features, labels):
        #print("Train: ", train_index, " Test: ", test_index)
        train_features, test_features = features[train_index], features[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        #print(test_features[0])
        #train and test the decision tree
        rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)        
        rf.fit(train_features, train_labels)
        label_prediction = rf.predict(test_features)
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
        print(str(metrics.accuracy_score(test_labels, label_prediction)))
        # tree = rf.estimators_[4]# Import tools needed for visualization
        # from sklearn.tree import export_graphviz
        # import pydot# Pull out one tree from the forest
        # tree = rf.estimators_[5]# Export the image to a dot file
        # outputFile = resultFolder+"tree"+str(treeNumber)+".dot"
        # export_graphviz(tree, out_file = outputFile, feature_names = feature_list, rounded = True, precision = 1)# Use dot file to create a graph
        # #(graph, ) = pydot.graph_from_dot_file('tree.dot')# Write graph to a png file
        # #graph.write_png('tree.png')
        treeNumber+=1

        #output cross validation performance
        #all_accuracies = cross_val_score(estimator=rf, X=features, y=labels, cv=10)

fig, ax = plt.subplots()
data = [accuracyResult, precisionResult, recallResult]
xlabels = ["Accuracy", "Precision", "Recall"]
ax.boxplot(data)
ax.set_xticklabels(xlabels)
ax.set_ylim(0,1)

resultFile.write("\nAverage Accuracy: "+str(np.average(accuracyResult))+"\n")
resultFile.write("Average Precision: "+str(np.average(precisionResult))+"\n")
resultFile.write("Average Recall: "+ str(np.average(recallResult))+"\n")
resultFile.write("Total Confusion matrix: \n["+str(totalConfusion[0][0])+","+ str(totalConfusion[0][1])+"] \n"+"["+str(totalConfusion[1][0])+","+ str(totalConfusion[1][1])+"] \n")
#print(cross_val_score(estimator=rf, X=features, y=labels, cv=skf, scoring="accuracy"))
# plt.savefig(resultFolder+"boxplotMeasures.png")
