import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  confusion_matrix, precision_score, recall_score
from sklearn import metrics
import pandas as pd
import numpy as np
df = pd.read_pickle("/local/s2656566/wateroverlast/regenwater_overlast/src/data/pkls/postcode6/postcode6_number1.pkl").reset_index()
resultFolder ="/local/s2656566/wateroverlast/regenwater_overlast/results/result_texts/" 
resultFile = open (resultFolder +"result_postcode6_rain_height.txt", "w+") 
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
    training_labels = training_labels.astype('int')
    training_features = np.asarray(training_frame.drop(columns=['target']))
    training_features = training_features.astype('float')
    test_labels = np.asarray(test_frame['target'])
    test_labels = test_labels.astype('int')
    test_features = np.asarray(test_frame.drop(columns=['target']))
    test_features = test_features.astype('float')
    
    feature_list = list(training_frame.drop(columns=['target']).columns)
    print(feature_list)    
    
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
    