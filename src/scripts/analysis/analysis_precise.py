import pandas as pd 
import random
import numpy as np 
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn import metrics

resultFolder ="/local/s2656566/wateroverlast/regenwater_overlast/results/result_texts/"
resultFile = open (resultFolder +"result_precise_height_all.txt", "w+") 

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
for path in ["precise2.pkl",  "precise3.pkl", "precise4.pkl"  ,"precise5.pkl" , "precise.pkl"]:
        print("Nu bezig met: " + path)
        df = pd.read_pickle(f"/local/s2656566/wateroverlast/regenwater_overlast/src/data/pkls/{path}").reset_index()
        df = df.dropna()
        df = df[["date", "target", "layers"]]
        listofarr = []
        column = "layers"

        df[column] = df.apply(normalize, axis=1)
        df[column] = df.apply(reshape, axis=1)
        
        concat_df = pd.concat(listofarr)

        df = concat_df.dropna(axis="columns", how="all")
        df = df.reset_index(drop=True)
        rain_p2000= df.drop(columns=['level_0', 'indexl', 'index'])
        # Only rain: 
        # rain_p2000 = df[["past3hours", "date", "target"]]
        directory = os.fsencode("src/test_frames/") 
        all_files = []
        files = os.listdir(directory)
        for file in files:
                all_files.append(os.fsdecode(file))
        ten_random_files = random.sample(all_files, 10)
        for filename in ten_random_files:
                test_frame = pd.read_csv(f"src/test_frames/{filename}", index_col=0)
                # Only height:
                list_399 = [str(x) for x in list(range(400))]
                test_frame = test_frame[list_399 + ['date', 'target']]
                # Only rain: 
                # test_frame = test_frame[["past3hours", "date", "target"]]
                # print(test_frame)
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

                resultFile.write("Fold "+path+"\n")
                resultFile.write(str(confusion)+'\n')
                resultFile.write("Accuracy: "+str(metrics.accuracy_score(test_labels, label_prediction))+"\n")
                resultFile.write("Precision: "+str(precision_score(test_labels, label_prediction))+"\n")
                resultFile.write("Recall: "+str(recall_score(test_labels, label_prediction)) + "\n\n")
resultFile.write("\nAverage accuracy: "+str(np.average(accuracyResult))+"\n")
resultFile.write("Average precision: "+str(np.average(precisionResult))+"\n")
resultFile.write("Average recall: "+ str(np.average(recallResult))+"\n")
resultFile.write("Total Confusion matrix: \n["+str(totalConfusion[0][0])+","+ str(totalConfusion[0][1])+"] \n"+"["+str(totalConfusion[1][0])+","+ str(totalConfusion[1][1])+"] \n")
resultFile.close()

# df = pd.read_pickle("/local/s2656566/wateroverlast/regenwater_overlast/src/data/pkls/precise.pkl").reset_index()
# resultFolder ="/local/s2656566/wateroverlast/regenwater_overlast/results/result_texts/" 
# resultFile = open (resultFolder +"result_precise_rain_height.txt", "w+") 
# df = df.dropna()

# # if only rain 
# # dict = {
# #     "rain" : [],
# #     "target": []
# # }
# listofarr = []
# column = "layers"

# def normalize(row):
#         height = row[column]
#         nans = height > 1000
#         height[nans] = np.nan
#         height = (height-np.nanmean(height))/np.nanstd(height)
#         height[np.isnan(height)] = 3
#         return height

# def reshape(arr):
#         result = np.reshape(arr[column], (20,20))
#         result = result.flatten()
#         dfArr = pd.DataFrame(result)
#         dfArr = dfArr.transpose()
#         arr = arr.to_frame()
#         arr = arr.drop("layers")
#         arr = arr.transpose()
#         arr = arr.reset_index()
#         dfArr = dfArr.reset_index()
#         arr = arr.join(dfArr, lsuffix="l")
#         listofarr.append(arr)

# df[column] = df.apply(normalize, axis=1)
# df[column] = df.apply(reshape, axis=1)

# concat_df = pd.concat(listofarr)

# df = concat_df.dropna(axis="columns", how="all")
# df = df.reset_index(drop=True)
# rain_p2000= df.drop(columns=['level_0', 'indexl', 'index'])

# accuracyResult = []
# precisionResult = []
# recallResult = []
# totalConfusion = [[0,0],[0,0]]
# for i in range(10):
#     test_frame = pd.read_csv(f"src/test_frames/frame_{i}.csv", index_col=0)
#     dates_test_frame = test_frame["date"].tolist()
#     training_frame = rain_p2000[~rain_p2000["date"].isin(dates_test_frame)]
    
#     training_frame = training_frame.sample(1700)
    
#     training_frame = training_frame.drop(columns=["date"])
#     test_frame = test_frame.drop(columns=["date"])
    
#     training_labels = np.asarray(training_frame['target'])
#     training_labels = training_labels.astype('int')
#     training_features = np.asarray(training_frame.drop(columns=['target']))
#     training_features = training_features.astype('float')
#     test_labels = np.asarray(test_frame['target'])
#     test_labels = test_labels.astype('int')
#     test_features = np.asarray(test_frame.drop(columns=['target']))
#     test_features = test_features.astype('float')
    
#     feature_list = list(training_frame.drop(columns=['target']).columns)
#     print(feature_list)    
    
#     rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
#     # print(training_labels)
#     rf.fit(training_features, training_labels)
    
#     label_prediction = rf.predict(test_features)  
#     confusion = confusion_matrix(test_labels,label_prediction)
#     totalConfusion[0][0] += confusion[0][0]
#     totalConfusion[0][1] += confusion[0][1]
#     totalConfusion[1][0] += confusion[1][0]
#     totalConfusion[1][1] += confusion[1][1]
    
#     accuracyResult.append(metrics.accuracy_score(test_labels, label_prediction))
#     precisionResult.append(precision_score(test_labels, label_prediction))
#     recallResult.append(recall_score(test_labels, label_prediction))
    
#     resultFile.write("Fold "+str(i)+"\n")
#     resultFile.write(str(confusion)+'\n')
#     resultFile.write("Accuracy: "+str(metrics.accuracy_score(test_labels, label_prediction))+"\n")
#     resultFile.write("Precision: "+str(precision_score(test_labels, label_prediction))+"\n")
#     resultFile.write("Recall: "+str(recall_score(test_labels, label_prediction)) + "\n\n")
# resultFile.write("\nAverage accuracy: "+str(np.average(accuracyResult))+"\n")
# resultFile.write("Average Precision: "+str(np.average(precisionResult))+"\n")
# resultFile.write("Average Recall: "+ str(np.average(recallResult))+"\n")
# resultFile.write("Total Confusion matrix: \n["+str(totalConfusion[0][0])+","+ str(totalConfusion[0][1])+"] \n"+"["+str(totalConfusion[1][0])+","+ str(totalConfusion[1][1])+"] \n")
# resultFile.close()
    
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





   
# # if __name__ == '__main__':
# #     randomForest()
        
