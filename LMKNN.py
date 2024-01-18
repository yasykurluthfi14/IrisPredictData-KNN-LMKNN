import scipy.spatial
import numpy as np
from operator import itemgetter
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



# class LMKNN:
#     def __init__(self, k):
#         self.k = k

#     def fit(self, X, y):
#         self.X_train = X
#         self.y_train = y
        
#     def distance(self, X1, X2):
#       return scipy.spatial.distance.euclidean(X1, X2)
    
#     def predict(self, X_test):
#         final_output = []
#         myclass = list(set(self.y_train))
#         for i in range(len(X_test)):
#             eucDist = []
#             votes = []
#             for j in range(len(X_train)):
#                 dist = scipy.spatial.distance.euclidean(X_train[j] , X_test[i])
#                 eucDist.append([dist, j, self.y_train[j]])
#             eucDist.sort()
            
#             minimum_dist_per_class = []
#             for c in myclass:
#               minimum_class = []
#               for di in range(len(eucDist)):
#                 if(len(minimum_class) != self.k):
#                   if(eucDist[di][2] == c):
#                     minimum_class.append(eucDist[di])
#                 else:
#                   break
#               minimum_dist_per_class.append(minimum_class)
           
#             indexData = []
#             for a in range(len(minimum_dist_per_class)):
#               temp_index = []
#               for j in range(len(minimum_dist_per_class[a])):
#                 temp_index.append(minimum_dist_per_class[a][j][1])
#               indexData.append(temp_index)

#             centroid = []
#             for a in range(len(indexData)):
#               transposeData = X_train[indexData[a]].T
#               tempCentroid = []
#               for j in range(len(transposeData)):
#                 tempCentroid.append(np.mean(transposeData[j]))
#               centroid.append(tempCentroid)
#             centroid = np.array(centroid)
           
#             eucDist_final = []
#             for b in range(len(centroid)):
#               dist = scipy.spatial.distance.euclidean(centroid[b] , X_test[i])
#               eucDist_final.append([dist, myclass[b]])
#             sorted_eucDist_final = sorted(eucDist_final, key=itemgetter(0))
#             final_output.append(sorted_eucDist_final[0][1])
#         return final_output
    
#     def score(self, X_test, y_test):
#         predictions = self.predict(X_test)
#         value = 0
#         for i in range(len(y_test)):
#           if(predictions[i] == y_test[i]):
#             value += 1
#         return value / len(y_test)
    
# train_path = 'iris.xlsx'
data_train = pd.read_excel('iris.xlsx')
    
countClass = data_train['Label'].value_counts().reset_index()
countClass.columns = ['Label', 'count']
print(countClass)

fig = px.pie(
    countClass, 
    values='count', 
    names="Label", 
    title='Class Distribution', 
    width=700, 
    height=500
)

fig.show()

features = data_train.iloc[:,:9].columns.tolist()
plt.figure(figsize=(18, 27))

for i, col in enumerate(features):
    plt.subplot(6, 4, i*2+1)
    plt.subplots_adjust(hspace =.25, wspace=.3)
    
    plt.grid(True)
    plt.title(col)
    sns.kdeplot(data_train.loc[data_train["Label"]=='Iris-setosa', col], label="Iris-setosa", color = "blue", shade=True, cut=0)
    sns.kdeplot(data_train.loc[data_train["Label"]=='Iris-versicolor', col], label="Iris-versicolor",  color = "yellow", shade=True,  cut=0)
    sns.kdeplot(data_train.loc[data_train["Label"]=='Iris-virginica', col], label="Iris-virginica",  color = "red", shade=True,  cut=0)

    plt.subplot(6, 4, i*2+2) 
    sns.boxplot(y = col, data = data_train, x="Label", palette = ["blue", "yellow", "red"])

label_train = data_train.iloc[:,-1].to_numpy()
fitur_train = data_train.iloc[:,:9].to_numpy()

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(fitur_train)
fitur_train_normalize = scaler.transform(fitur_train)
# from sklearn.model_selection import KFold

# kf = KFold(n_splits=10, random_state=1, shuffle=True) 
# kf.get_n_splits(fitur_train_normalize)

# acc_LMKNN = [] 
# for train_index, test_index in kf.split(fitur_train_normalize):
#   lmknn = LMKNN(3)
#   X_train, X_test = fitur_train_normalize[train_index], fitur_train_normalize[test_index]
#   y_train, y_test = label_train[train_index], label_train[test_index]

#   lmknn.fit(X_train, y_train) 
#   result = lmknn.score(X_test, y_test)
#   acc_LMKNN.append(result)

# print('LMKNN : ',acc_LMKNN)
# print('Mean :', np.mean(acc_LMKNN))