import numpy as np
import pandas as pd
import time
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class KNN:
    
    def proses(self, k, path):
        # Memuat dataset iris
        
        if path == 'Iris.xlsx' :
        
            print(path)

            df = pd.read_excel(path)
            X = df[['A1', 'A2','A3','A4']].values
            Y = df["Label"].values

              # Membuat objek KNN dengan k=3
            knn = KNeighborsClassifier(n_neighbors=k, metric ='euclidean')

            scores = []
            scores.append(['Uji ke','Akurasi','Precision','Recall','F-Measure','Waktu Komputasi'])

            cv = KFold(n_splits=10)
            index_hasil = 1

            for index_hasil, (train_index, test_index)in enumerate(cv.split(X),1):
                X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
                start_time = time.time()
            
                knn.fit(X_train, Y_train)

                y_pred = knn.predict(X_test)
                Cm = confusion_matrix(Y_test, y_pred)
                print(Cm)
                acc = round(accuracy_score(Y_test,y_pred),2)
                prec = round(precision_score(Y_test,y_pred, zero_division=1, average = 'macro'),2)
                rec = round(recall_score(Y_test,y_pred, zero_division=1, average = 'macro'),2)
                f1 = round(f1_score(Y_test,y_pred, average = 'macro'),2)
                execution_time = round((time.time() - start_time),2)
                scores.append([index_hasil,acc,prec,rec,f1,execution_time])
                index_hasil +=1

            temp = ['Rata-rata', 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(1,7):
                for j in range(2,6):
                    temp[j] += scores[i][j]
            
            for i in range(2,6):
                temp[i] = round((temp[i]/6),2)
                    
            scores.append(temp)
            print(scores)
            return scores

        else:
            print(path)

            df = pd.read_excel(path)
            X = df[['A1', 'A2']].values
            Y = df["Label"].values

                # Membuat objek KNN dengan k=3
            knn = KNeighborsClassifier(n_neighbors=k, metric ='euclidean')

            scores = []
            scores.append(['Uji ke','Akurasi','Precision','Recall','F-Measure','Waktu Komputasi'])

            cv = KFold(n_splits=10)
            index_hasil = 1

            for index_hasil, (train_index, test_index)in enumerate(cv.split(X),1):
                X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
                start_time = time.time()
            
                knn.fit(X_train, Y_train)

                y_pred = knn.predict(X_test)
                Cm = confusion_matrix(Y_test, y_pred)
                print(Cm)
                acc = round(accuracy_score(Y_test,y_pred),2)
                prec = round(precision_score(Y_test,y_pred, zero_division=1, average = 'macro'),2)
                rec = round(recall_score(Y_test,y_pred, zero_division=1, average = 'macro'),2)
                f1 = round(f1_score(Y_test,y_pred, average = 'macro'),2)
                execution_time = round((time.time() - start_time),2)
                scores.append([index_hasil,acc,prec,rec,f1,execution_time])
                index_hasil +=1

            temp = ['Rata-rata', 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(1,7):
                for j in range(2,6):
                    temp[j] += scores[i][j]
            
            for i in range(2,6):
                temp[i] = round((temp[i]/6),2)
                    
            scores.append(temp)
            print(scores)

            
            return scores


# from tabulate import tabulate

#             # ...
#             # Bagian kode sebelumnya

#             # Menampilkan scores dengan tabulate
#             table = tabulate(scores, headers="firstrow", tablefmt="fancy_grid")
#             print(table)