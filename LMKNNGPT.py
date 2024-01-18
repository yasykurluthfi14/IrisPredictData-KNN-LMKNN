import numpy as np
import pandas as pd
from collections import Counter

class LMKNN:
    def __init__(self, k, threshold):
        self.k = k
        self.threshold = threshold

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for x in X:
            labels = self._get_neighbors_labels(x)
            if len(labels) > 0:
                most_common = Counter(labels).most_common(1)
                y_pred.append(most_common[0][0])
            else:
                y_pred.append(None)
        return np.array(y_pred)

    def _get_neighbors_labels(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        unique_labels = np.unique(k_nearest_labels)
        labels = []
        for label in unique_labels:
            label_count = k_nearest_labels.count(label)
            if label_count >= self.threshold:
                labels.append(label)
        return labels

# Membuat dataset contoh
data = pd.read_excel('iris.xlsx')

X_train = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
y_train = np.array([0, 0, 1, 1, 0, 1])

# Membuat objek LMKNN
lmknn = LMKNN(k=3, threshold=2)

# Melatih model
lmknn.fit(X_train, y_train)

# Data yang ingin diprediksi
X_test = np.array([[3, 4], [5, 6]])

# Melakukan prediksi
predictions = lmknn.predict(X_test)

# Menampilkan hasil prediksi
for i, pred in enumerate(predictions):
    if pred is not None:
        print(f"Data {X_test[i]} diklasifikasikan sebagai kelas {pred}")
    else:
        print(f"Data {X_test[i]} tidak memiliki kelas yang sesuai")
