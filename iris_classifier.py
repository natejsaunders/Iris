import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn import neighbors
from sklearn.model_selection import train_test_split

def load_iris():
    return pd.read_csv("iris.csv")

iris = load_iris()

def accuracy_test(x, y):
    tot = 0
    for i, p in enumerate(x):
        if p == y.iloc[i]: tot+=1

    return tot/len(x)

def test_k(data, k):
    train, test = train_test_split(data, test_size=0.2)
    knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(train.drop('variety' , axis=1), train['variety'])

    predictions = knn_classifier.predict(test.drop('variety', axis=1))
    return accuracy_test(predictions, train['variety'])

ks = []
for k in range(1, 50):
    ks.append(test_k(iris, k))

plt.plot(ks)
#sns.relplot(data=iris, x='sepal_length', y='sepal_width', hue='variety')

plt.show()