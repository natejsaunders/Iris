import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn import neighbors
from sklearn.model_selection import train_test_split

import random

def load_iris():
    return pd.read_csv("iris.csv")

iris = load_iris()

train, test = train_test_split(iris, test_size=0.2)

knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(train.drop('variety' , axis=1), train['variety'])

predictions = knn_classifier.predict(test.drop('variety', axis=1))

for i, prediction in enumerate(predictions):
    print(f"{prediction}:{train.iloc[i]['variety']}")

sns.relplot(data=iris, x='sepal_length', y='sepal_width', hue='variety')
plt.show()