import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

digits = datasets.load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.33, random_state=42)




knn = KNeighborsClassifier(n_neighbors=13)

knn.fit(X_train, y_train) # fitting the model
pred = knn.predict(X_test) # predict the response
print(pred)
print(accuracy_score(y_test, pred))


