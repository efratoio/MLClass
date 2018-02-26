import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

digits = datasets.load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.33, random_state=42)


MSE=[]
neighbors=[]
for k in range(1,80,2):
	knn = KNeighborsClassifier(n_neighbors=k)
	scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
	acc = scores.mean()
	MSE.append(1-acc)
	neighbors.append(k)



plt.plot(neighbors,MSE,'ro')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy Rate')
plt.show()


