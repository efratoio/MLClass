
from sklearn import datasets
import numpy as np
from collections import Counter


def KNN_predict(X_train, y_train, X_test, k):
	# create list for distances and targets
	distances = []
	targets = []

	for i in range(len(X_train)):
		# first we compute the euclidean distance
		distance = np.sqrt(np.sum(np.square(X_test - X_train[i, :])))
		# add it to list of distances
		distances.append([distance, i])

	# sort the list
	distances = sorted(distances)

	# make a list of the k neighbors' targets
	for i in range(k):
		index = distances[i][1]
		targets.append(y_train[index])

	# return most common target
	return Counter(targets).most_common(1)[0][0]


iris = datasets.load_iris()

print(iris.target[0]==KNN_predict(iris.data,iris.target,iris.data[0],1))
