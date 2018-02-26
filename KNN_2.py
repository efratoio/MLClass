
from sklearn import datasets
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split


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



def estimate_accuracy(X_train,y_train,X_test,y_test,k):

	prediction_results = np.array([y_test[i]==KNN_predict(X_train,y_train,X_test[i,:],k) for i in range(X_test.shape[0])])
	
	return prediction_results.sum()/prediction_results.size



iris = datasets.load_iris()


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)

acc = estimate_accuracy(X_train,y_train,X_test,y_test,3)

print("The accuracy is %f for k=3"%acc)
