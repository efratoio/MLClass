import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

# 1. load the data
boston = datasets.load_boston()
data, target = boston.data, boston.target

# 1.1 Filtering the data
mask = target != 50.0
target = target[mask]
data = data[mask]
data =  data[:,[5,10,12]]
# 
# poly = PolynomialFeatures(degree=3	)
# data = poly.fit_transform(data)


# 2. split to train/test sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=53)
plt.figure(1)

total_scores = []
for degree in range(2,7):
	# print(X_train.shape)
	test_error = []
	train_error = []
	alphas = []
	for alpha in np.arange(0.01,2.,0.1):
		# 3. creating regression model instance (still not trained!)
		reg = make_pipeline(PolynomialFeatures(degree), linear_model.Ridge(alpha=alpha))

		# 4. training the model 
		reg.fit(X_train,y_train)
		# print(reg._final_estimator.coef_)

		# 5. predict (reg is now the predictive model)
		y_predictions_train = reg.predict(X_train)
		# y_predictions = reg.predict(X_test)

		# 6. evaluate are model! 
		alphas.append(alpha)
		train_error.append(mean_squared_error(y_train, y_predictions_train))
		test_error.append(-cross_val_score(reg, X_train, y_train, cv=10, scoring='neg_mean_squared_error').mean())
		# test_error.append(mean_squared_error(y_test, y_predictions))
	total_scores.append(test_error)
# Done:)
# print(test_error)
# print(train_error)

##################################################################
# The code underneath has noting to do with machine learning -
# only for visualization
	plt.subplot(730+degree-1)
	plt.plot(alphas, train_error, 'g-' ,alphas,test_error,'r--')

	plt.xlabel("Poly Degrees")
	plt.ylabel('MSE')
	plt.title("degree %d"%degree)
h_a = np.array(total_scores)

for row in total_scores:
	print
print(total_scores)
# plt.ylim(9,11)
# print("Min error is %f for %f alpha and degree %d"%(h_a.min(),alphas[h_a.argmin(axis=1)],range(2,7)[h_a.argmin(axis=0)]))
plt.show()
