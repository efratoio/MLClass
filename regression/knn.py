import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor

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
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

# print(X_train.shape)
test_error = []
train_error = []
alphas = []

for k in np.arange(4,13):
	# 3. creating regression model instance (still not trained!)
	reg = KNeighborsRegressor(n_neighbors=k)
	# 4. training the model 
	reg.fit(X_train,y_train)
	# print(reg._final_estimator.coef_)

	# 5. predict (reg is now the predictive model)
	y_predictions_train = reg.predict(X_train)
	y_predictions = reg.predict(X_test)

	# 6. evaluate are model! 
	alphas.append(k)
	train_error.append(mean_squared_error(y_train, y_predictions_train))
	test_error.append(mean_squared_error(y_test, y_predictions))

# Done:)
# print(test_error)
# print(train_error)

##################################################################
# The code underneath has noting to do with machine learning -
# only for visualization
plt.plot(alphas, train_error, 'g-' ,alphas,test_error,'r--')

plt.xlabel("Poly Degrees")
plt.ylabel('MSE')

h_a = np.array(test_error)

# plt.ylim(9,11)
print("Min error is %f for %d neughbours"%(h_a.min(),alphas[h_a.argmin()]))
plt.show()