import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


# 1. load the data
boston = datasets.load_boston()
data, target = boston.data, boston.target

# 1.1 Filtering the data
mask = target != 50.0
target = target[mask]
data = data[mask]
data = data[:,[5,10,12]]

# 2. split to train/test sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)


test_error = []
train_error = []
degrees = []
for degree in range(2,5):
	# 3. creating regression model instance (still not trained!)
	reg = make_pipeline(PolynomialFeatures(degree), linear_model.LinearRegression())

	# 4. training the model 
	reg.fit(X_train,y_train)


	# 5. predict (reg is now the predictive model)
	y_predictions_train = reg.predict(X_train)
	test_error.append(-cross_val_score(reg, X_train, y_train, cv=10, scoring='neg_mean_squared_error').mean())

	# 6. evaluate are model! 

	degrees.append(degree)
	train_error.append(mean_squared_error(y_train, y_predictions_train))

# Done:)

##################################################################
# The code underneath has noting to do with machine learning -
# only for visualization
plt.plot(degrees, train_error, 'g-' ,degrees,test_error,'r--')

h_a = np.array(test_error)
print("Min error is %f for %d degree"%(h_a.min(),degrees[h_a.argmin()]))

plt.xlabel("Poly Degrees")
plt.ylabel('MSE')

plt.show()