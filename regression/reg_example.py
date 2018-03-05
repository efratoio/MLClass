
# importing numpy
import numpy as np

# importing modules from sklearn
from sklearn import datasets, linear_model

# import the module for splitting the dataset
from sklearn.model_selection import train_test_split

# importing module for evaluate the prediction model
from sklearn.metrics import mean_squared_error

# librarry for plotting graphs and vizualizing data
import matplotlib.pyplot as plt

# 1. example data
data, target = np.array([[0], [1], [2], [3], [4] ]), np.array([1, 2, 3, 4 ,5])

# 2. split to train/test sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

# 3. creating regression model instance (still not trained!)
reg = linear_model.LinearRegression()

# 4. training the model 
reg.fit(X_train,y_train)

print(X_test)
# 5. predict (reg is now the predictive model)
y_predictions = reg.predict(X_test)

# 6. evaluate are model! 
print("Mean squared error is %f "%mean_squared_error(y_test, y_predictions))

# Done:)

##################################################################
# The code underneath has noting to do with machine learning -
# only for visualization
x_plot = [[data.min()],[data.max()]]
y_plot = reg.predict(x_plot)
plt.scatter(data, target,  color='black', marker='x')
plt.plot(x_plot, y_plot , color='green', linewidth=2)

plt.xticks(())
plt.yticks(())

plt.show()