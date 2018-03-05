import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

boston = datasets.load_boston()

feature_num = 0

##################################################################
### beginning of your code
##################################################################

# 1. load the data

# 2. split to train/test sets

# 3. creating regression model instance (still not trained!)

# 4. training the model 

# 5. predict (reg is now the predictive model)

# 6. evaluate are model! 

# Done:)


##################################################################
### end of your code
##################################################################
# The code underneath has noting to do with machine learning -
# only for visualization
x_plot = [[data.min()],[data.max()]]
y_plot = reg.predict(x_plot)
plt.scatter(data, target,  color='black', marker='x')
plt.plot(x_plot, y_plot , color='green', linewidth=2)
plt.xlabel(boston.feature_names[feature_num])
plt.ylabel('House Prices')
plt.xticks(())
plt.yticks(())

plt.show()