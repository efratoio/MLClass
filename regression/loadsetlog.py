import csv
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


dicts = []

featur1 = "Runtime (Minutes)"
featur2 = "Votes"
label = "Year"


with open("IMDB-Movie-Data.csv") as f:
	reader = csv.DictReader(f)
	for row in reader:
		dicts.append(row)




data = []
target = []

for d in dicts:
	if not label in d or not  featur1 in d or featur2 not in d:
		continue
	target.append(d[label])
	data.append([d[featur1],d[featur2]])


data = np.array(data,dtype='float64')

years_dict = [str(y) for y in range(2006,2017)]

target = np.array([years_dict.index(str(y)) for y in target],dtype='float64')


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

# 3. creating regression model instance (still not trained!)
logreg = linear_model.LogisticRegression(C=1e5)

# 4. training the model 
logreg.fit(X_train,y_train)

y_predictions_train = logreg.predict(X_train)

# 5. predict (reg is now the predictive model)
y_predictions = logreg.predict(X_test)

print("Train mean squared error is %f "%mean_squared_error(y_train, y_predictions_train))

# 6. evaluate are model! 
print("Mean squared error is %f "%mean_squared_error(y_test, y_predictions))

# Done:)


