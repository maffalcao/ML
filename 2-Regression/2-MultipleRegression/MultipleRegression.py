import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path = 'C:\\Git\\ML\\2-Regression\\2-MultipleRegression\\50_Startups.csv'
dataset = pd.read_csv(csv_path)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# in multiple regression there's no need to apply feature scaling

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_predict = regressor.predict(X_test)

np.set_printoptions(precision = 2)

print(np.concatenate((y_predict.reshape(len(y_predict),1), y_test.reshape(len(y_test),1)),1))

#making a single prediction
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

#Getting the final linear regression equation with the values of the coefficients

# Profit=86.6×Dummy State 1−873×Dummy State 2+786×Dummy State 3+0.773×R&D Spend+0.0329×Administration+0.0366×Marketing Spend+42467.53

print(regressor.coef_)
print(regressor.intercept_)