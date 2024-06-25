import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path = 'C:\\Git\\ML\\2-Regression\\3-PolinomialRegression\\Position_Salaries.csv'
dataset = pd.read_csv(csv_path)

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures

polynomial_regressor = PolynomialFeatures(degree = 4)
X_poly = polynomial_regressor.fit_transform(X)

linear_regressor2 = LinearRegression()
linear_regressor2.fit(X_poly, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.plot(X, linear_regressor2.predict(X_poly), color = 'green')
plt.title('Truth of Blue (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
#plt.show()


print(linear_regressor.predict([[6.5]]))
print(linear_regressor2.predict(polynomial_regressor.fit_transform([[6.5]])))






