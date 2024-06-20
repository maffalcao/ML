import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path = 'C:\\Git\\ML\\2-Regression\\1-LinearRegression\\Salary_Data.csv'
dataset = pd.read_csv(csv_path)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the teste e results

y_predict = regressor.predict(X_test)

# visualizing the treainit set result

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'green')
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'green')
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')

plt.show()

# making a single regression
print(regressor.predict([[12]]))

# getting the final linear regression equation with the values of the coefficients

# y^ = y intercept constant + dependant variable x slope coeficient

# Salary = regressor.coef_ × YearsExperience + regressor.intercept_
print(regressor.coef_)
print(regressor.intercept_)