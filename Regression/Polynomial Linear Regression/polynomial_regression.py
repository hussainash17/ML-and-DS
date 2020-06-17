# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# 1:2 says its a matrix not a vector
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Fitting Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree=10)
X_polynomial = polynomial_regressor.fit_transform(X)

linear_regressor1 = LinearRegression()
linear_regressor1.fit(X_polynomial, y)

# visualising the linear regression results
plt.scatter(X, y, color='red')
plt.plot(X, linear_regressor.predict(X), color='blue')
plt.title('Truth or Bluff ( Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# visualising the polynomial regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, linear_regressor1.predict(polynomial_regressor.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff ( Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# if we increment the degree of polynomial
# it will fit much closely and error is much lower