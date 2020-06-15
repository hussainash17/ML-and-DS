import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
# first : means taking all the columns
# second :-1 means taking all the columns except last column
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# taking care of a missing data
from sklearn.impute import SimpleImputer

# axis = 0 means taking mean of the column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# 1:3 means taking column 1 and 2 which is excluding
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
# X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import  ColumnTransformer

lebelencoder_X = LabelEncoder()
X[:, 0] = lebelencoder_X.fit_transform(X[:, 0])

ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)

# for purchased column
lebelencoder_Y = LabelEncoder()
Y = lebelencoder_Y.fit_transform(Y)


# spliting the dataset into training and testing set
from sklearn.model_selection import  train_test_split
# test_size = 0.5 means 50% are going to training set and 50% in test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(X_test)
