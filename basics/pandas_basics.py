import numpy as np
import pandas as pd
from numpy.random import randn

labels = ['a', 'b', 'c']
my_data = [10, 20, 30]
numpy_arr = np.array(my_data)
d = {'a': 10, 'b': 20, 'c': 30}

# series can hold any data types
# series are so fast for retrieving something
#  like hash tables
# creates a series based on my_data
series = pd.Series(my_data)

# creates a series of key value by labels and my_data
series = pd.Series(my_data, labels)

# seed is used for make same random number always
np.random.seed(101)
# create a dataframe of 5x4
# basically its a series
df = pd.DataFrame(randn(5, 4), ['A', 'B', 'C', 'D', 'E'], ['W', 'X', 'Y', 'Z'])

# selecting columns
col = df['W']

# add a new column
df['new'] = df['W'] + df['X']

# delete a column
# if inplace is true then it will deleted from mail df
# if false then main df still same
df.drop('new', axis=1, inplace=True)

# delete a row
# axis comes from the shape of the matrix
# row is the first index of the shape and column is the
# second index of the shape like (5,4) where 5 is in index 0
df.drop('E', axis=0, inplace=False)

# selecting rows
row = df.loc['A']
row = df.iloc[2]

# select a specific row value
row = df.loc['B', 'Z']

# selecting both row and column
row = df.loc[['A', 'B'], ['X', 'Z']]

# conditional selection
dff = df[df > 0]

# conditional selection of column
# it removes row C because its W value is less then 0
dfff = df[df['W'] > 0]

# conditional selection of row
# it removes all the rows except C because all the  values are less then 0
df = df[df['Z'] < 0]

print(df)
