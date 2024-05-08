import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
csv_path = 'C:\\Git\\ML\\1-DataProcessing\\Data.csv'
dataset = pd.read_csv(csv_path)

# matrix of features (X) and the dependent variable vector (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#print(x)
#print(y)

# taking care of missing data
# imputation is the process of replacing missing data with substituted values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

# acting only on numberical coluns from X
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# encoding categorical data
# one hot encoding is a technique used in machine learning to represent categorical variables as binary vectors.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# transformers: This parameter is a list of tuples, where each tuple contains three elements:
# A name for the transformation (e.g., 'encoder')
# The transformer to apply (e.g., OneHotEncoder())
# The column indices to apply the transformation to (e.g., [0])

# remainder: This parameter specifies what to do with the remaining columns of the dataset 
# that are not explicitly transformed. Here, it's set to 'passthrough', which means that the 
# remaining columns will be kept as they are without any transformation.

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))


# encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# feature scaling
# technique used in machine learning to standardize or normalize the range of independent variables or 
# features of the dataset. The goal of feature scaling is to ensure that all features have the same scale 
# to prevent some features from dominating the others and to make the learning algorithm converge faster.

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.fit_transform(X_test[:, 3:])



print(X_train)