# For numerical computations
import numpy as np

# Representation and manipulation of data
import pandas as pd

# To randomly split data into training and testing set
from sklearn.model_selection import train_test_split

# To standardize and normalize data
from sklearn import preprocessing

# Random forest uses multiple different decision trees to get a better ouput
from sklearn.ensemble import RandomForestRegressor

# To create a pipeline from standardization to prediction
from sklearn.pipeline import make_pipeline

# Parameter estimation with GridSearch using Cross Validation
from sklearn.model_selection import GridSearchCV

# Scoring the accuracy of predictions
from sklearn.metrics import mean_squared_error, r2_score

# To store the model for future use
from sklearn.externals import joblib

# Red Wine dataset from University of Massachussettes, Amherst
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

# Read data into pandas dataframe
data = pd.read_csv(dataset_url, sep=";")

# y is a vector of wine quality
y = data.quality

# x contains features such as different acidities, sugar levels, chlorides
X = data.drop('quality', axis=1)

# Split into training and testing examples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

# Making a pipeline of standardization using Standard Scaler and building model of Random Forest Regressor
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

# Tuning hyperparameters
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                    'randomforestregressor__max_depth': [None, 5, 3, 1]}

# Exhaustive search over specified parameter values for pipeline estimator
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

# Fit the model
clf.fit(X_train, y_train)

# Get predictions for test data
y_pred = clf.predict(X_test)

print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R2 Score', r2_score(y_test, y_pred))