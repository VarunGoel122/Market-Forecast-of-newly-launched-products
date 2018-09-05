# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 16:07:18 2018
@author: ASISH CHAKRAPANI
"""
# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Laptop.csv')
data = pd.read_csv('data1.csv', header = None)
dataset = pd.concat([dataset, data], axis=0,  ignore_index=True)
X = dataset.iloc[:,0:8].values
#find the right number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 20), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
dataset['Cluster']= y_kmeans
# finding cluster number of the new data
l = dataset.iloc[576,9]
# Grouping by cluster Number
g=dataset.groupby('Cluster')
df= g.get_group(l)
a = df.iloc[:, :-2].values
b = df.iloc[:, 7].values
# train test split
a_train = df.iloc[:-1,:-2].values
b_train = df.iloc[:-1, 8].values
a_test = df.iloc[-1:, :-2].values
b_test = df.iloc[-1: , 8].values
#model fitting
#from sklearn.cross_validation import train_test_split
#a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.1, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(a_train, b_train)
# Predicting the Test set results
b_pred = regressor.predict(a_test)
b_pred = np.round(b_pred)
from sklearn.metrics import mean_squared_error
from math import sqrt
ms = mean_squared_error(b_test, b_pred)
ms1 = sqrt(mean_squared_error(b_test, b_pred ))