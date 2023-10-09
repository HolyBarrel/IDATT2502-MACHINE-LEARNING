import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d, art3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

data = pd.read_csv('agaricus-lepiota.data', sep=',', header=0, engine='python')

#https://saturncloud.io/blog/how-to-draw-a-distribution-of-a-column-in-pandas/#:~:text=To%20draw%20a%20distribution%20of%20a%20column%20in%20Pandas%2C%20we,the%20distribution%20of%20a%20dataset.
print(data.head())

print(data.describe())

print("\nData shape: ")
print(data.shape)

print("\nData info: ")
print(data.info())

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(data)


pca = PCA(n_components=3) 
X_reduced = pca.fit_transform(X_encoded.toarray())

db = DBSCAN(min_samples=10).fit(X_reduced)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)


kmeans = KMeans(n_clusters=n_clusters_)
kmeans.fit(X_reduced)
y_kmeans = kmeans.predict(X_reduced)
db = DBSCAN(min_samples=10).fit(X_reduced)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y_kmeans, s=50, cmap='cool', alpha=0.05)
centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=200, alpha=0.95)

plt.title(f'K-Means Clustering with {n_clusters_} Clusters')
plt.xlabel('First Principal Component (Captures Most Variance)')
plt.ylabel('Second Principal Component (Captures Second Most Variance)')
plt.show()

for n_clusters in range(n_clusters_, 11):
    # Performs K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_reduced)
    y_kmeans = kmeans.predict(X_reduced)
    
    # Plots the clusters
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y_kmeans, s=50, cmap='cool', alpha=0.05)
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=200, alpha=0.95)
    
    plt.title(f'K-Means Clustering with {n_clusters} Clusters')
    plt.xlabel('First Principal Component (Captures Most Variance)')
    plt.ylabel('Second Principal Component (Captures Second Most Variance)')
    plt.show()