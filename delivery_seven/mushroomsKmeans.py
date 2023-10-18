import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d, art3d

data = pd.read_csv('agaricus-lepiota.data', sep=',', header=0, engine='python')
# data.pop('edibility')

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

X_encoded = pd.get_dummies(data)

pca = PCA(n_components=30) 

X_reduced = pca.fit_transform(X_encoded)

def compute_silhouette_scores(X, k_values):
    scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        scores.append(score)
    return scores

def plot_silhouette_scores(k_values, silhouette_scores):
    plt.figure()
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.show()

k_values = range(2, 31)
silhouette_scores = compute_silhouette_scores(X_reduced, k_values)

plot_silhouette_scores(k_values, silhouette_scores)

optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]

print(f'Optimal k: {optimal_k}')

kmeans = KMeans(n_clusters=optimal_k, random_state=42).fit(X_reduced)
cluster_centers = kmeans.cluster_centers_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=kmeans.labels_, cmap='Set1', alpha=0.05)
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], c='black', marker='x', s=100, label='Cluster Centers')

ax.set_title(f'3D Visualization with {optimal_k} Clusters')
ax.legend()
plt.show()