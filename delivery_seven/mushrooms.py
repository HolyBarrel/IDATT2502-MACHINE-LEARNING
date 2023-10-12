import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d, art3d

data = pd.read_csv('agaricus-lepiota.data', sep=',', header=0, engine='python')

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA



def cluster_scan(data, samples, eps):
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(data)

    pca = PCA(n_components=3) 
    X_reduced = pca.fit_transform(X_encoded.toarray())

    db = DBSCAN(min_samples=samples, eps=eps).fit(X_reduced)
    labels = db.labels_

    return X_reduced, labels


eps_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

plot_eps = []
plot_min_samples = []
plot_silhouette_scores = []

for i in range(1, 30):
    for eps in eps_values:
        # print("min_samples=", i, ", eps=", eps)
        X_reduced, labels = cluster_scan(data, i, eps)

        # If no noise, then DBSCAN found no clusters. Skip silhouette calculation.
        if -1 not in labels:
            # print("DBSCAN found no clusters. Skipping silhouette calculation.")
            continue

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        
        # Silhouette score is only meaningful if there's more than one cluster
        if n_clusters_ > 1:
            silhouette_avg = silhouette_score(X_reduced, labels)
            # print("Estimated number of clusters: %d" % n_clusters_)
            # print("Estimated number of noise points: %d" % n_noise_)
            # print("Silhouette Coefficient: %0.3f" % silhouette_avg)
            plot_eps.append(eps)
            plot_min_samples.append(i)
            plot_silhouette_scores.append(silhouette_avg)
        else:
            continue
            # print("Only one cluster found. Silhouette score is not meaningful.")

max_silhouette_score = max(plot_silhouette_scores)
max_index = plot_silhouette_scores.index(max_silhouette_score)
corresponding_eps = plot_eps[max_index]
corresponding_min_samples = plot_min_samples[max_index]

print(f"Max silhouette score: {max_silhouette_score}")
print(f"Corresponding eps: {corresponding_eps}")
print(f"Corresponding min_samples: {corresponding_min_samples}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(plot_eps, plot_min_samples, plot_silhouette_scores, c=plot_silhouette_scores, cmap='viridis')
ax.set_xlabel('Eps')
ax.set_ylabel('Min_samples')
ax.set_zlabel('Silhouette Score')
plt.colorbar(ax.scatter(plot_eps, plot_min_samples, plot_silhouette_scores, c=plot_silhouette_scores, cmap='viridis'))
plt.title('Silhouette Score vs. Eps and Min_samples')
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