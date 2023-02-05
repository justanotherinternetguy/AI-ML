import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from numpy.random import uniform
import random

CENTERS = 4

# generate
X_train, true_labels = make_blobs(n_samples=200,
                                  centers=CENTERS,
                                  cluster_std=1,
                                  random_state=50)

X_train = StandardScaler().fit_transform(X_train)



### HELPER ###
# euclidean dist for distance between a point and a dataset of points
def euclidean_distance(point, data):
    return np.sqrt(np.sum((point - data) ** 2, axis=1))

# init-ed with a value of k and a max number of iterations for finding centroid locs
class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X_train):
        # random select centroid init pts, uniformly distributed across dataset
        min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]

        # iterate thru centroids until optimized or pass max_iter
        iters = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iters < self.max_iter:
            # sort datapoints, assign to nearest centroid
            sorted_pts = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = euclidean_distance(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_pts[centroid_idx].append(x)

            # push curr centroids to previous centroids, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_pts]

            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            iters+=1

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean_distance(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)

        return centroids, centroid_idxs

km = KMeans(n_clusters=CENTERS)
km.fit(X_train)

class_centers, classification = km.evaluate(X_train)

sns.scatterplot(
    x = [X[0] for X in X_train],
    y = [X[1] for X in X_train],
    hue = classification,
    legend=None
)

plt.plot(
    [x for x, _ in km.centroids],
    [y for _, y in km.centroids],
    'X',
    markersize=10
)


plt.xlabel("x")
plt.ylabel("y")
plt.show()
