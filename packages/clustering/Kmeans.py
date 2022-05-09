from sklearn.cluster import KMeans
import numpy as np

class Kmeans:

    def __init__(self, max_k=20, min_cluster_size=0.05):
        self.sse = []
        self.best_k = None
        self.K = range(2, max_k)
        self.min_cluster_size = min_cluster_size
        self.km = None


    def calculate_errors(self, vectors):
        sse = [] # sum of squared errors
        min_size = int(self.min_cluster_size * vectors.shape[0])	
        
        for k in self.K:
            km = KMeans(n_clusters=k, max_iter=200, n_init=10)
            km = km.fit(vectors)
            sse.append(km.inertia_)  

            # number of elments in each cluster
            cluster_sizes = np.bincount(km.labels_)

            sz = np.min(cluster_sizes)
            if sz < min_size : break

        self.sse = sse


    def choose_best_k(self, vectors):

        self.calculate_errors(vectors)

        # distance from each point to the line
        start_line = np.array([self.K[0], self.sse[0]])
        end_line = np.array([self.K[-1], self.sse[-1]])

        distances = []
        for x, y in zip(self.K, self.sse):
            point = np.array([x, y])
            diff = end_line - start_line
            d = np.linalg.norm(np.cross(diff, start_line-point)) / np.linalg.norm(diff)
            distances.append(d)

        self.best_k  = np.argmax(distances) + self.K[0]
        return self.best_k

    def fit(self, vectors, **kwargs):
        
        self.choose_best_k(vectors)
        
        kwargs['n_clusters'] = self.best_k
        kwargs['max_iter'] = 200

        self.km = KMeans(**kwargs).fit(vectors)