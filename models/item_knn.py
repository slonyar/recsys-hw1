import numpy as np
from scipy import sparse


class ItemKNN:

    def __init__(self, neighbors=50, shrinkage=10):
        self.neighbors = neighbors
        self.shrinkage = shrinkage

    def fit(self, positive):
        self.history = sparse.csr_matrix(positive, dtype=np.float64)
        co = (self.history.T @ self.history).tocsr()
        norms = np.sqrt(co.diagonal())
        co.setdiag(0)
        co.eliminate_zeros()
        for i in range(co.shape[0]):
            start, end = co.indptr[i:i + 2]
            ids = co.indices[start:end]
            count = co.data[start:end].copy()
            co.data[start:end] = count / (norms[i] * norms[ids])
            co.data[start:end] *= count / (count + self.shrinkage)
            if end - start > self.neighbors:
                order = np.lexsort((ids, -co.data[start:end]))
                co.data[start + order[self.neighbors:]] = 0
        co.eliminate_zeros()
        self.weights = co
        return self

    def score(self, users):
        return (self.history[users] @ self.weights).toarray()
