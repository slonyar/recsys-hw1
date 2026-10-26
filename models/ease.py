import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve


class EASE:

    def __init__(self, reg=100):
        if reg <= 0:
            raise ValueError("Regularization must be positive")
        self.reg = reg

    def fit(self, positive):
        self.history = sparse.csr_matrix(positive, dtype=np.float64)
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::gram.shape[0] + 1] += self.reg
        factor = cho_factor(gram, overwrite_a=True, check_finite=False)
        precision = cho_solve(factor, np.eye(gram.shape[0]), check_finite=False)
        precision /= -np.diag(precision).copy()[None, :]
        np.fill_diagonal(precision, 0)
        self.weights = precision
        return self

    def score(self, users):
        return self.history[users] @ self.weights
