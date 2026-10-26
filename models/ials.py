import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve


class ImplicitALS:

    def __init__(self, factors=32, alpha=20, reg=1.0, iterations=10, seed=42):
        self.factors = factors
        self.alpha = alpha
        self.reg = reg
        self.iterations = iterations
        self.seed = seed

    def _update(self, target, fixed):
        gram = fixed.T @ fixed + self.reg * np.eye(self.factors)
        result = np.zeros((target.shape[0], self.factors))
        for row in range(target.shape[0]):
            start, end = target.indptr[row:row + 2]
            if start == end:
                continue
            features = fixed[target.indices[start:end]]
            lhs = gram + self.alpha * (features.T @ features)
            rhs = (1 + self.alpha) * (features.T @ target.data[start:end])
            result[row] = cho_solve(cho_factor(lhs, check_finite=False), rhs,
                                    check_finite=False)
        return result

    def fit(self, positive, target=None):
        positive = sparse.csr_matrix(positive, dtype=np.float64)
        positive.sort_indices()
        target = positive.copy() if target is None else sparse.csr_matrix(target)
        target.sort_indices()
        if not (np.array_equal(target.indptr, positive.indptr)
                and np.array_equal(target.indices, positive.indices)):
            raise ValueError("Target and positive confidence support must match")
        if self.reg <= 0 or self.alpha < 0 or not np.isfinite(target.data).all():
            raise ValueError("Invalid ALS parameters or target")
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, 0.05, (positive.shape[1], self.factors))
        transpose = target.T.tocsr()
        for _ in range(self.iterations):
            self.user_factors = self._update(target, self.item_factors)
            self.item_factors = self._update(transpose, self.user_factors)
        return self

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
