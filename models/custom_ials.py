import numpy as np
from scipy import sparse

from .ials import ImplicitALS


class CustomIALS(ImplicitALS):

    def __init__(self, half_life_days=730, **kwargs):
        super().__init__(**kwargs)
        self.half_life_days = half_life_days

    def make_target(self, events, shape, cutoff):
        positive = events[events.rating >= 4]
        age = (cutoff - positive.timestamp.to_numpy()) / 86400
        if np.any(age < 0):
            raise ValueError("Events after the fitting cutoff")
        strength = (positive.rating.to_numpy() - 3) / 2
        if self.half_life_days is not None:
            if self.half_life_days <= 0:
                raise ValueError("Half-life must be positive")
            strength *= 0.5 + 0.5 * np.exp2(-age / self.half_life_days)
        return sparse.csr_matrix((strength, (positive.user, positive.item)), shape=shape)

    def fit(self, positive, events, cutoff):
        return super().fit(positive, self.make_target(events, positive.shape, cutoff))
