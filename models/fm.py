import numpy as np
import torch
from torch import nn


class FactorizationMachine:

    def __init__(self, factors=32, learning_rate=0.01, reg=0.001,
                 epochs=12, batch_size=2048, seed=42):
        self.factors = factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def _pairs(self, users, items):
        ids = torch.cat((users[:, None] + 1, self.item_ids[items]), dim=1)
        values = torch.cat((torch.ones((len(users), 1)), self.item_values[items]), dim=1)
        vectors = self.embedding(ids) * values[:, :, None]
        linear = (self.linear(ids).squeeze(-1) * values).sum(dim=1)
        interactions = 0.5 * (vectors.sum(dim=1).square()
                              - vectors.square().sum(dim=1)).sum(dim=1)
        return linear + interactions, vectors

    def fit(self, positive, seen, feature_ids, feature_values):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.n_users, self.n_items = positive.shape
        self.item_ids = torch.as_tensor(feature_ids, dtype=torch.long)
        self.item_values = torch.as_tensor(feature_values, dtype=torch.float32)
        n_features = max(int(feature_ids.max()) + 1, self.n_users + 1)
        self.embedding = nn.Embedding(n_features, self.factors, padding_idx=0)
        self.linear = nn.Embedding(n_features, 1, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.03)
        nn.init.zeros_(self.linear.weight)
        with torch.no_grad():
            self.embedding.weight[0] = 0
        optimizer = torch.optim.Adam(list(self.embedding.parameters())
                                     + list(self.linear.parameters()), lr=self.learning_rate)
        users, items = positive.nonzero()
        keep = np.asarray(seen.sum(axis=1)).ravel()[users] < self.n_items
        users, items = users[keep], items[keep]
        if not len(users):
            raise ValueError("No positive pairs with unseen candidates")
        self.losses = []
        for _ in range(self.epochs):
            total = 0.0
            order = rng.permutation(len(users))
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                u, p = users[batch], items[batch]
                neg = rng.integers(self.n_items, size=len(batch))
                invalid = np.asarray(seen[u, neg]).ravel() > 0
                while invalid.any():
                    neg[invalid] = rng.integers(self.n_items, size=int(invalid.sum()))
                    invalid = np.asarray(seen[u, neg]).ravel() > 0
                ut = torch.as_tensor(u, dtype=torch.long)
                pos, vp = self._pairs(ut, torch.as_tensor(p, dtype=torch.long))
                negative, vn = self._pairs(ut, torch.as_tensor(neg, dtype=torch.long))
                loss = torch.nn.functional.softplus(negative - pos).mean()
                loss = loss + self.reg * (vp.square().sum() + vn.square().sum()) / len(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch)
            self.losses.append(total / len(users))
        self._cache_items()
        return self

    def _cache_items(self):
        with torch.no_grad():
            vectors = self.embedding(self.item_ids) * self.item_values[:, :, None]
            summed = vectors.sum(dim=1)
            bias = (self.linear(self.item_ids).squeeze(-1) * self.item_values).sum(dim=1)
            bias += 0.5 * (summed.square() - vectors.square().sum(dim=1)).sum(dim=1)
            self.item_vectors = summed.numpy()
            self.item_bias = bias.numpy()
            self.user_vectors = self.embedding.weight[1:self.n_users + 1].detach().numpy()
            self.user_bias = self.linear.weight[1:self.n_users + 1, 0].detach().numpy()

    def score(self, users):
        return (self.user_vectors[users] @ self.item_vectors.T
                + self.item_bias[None, :] + self.user_bias[users, None])
