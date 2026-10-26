import gzip
import html
import json
import re
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from scipy import sparse


SOURCES = {
    "Musical_Instruments_5.json.gz":
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Musical_Instruments_5.json.gz",
    "meta_Musical_Instruments.json.gz":
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Musical_Instruments.json.gz",
}


def download_data(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = directory / name
        if not path.exists():
            try:
                urlretrieve(url, path)
            except Exception:
                path.unlink(missing_ok=True)
                raise


def clean_text(value):
    return " ".join(re.sub(r"<[^>]*>", " ", html.unescape(str(value or ""))).split())


def load_data(directory):
    with gzip.open(Path(directory) / "Musical_Instruments_5.json.gz", "rt") as source:
        records = [json.loads(line) for line in source]
    raw = pd.DataFrame(records)[["reviewerID", "asin", "overall", "unixReviewTime"]]
    raw.columns = ["user_id", "item_id", "rating", "timestamp"]
    valid = (raw.notna().all(axis=1) & raw.rating.between(1, 5)
             & (raw.timestamp > 0))
    cleaned = raw[valid].sort_values("timestamp", kind="stable")
    cleaned = cleaned.drop_duplicates(["user_id", "item_id"], keep="first").reset_index(drop=True)
    metadata = {}
    meta_rows = 0
    category_nonempty = 0
    with gzip.open(Path(directory) / "meta_Musical_Instruments.json.gz", "rt") as source:
        for line in source:
            record = json.loads(line)
            meta_rows += 1
            category_nonempty += bool(record.get("category"))
            metadata.setdefault(record["asin"], {
                "title": clean_text(record.get("title")),
                "brand": clean_text(record.get("brand")),
            })
    meta = pd.DataFrame.from_dict(metadata, orient="index")
    meta.index.name = "item_id"
    stats = {"raw_rows": len(raw), "invalid_rows": int((~valid).sum()),
             "repeated_pairs": int(valid.sum() - len(cleaned)),
             "clean_rows": len(cleaned), "users": cleaned.user_id.nunique(),
             "items": cleaned.item_id.nunique(), "metadata_rows": meta_rows,
             "metadata_duplicate_ids": meta_rows - len(meta),
             "metadata_nonempty_categories": category_nonempty}
    return raw, cleaned, meta, stats


def split_data(data):
    first, second = np.quantile(data.timestamp, [0.8, 0.9], method="nearest")
    first, second = int(first), int(second)
    raw_parts = [data[data.timestamp <= first],
                 data[(data.timestamp > first) & (data.timestamp <= second)],
                 data[data.timestamp > second]]
    user_ids = sorted(raw_parts[0].user_id.unique())
    item_ids = sorted(raw_parts[0].item_id.unique())
    users = {key: i for i, key in enumerate(user_ids)}
    items = {key: i for i, key in enumerate(item_ids)}
    parts, rows = [], []
    for label, frame in zip(["train", "validation", "test"], raw_parts):
        warm_user, warm_item = frame.user_id.isin(users), frame.item_id.isin(items)
        warm = frame[warm_user & warm_item].copy()
        warm["user"] = warm.user_id.map(users).astype(int)
        warm["item"] = warm.item_id.map(items).astype(int)
        parts.append(warm)
        rows.append({"split": label, "all_events": len(frame),
                     "cold_user_events": int((~warm_user).sum()),
                     "cold_item_events": int((~warm_item).sum()),
                     "warm_events": len(warm),
                     "warm_positive_events": int((warm.rating >= 4).sum())})
    return parts, (len(users), len(items)), item_ids, (first, second), pd.DataFrame(rows)


def interaction_matrix(events, shape, positive_only=False):
    frame = events[events.rating >= 4] if positive_only else events
    return sparse.csr_matrix((np.ones(len(frame)), (frame.user, frame.item)), shape=shape)


def item_features(metadata, item_ids, n_users, with_metadata=True):
    frame = metadata.reindex(item_ids).fillna("")
    tokens = [sorted(set(re.findall(r"[a-z]{3,}", title.lower()))) for title in frame.title]
    frequency = Counter(word for row in tokens for word in row)
    stop = {"the", "and", "with", "from", "for", "feat", "featuring", "version",
            "album", "single", "explicit", "remastered", "original", "import"}
    vocabulary = [w for w, count in sorted(frequency.items(), key=lambda x: (-x[1], x[0]))
                  if count >= 5 and w not in stop][:512]
    brands = sorted(set(frame.brand.str.lower()) - {""})
    offset = n_users + 1 + len(item_ids)
    brand_map = {a: offset + j for j, a in enumerate(["<missing>"] + brands)}
    token_map = {w: offset + len(brand_map) + j for j, w in enumerate(vocabulary)}
    indices, values = [], []
    for i, (brand, words) in enumerate(zip(frame.brand.str.lower(), tokens)):
        ids, weights = [n_users + 1 + i], [1.0]
        if with_metadata:
            ids.append(brand_map.get(brand, brand_map["<missing>"]))
            weights.append(1.0)
            selected = [w for w in words if w in token_map][:8]
            ids.extend(token_map[w] for w in selected)
            if selected:
                weights.extend([1 / np.sqrt(len(selected))] * len(selected))
        indices.append(ids)
        values.append(weights)
    width = max(map(len, indices))
    ids = np.zeros((len(indices), width), dtype=np.int64)
    weights = np.zeros_like(ids, dtype=np.float32)
    for row, (index, value) in enumerate(zip(indices, values)):
        ids[row, :len(index)] = index
        weights[row, :len(value)] = value
    return ids, weights, {"brands": len(brands), "title_words": len(vocabulary),
                          "missing_brand": int((frame.brand == "").sum()),
                          "missing_title": int((frame.title == "").sum())}


def evaluation_users(events, seen, shape, k=10):
    truth = interaction_matrix(events, shape, positive_only=True)
    truth = truth - truth.multiply(seen)
    truth.eliminate_zeros()
    eligible = (np.asarray(truth.sum(axis=1)).ravel() > 0)
    eligible &= shape[1] - seen.getnnz(axis=1) >= k
    return np.flatnonzero(eligible), truth


def recommend(model, users, seen, k=10):
    scores = np.asarray(model.score(users)).copy()
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite prediction")
    if np.any(scores.shape[1] - seen[users].getnnz(axis=1) < k):
        raise ValueError("Fewer than k unseen candidates")
    rows, cols = seen[users].nonzero()
    scores[rows, cols] = -np.inf
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def ranking_metrics(recommendations, users, truth):
    k = recommendations.shape[1]
    selected = truth[users[:, None], recommendations]
    hits = (selected.toarray() if sparse.issparse(selected) else np.asarray(selected)) > 0
    relevant_count = np.asarray(truth[users].sum(axis=1)).ravel()
    discount = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.cumsum(discount)[np.minimum(relevant_count.astype(int), k) - 1]
    return pd.DataFrame({"user": users, "NDCG@10": (hits * discount).sum(axis=1) / ideal,
                         "Recall@10": hits.sum(axis=1) / relevant_count})


def evaluate(model, users, seen, truth, batch_size=128):
    per_user, unique_items = [], set()
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        recs = recommend(model, batch, seen)
        per_user.append(ranking_metrics(recs, batch, truth))
        unique_items.update(recs.ravel().tolist())
    detail = pd.concat(per_user, ignore_index=True)
    metrics = detail[["NDCG@10", "Recall@10"]].mean().to_dict()
    metrics["Coverage@10"] = len(unique_items) / seen.shape[1]
    metrics["users"] = len(users)
    return metrics, detail
