from __future__ import annotations

from typing import List, Dict
from collections import defaultdict

from .models import Job
from .hash_utils import flatten_category_tokens

_HAS_SKLEARN = True
try:
    import numpy as np  # type: ignore
    from sklearn.feature_extraction.text import HashingVectorizer # type: ignore
    from sklearn.neighbors import NearestNeighbors # type: ignore
    from sklearn.ensemble import IsolationForest # type: ignore
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


def _ensure_sklearn():
    if not _HAS_SKLEARN:
        raise RuntimeError(
            "numpy is required for this backend. Install it with 'pip install numpy'. "
            "scikit-learn is required for this backend. Install it with 'pip install scikit-learn'."
        )


def sklearn_cosine_distance(a: Job, b: Job) -> float:
    # NumPy (assumes vectors are L2-normalized from HashingVectorizer)
    va = np.asarray(a.sklearn_hashvector, dtype=np.float32)
    vb = np.asarray(b.sklearn_hashvector, dtype=np.float32)
    if va.size == 0 or vb.size == 0 or va.shape != vb.shape: return 1.0
    dot = float(np.dot(va, vb))           # cosine similarity
    dot = min(max(dot, 0.0), 1.0)         # numerical clamp
    return 1.0 - dot                      # cosine distance in [0,1]


def sklearn_hash_clusters(
    jobs: List[Job],
    eps: float = 0.2,
    n_features: int = 2**16,
) -> List[List[Job]]:
    """
    Cluster jobs using HashingVectorizer + NearestNeighbors (cosine radius).

    Uses text + encoded 3D location + categories as extra tokens.
    """
    _ensure_sklearn()

    texts = []
    for j in jobs:
        loc_tokens = []
        if j.location is not None:
            j.location.compute_xyz()
            loc_tokens = [
                f"locx_{round(j.location.x / 10_000)}",
                f"locy_{round(j.location.y / 10_000)}",
                f"locz_{round(j.location.z / 10_000)}",
            ]

        cat_tokens = flatten_category_tokens(j)
        augmented = " ".join(
            [j.title, j.text] + loc_tokens + cat_tokens
        )
        texts.append(augmented)

    hv = HashingVectorizer(
        n_features=n_features,
        norm="l2",
        alternate_sign=False,
    )
    X = hv.transform(texts)

    # persist each job's vector row for reuse later
    for j, row_idx in zip(jobs, range(X.shape[0])):
        j.sklearn_hashvector =  X[row_idx].toarray().ravel().astype("float32").tolist()

    nn = NearestNeighbors(metric="cosine")
    nn.fit(X)

    n = len(jobs)
    parent = list(range(n))

    def find(i: int) -> int:
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        xi = X[i]
        distances, indices = nn.radius_neighbors(
            xi, radius=eps, return_distance=True
        )
        for d, j in zip(distances[0], indices[0]):
            if i == j:
                continue
            union(i, j)

    clusters: Dict[int, List[Job]] = defaultdict(list)
    for idx, job in enumerate(jobs):
        root = find(idx)
        clusters[root].append(job)

    return list(clusters.values())


def filter_outliers(
    jobs: List[Job],
    contamination: float = 0.05,
) -> List[Job]:
    """
    Remove outlier jobs based on numeric features using IsolationForest.
    """
    _ensure_sklearn()

    if not jobs:
        return jobs

    X = [j.numerical_vector(j) for j in jobs]
    iso = IsolationForest(
        contamination=contamination,
        random_state=0,
    )
    labels = iso.fit_predict(X)  # 1 = normal, -1 = outlier

    filtered: List[Job] = []
    for j, lbl in zip(jobs, labels):
        if lbl == 1:
            filtered.append(j)
    return filtered

