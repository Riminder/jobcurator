from __future__ import annotations

from typing import List, Dict
from collections import defaultdict
from statistics import mean, pstdev

from .models import Job
from .hash_utils import flatten_category_tokens

_HAS_SKLEARN = True
try:
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.neighbors import NearestNeighbors
    from sklearn.ensemble import IsolationForest
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


def _ensure_sklearn():
    if not _HAS_SKLEARN:
        raise RuntimeError(
            "scikit-learn is required for this feature. Install it with 'pip install scikit-learn'."
        )


def job_to_vector(job: Job):
    """
    Numeric feature vector for ML-based filters / stats:

    - length_tokens
    - completion_score_val
    - quality
    - avg salary
    - 3D location (x,y,z)
    - category richness (count of flattened tokens)
    """
    if job.location is not None:
        job.location.compute_xyz()
        x = job.location.x
        y = job.location.y
        z = job.location.z
    else:
        x = y = z = 0.0

    sal = 0.0
    if job.salary is not None:
        vals = []
        if job.salary.min_value is not None:
            vals.append(job.salary.min_value)
        if job.salary.max_value is not None:
            vals.append(job.salary.max_value)
        if vals:
            sal = sum(vals) / len(vals)

    cat_tokens = flatten_category_tokens(job)
    cat_count = float(len(cat_tokens))

    return [
        float(job.length_tokens),
        float(job.completion_score_val),
        float(job.quality),
        float(sal),
        float(x),
        float(y),
        float(z),
        cat_count,
    ]


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

    X = [job_to_vector(j) for j in jobs]
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


def compute_job_stats(jobs: List[Job]) -> dict:
    """
    Simple stats on length and quality using stdlib only.
    """
    if not jobs:
        return {
            "length_mean": 0.0,
            "length_std": 0.0,
            "quality_mean": 0.0,
            "quality_std": 0.0,
            "count": 0,
        }

    lengths = [j.length_tokens for j in jobs]
    qualities = [j.quality for j in jobs]

    return {
        "length_mean": float(mean(lengths)),
        "length_std": float(pstdev(lengths)) if len(lengths) > 1 else 0.0,
        "quality_mean": float(mean(qualities)),
        "quality_std": float(pstdev(qualities)) if len(qualities) > 1 else 0.0,
        "count": len(jobs),
    }

