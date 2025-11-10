from __future__ import annotations

from typing import Dict, List
from collections import defaultdict

from .models import Job
from .hash_utils import flatten_category_tokens



_HAS_FAISS = True
try:
    import numpy as np # type: ignore
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    _HAS_FAISS = False


def _ensure_faiss():
    if not _HAS_FAISS:
        raise RuntimeError(
            "numpy is required for this backend. Install with: pip install numpy. "
            "faiss is required for this backend. Install it with 'pip install faiss-cpu' "
            "or the appropriate FAISS package for your platform."
        )


def signature_bits_vector(sig: int, dim_sig: int) -> np.ndarray:
    """
    Convert the job.signature into a {0,1}^dim_sig vector (float32).
    """
    dim_sig = min(dim_sig, 128)
    bits = []
    for i in range(dim_sig):
        bit = (sig >> i) & 1
        bits.append(float(bit))
    return np.array(bits, dtype="float32")


def build_faiss_vector(job: Job, dim_sig: int = 128) -> np.ndarray:
    """
    Full FAISS vector = [signature bits, normalized 3D location, category richness].

    - signature bits: dim_sig dims in {0,1}
    - location: x,y,z normalized by Earth radius
    - categories: number of flattened category tokens
    """
    sig_vec = signature_bits_vector(job.signature, dim_sig)

    if job.location is not None:
        job.location.compute_xyz()
        R = 6_400_000.0
        loc_vec = np.array(
            [
                job.location.x / R,
                job.location.y / R,
                job.location.z / R,
            ],
            dtype="float32",
        )
    else:
        loc_vec = np.zeros(3, dtype="float32")

    cat_tokens = flatten_category_tokens(job)
    cat_vec = np.array([float(len(cat_tokens))], dtype="float32")

    return np.concatenate([sig_vec, loc_vec, cat_vec], axis=0)

def _vec(job) -> np.ndarray:
    """float32, contiguous 1D array from stored list."""
    v = np.asarray(job.faiss_hashvector, dtype=np.float32)
    return np.ascontiguousarray(v)

# Distances
def l2sq(a, b) -> float:           # squared L2 (same metric FAISS IndexFlatL2 uses)
    va, vb = _vec(a), _vec(b)
    d = va - vb
    return float(np.dot(d, d))

def faiss_cosine_distance(a, b) -> float:    # cosine distance in [0, 2] (≈[0,1] if nonnegative)
    va, vb = _vec(a), _vec(b)
    na = np.linalg.norm(va); nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0: return 1.0
    sim = float(np.dot(va, vb) / (na * nb))
    return 1.0 - max(min(sim, 1.0), -1.0)

def faiss_hash_clusters(
    jobs: List[Job],
    dim: int = 128,
    max_neighbors: int = 10,
    hamming_threshold: int = 20,
) -> List[List[Job]]:
    """
    Cluster jobs using FAISS on composite vectors:

        [signature bits, normalized 3D location (x,y,z), category richness]

    For signature bits (0/1), squared L2 ≈ Hamming distance.
    """
    _ensure_faiss()

    if not jobs:
        return []

    n = len(jobs)
    dim_sig = min(dim, 128)
    extra_dim = 4  # x,y,z,cat_count
    full_dim = dim_sig + extra_dim

    X = np.zeros((n, full_dim), dtype="float32")
    for idx, job in enumerate(jobs):
        X[idx] = build_faiss_vector(job, dim_sig=dim_sig)
        job.faiss_dim_sig = dim_sig
        job.faiss_hashvector = X[idx].toarray().ravel().astype(np.float32).tolist()

    index = faiss.IndexFlatL2(full_dim)
    index.add(X)

    max_neighbors = min(max_neighbors, n)

    parent = list(range(n))

    def find(i: int) -> int:
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    max_dist = float(hamming_threshold)

    for i in range(n):
        xi = X[i : i + 1]
        distances, indices = index.search(xi, max_neighbors)
        for d, j in zip(distances[0], indices[0]):
            if j == i:
                continue
            if d <= max_dist:
                union(i, j)

    clusters_dict: Dict[int, List[Job]] = defaultdict(list)
    for idx, job in enumerate(jobs):
        root = find(idx)
        clusters_dict[root].append(job)

    return list(clusters_dict.values())