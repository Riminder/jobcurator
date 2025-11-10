from __future__ import annotations

from typing import List, Dict, Set, Optional
from collections import defaultdict

from .models import Job
from .hash_utils import (
    normalize_text,
    flatten_category_tokens,
    location_bucket,
    salary_bucket,
    hash_int,
    geo_distance_km,
    multiprobe_band_keys,
)


def job_to_shingles(
    job: Job,
    word_shingle_size: int = 3,
) -> List[str]:
    """
    Build a set-like list of shingles for MinHash:

    - word n-grams on title+text
    - flattened categories
    - coarse location bucket
    - salary bucket
    """
    base = normalize_text(f"{job.title} {job.text}")
    words = base.split()

    shingles: List[str] = []

    # word n-grams
    if len(words) >= word_shingle_size:
        for i in range(len(words) - word_shingle_size + 1):
            shingles.append("w:" + " ".join(words[i : i + word_shingle_size]))
    else:
        for w in words:
            shingles.append("w:" + w)

    # categories
    cat_tokens = flatten_category_tokens(job)
    for t in cat_tokens:
        shingles.append("c:" + t)

    # coarse location
    loc_b = location_bucket(job)
    if loc_b:
        shingles.append("loc:" + loc_b)

    # salary bucket
    sal_b = salary_bucket(job)
    if sal_b:
        shingles.append("sal:" + sal_b)

    return shingles


def minhash_signature(
    tokens: List[str],
    num_perm: int = 64,
) -> List[int]:
    """
    Simple MinHash: for each permutation index i, we use
    a seeded hash and take the minimum over all tokens.
    """
    if not tokens:
        return [0] * num_perm

    # large initial value
    max_hash = (1 << 31) - 1
    sig = [max_hash] * num_perm

    for t in tokens:
        for i in range(num_perm):
            h = hash_int(t, seed=1000 + i, bits=32)
            if h < sig[i]:
                sig[i] = h
    return sig


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    uni = len(a | b)
    return inter / uni

def minhash_jaccard_distance(a: Job, b: Job) -> float:
    sig_a = getattr(a, "minhash_sig", None)
    sig_b = getattr(b, "minhash_sig", None)

    # If either is missing, fall back (choose policy: return 1.0 = maximally different)
    if not isinstance(sig_a, list) or not isinstance(sig_b, list):
        return 1.0

    # Require same length
    if len(sig_a) != len(sig_b) or len(sig_a) == 0:
        return 1.0

    eq = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    jaccard_hat = eq / float(len(sig_a))
    return 1.0 - jaccard_hat


def minhash_hash_clusters(
    jobs: List[Job],
    num_perm: int = 64,
    bands: int = 8,
    jaccard_threshold: float = 0.8,
    max_cluster_distance_km: float = 50.0,
    use_multiprobe: bool = False,
    max_multiprobe_flips: int = 1,
) -> List[List[Job]]:
    """
    Cluster jobs using MinHash + Jaccard LSH:

    - tokens = shingles from text + categories + coarse loc + salary
    - similarity = Jaccard(tokens_a, tokens_b)
    - candidate pairs from LSH banding + optional multi-probe
    - optional geo distance filter like default_hash backend
    """
    if not jobs:
        return []

    assert num_perm % bands == 0, "num_perm must be divisible by bands"
    rows_per_band = num_perm // bands

    # 1) tokens + MinHash signatures
    tokens_map: Dict[str, Set[str]] = {}
    sig_map: Dict[str, List[int]] = {}

    for j in jobs:
        toks = job_to_shingles(j)
        tokens_map[j.id] = set(toks)
        sig_map[j.id] = ensure_minhash(j, num_perm=num_perm)


    # 2) LSH buckets (banding + optional multi-probe)
    buckets: Dict[int, List[Job]] = defaultdict(list)

    for j in jobs:
        sig = sig_map[j.id]
        for b in range(bands):
            band_vals = sig[b * rows_per_band : (b + 1) * rows_per_band]
            band_str = ",".join(str(v) for v in band_vals)

            # hash band contents into a 64-bit value
            band_hash_val = hash_int(band_str, seed=b, bits=64)

            if use_multiprobe:
                # reuse the same multiprobe pattern as default_hash,
                # here on the 64-bit band hash value
                keys = multiprobe_band_keys(
                    band_idx=b,
                    band_bits=band_hash_val,
                    r=64,
                    max_flips=max_multiprobe_flips,
                )
            else:
                keys = [(b, band_hash_val)]

            for (bi, val) in keys:
                bucket_key = hash_int(f"{bi}:{val}", seed=777, bits=64)
                buckets[bucket_key].append(j)

    # 3) union–find on candidate pairs, with Jaccard + geo filters
    parent: Dict[str, str] = {j.id: j.id for j in jobs}

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for bucket_jobs in buckets.values():
        n = len(bucket_jobs)
        for i in range(n):
            for k in range(i + 1, n):
                a = bucket_jobs[i]
                b = bucket_jobs[k]

                jac = jaccard_similarity(tokens_map[a.id], tokens_map[b.id])
                if jac < jaccard_threshold:
                    continue

                if geo_distance_km(a.location, b.location) > max_cluster_distance_km:
                    continue

                union(a.id, b.id)

    clusters_dict: Dict[str, List[Job]] = defaultdict(list)
    for j in jobs:
        root = find(j.id)
        clusters_dict[root].append(j)

    return list(clusters_dict.values())

def ensure_minhash(job: Job, num_perm: int = 64) -> List[int]:
    """
    Return a MinHash signature for this job, computing and storing it on first use.
    Recomputes only if the stored signature length doesn't match num_perm.
    """
    if job.minhash_sig and job.minhash_len == num_perm:
        return job.minhash_sig

    toks = job_to_shingles(job)
    sig = minhash_signature(toks, num_perm=num_perm)
    job.minhash_sig = sig
    return sig