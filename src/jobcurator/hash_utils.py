from __future__ import annotations

from typing import Dict, List
from collections import defaultdict
from datetime import datetime
import hashlib
import math

from .models import Job, Location3DField


# -------------------------
# Basic helpers
# -------------------------


def normalize_text(s: str) -> str:
    if not s:
        return ""
    return " ".join(s.lower().strip().split())


def hash_int(value: str, seed: int = 0, bits: int = 64) -> int:
    """Stable hash -> int with fixed bits using blake2b."""
    h = hashlib.blake2b(digest_size=bits // 8, person=str(seed).encode("utf-8"))
    h.update(value.encode("utf-8"))
    return int.from_bytes(h.digest(), "big")


def hamming_distance(a: int, b: int) -> int:
    """
    Hamming distance between two integers; compatible with Python < 3.8 (no int.bit_count).
    """
    x = a ^ b
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count


def percentile(sorted_vals, q: float) -> float:
    """q in [0,1], simple percentile on sorted list."""
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# -------------------------
# Quality scoring
# -------------------------


def compute_token_length(job: Job) -> int:
    """Token length of title + text."""
    text = f"{job.title} {job.text}"
    return len(normalize_text(text).split())


def length_score(len_tokens: int, p10: float, p90: float) -> float:
    if p90 <= p10:
        return 0.5
    clipped = max(p10, min(p90, len_tokens))
    return (clipped - p10) / (p90 - p10)


def completion_score(job: Job) -> float:
    """
    Completion based on presence of key fields.
    """
    fields_present = 0
    fields_total = 0

    def add_field(value_present: bool):
        nonlocal fields_present, fields_total
        fields_total += 1
        if value_present:
            fields_present += 1

    add_field(bool(normalize_text(job.title)))
    add_field(bool(normalize_text(job.text)))
    add_field(job.location is not None)

    if job.salary is not None:
        salary_non_empty = (job.salary.min_value is not None or
                            job.salary.max_value is not None)
        add_field(salary_non_empty)

    has_categories = any(cats for cats in (job.categories or {}).values())
    add_field(has_categories)

    add_field(bool(job.company))
    add_field(bool(job.contract_type))

    return fields_present / fields_total if fields_total > 0 else 0.0


def freshness_score(job: Job) -> float:
    if not job.created_at:
        return 0.5
    age_days = (datetime.utcnow() - job.created_at).days
    if age_days <= 0:
        return 1.0
    if age_days >= 365:
        return 0.0
    return 1.0 - (age_days / 365.0)


def source_quality(job: Job) -> float:
    if job.source is None:
        return 0.5
    preferred = {"direct", "first_party"}
    if job.source in preferred:
        return 0.8
    return 0.5


def compute_quality(job: Job,
                    w_len=0.3,
                    w_comp=0.4,
                    w_fresh=0.2,
                    w_src=0.1) -> float:
    return (
        w_len   * job.length_score +
        w_comp  * job.completion_score_val +
        w_fresh * freshness_score(job) +
        w_src   * source_quality(job)
    )


# -------------------------
# Categories → flat tokens
# -------------------------


def flatten_category_tokens(job: Job) -> List[str]:
    tokens: List[str] = []
    for dim, cats in (job.categories or {}).items():
        for c in cats:
            tokens.append(f"{dim}:label:{normalize_text(c.label)}")
            for lvl, name in enumerate(c.level_path or []):
                tokens.append(f"{dim}:path:{lvl}:{normalize_text(name)}")
            tokens.append(f"{dim}:id:{c.id}")
    return tokens


# -------------------------
# SimHash & signatures
# -------------------------


def simhash_from_tokens(tokens, bits: int = 64) -> int:
    if not tokens:
        return 0
    v = [0] * bits
    for tok in tokens:
        h = hash_int(tok, seed=1, bits=bits)
        for i in range(bits):
            bit = (h >> i) & 1
            v[i] += 1 if bit else -1
    fp = 0
    for i in range(bits):
        if v[i] > 0:
            fp |= (1 << i)
    return fp


def text_tokens(job: Job):
    return normalize_text(f"{job.title} {job.text}").split()


def build_simhash(job: Job, bits: int = 64) -> int:
    return simhash_from_tokens(text_tokens(job), bits=bits)


def build_meta_signature(job: Job, bits: int = 64) -> int:
    values = []

    values.append("title:" + normalize_text(job.title))

    for tok in flatten_category_tokens(job):
        values.append("cat:" + tok)

    if job.location:
        loc_str = f"{job.location.city or ''}|{job.location.country_code or ''}"
        coord_str = f"{job.location.lat:.6f},{job.location.lon:.6f},{job.location.alt_m:.2f}"
        values.append("loc:" + normalize_text(loc_str))
        values.append("coord:" + coord_str)

    if job.salary:
        salary_str = f"{job.salary.min_value}-{job.salary.max_value}|{job.salary.currency}|{job.salary.period}"
        values.append("sal:" + salary_str)

    bitset = 0
    num_hashes = 4
    for v in values:
        for k in range(num_hashes):
            h = hash_int(v, seed=100 + k, bits=32)
            pos = h % bits
            bitset |= (1 << pos)
    return bitset


def composite_signature(job: Job) -> int:
    sim = build_simhash(job, bits=64)
    meta = build_meta_signature(job, bits=64)
    return (sim << 64) | meta


# -------------------------
# Geo utilities
# -------------------------


def geo_distance_km(loc1: Location3DField,
                    loc2: Location3DField,
                    earth_radius_m: float = 6_371_000.0) -> float:
    if loc1.x == loc1.y == loc1.z == 0.0:
        loc1.compute_xyz(earth_radius_m)
    if loc2.x == loc2.y == loc2.z == 0.0:
        loc2.compute_xyz(earth_radius_m)

    dx = loc1.x - loc2.x
    dy = loc1.y - loc2.y
    dz = loc1.z - loc2.z
    return math.sqrt(dx*dx + dy*dy + dz*dz) / 1000.0


def location_bucket(job: Job) -> str:
    if not job.location:
        return ""
    lat_bucket = round(job.location.lat, 1)
    lon_bucket = round(job.location.lon, 1)
    return f"{lat_bucket:.1f},{lon_bucket:.1f}"


def salary_bucket(job: Job) -> str:
    if not job.salary:
        return ""
    if job.salary.min_value is None and job.salary.max_value is None:
        return ""
    values = [v for v in [job.salary.min_value, job.salary.max_value] if v is not None]
    avg = sum(values) / len(values)
    bucket = int(avg // 5000) * 5000
    return f"{bucket}"


# -------------------------
# Exact hash & LSH clustering
# -------------------------


def build_exact_hash(job: Job) -> int:
    canonical_title = normalize_text(job.title)
    cat_tokens = flatten_category_tokens(job)
    categories_norm = ",".join(sorted(cat_tokens))
    loc = location_bucket(job)
    sal = salary_bucket(job)
    key = "||".join([canonical_title, categories_norm, loc, sal])
    text_norm = normalize_text(job.text)
    return hash_int(key + "||" + text_norm, seed=42, bits=64)


def split_into_bands(simhash: int, bits: int = 64, bands: int = 8):
    assert bits % bands == 0
    r = bits // bands
    for b in range(bands):
        band_bits = 0
        for i in range(r):
            if (simhash >> (b * r + i)) & 1:
                band_bits |= (1 << i)
        yield b, band_bits


def build_clusters_with_lsh(
    jobs: List[Job],
    sim_bits: int = 64,
    bands: int = 8,
    d_sim_threshold: int = 20,
    max_cluster_distance_km: float = 150.0,
) -> List[List[Job]]:
    """
    Clusters jobs:
      - candidate pairs from LSH on SimHash(text)
      - filtered by SimHash Hamming distance and geo distance
    """
    simhash_map: Dict[str, int] = {}
    for job in jobs:
        simhash_map[job.id] = (job.signature >> 64) & ((1 << sim_bits) - 1)

    buckets: Dict[int, List[Job]] = defaultdict(list)
    for job in jobs:
        sim = simhash_map[job.id]
        for band_idx, band_bits in split_into_bands(sim, bits=sim_bits, bands=bands):
            bkey = hash_int(f"{band_idx}:{band_bits}", seed=999, bits=64)
            buckets[bkey].append(job)

    parent: Dict[str, str] = {job.id: job.id for job in jobs}

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for bucket_jobs in buckets.values():
        n = len(bucket_jobs)
        for i in range(n):
            for j in range(i + 1, n):
                a = bucket_jobs[i]
                b = bucket_jobs[j]
                da = hamming_distance(simhash_map[a.id], simhash_map[b.id])
                if da > d_sim_threshold:
                    continue
                if geo_distance_km(a.location, b.location) > max_cluster_distance_km:
                    continue
                union(a.id, b.id)

    clusters_dict: Dict[str, List[Job]] = defaultdict(list)
    for job in jobs:
        root = find(job.id)
        clusters_dict[root].append(job)
    return list(clusters_dict.values())
