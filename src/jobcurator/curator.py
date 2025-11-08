from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
import math

from .models import Job
from .hash_utils import (
    compute_token_length,
    percentile,
    length_score,
    completion_score,
    compute_quality,
    build_exact_hash,
    composite_signature,
    hamming_distance,
    build_clusters_with_lsh,
)
from .sklearn_backends import (
    sklearn_hash_clusters,
    filter_outliers,
)
from .faiss_backends import faiss_hash_clusters


@dataclass
class JobCurator:
    """
    Main entrypoint for dedupe + compression.

    backend:
      - "default_hash" : SimHash + LSH (default)
      - "sklearn_hash" : HashingVectorizer + NearestNeighbors (scikit-learn)
      - "faiss_hash"   : FAISS on hash-based signature + 3D location + categories
    """
    ratio: float = 1.0
    alpha: float = 0.6
    max_per_cluster_in_pool: int = 3
    d_sim_threshold: int = 20
    max_cluster_distance_km: float = 150.0

    backend: Literal["default_hash", "sklearn_hash", "faiss_hash"] = "default_hash"

    use_outlier_filter: bool = False
    outlier_contamination: float = 0.05

    def dedupe_and_compress(self,
                            jobs: List[Job],
                            ratio: Optional[float] = None) -> List[Job]:
        r = self.ratio if ratio is None else ratio

        if r >= 1.0:
            return list(jobs)
        if r <= 0.0:
            return []
        if not jobs:
            return []

        # Optional outlier filtering
        if self.use_outlier_filter:
            jobs = filter_outliers(jobs, contamination=self.outlier_contamination)
            if not jobs:
                return []

        N_original = len(jobs)
        K = math.ceil(N_original * r)

        # 1) length stats
        lengths = [compute_token_length(j) for j in jobs]
        lengths_sorted = sorted(lengths)
        p10 = percentile(lengths_sorted, 0.10)
        p90 = percentile(lengths_sorted, 0.90)

        # 2) compute internal scores + hashes
        for job, l in zip(jobs, lengths):
            job.length_tokens = l
            job.length_score = length_score(l, p10, p90)
            job.completion_score_val = completion_score(job)
            job.quality = compute_quality(job)
            job.exact_hash = build_exact_hash(job)
            job.signature = composite_signature(job)

        # 3) exact dedup
        seen_exact: Dict[int, str] = {}
        unique_jobs: List[Job] = []
        for job in jobs:
            if job.exact_hash in seen_exact:
                continue
            seen_exact[job.exact_hash] = job.id
            unique_jobs.append(job)

        if not unique_jobs:
            return []

        # 4) clusters: choose backend
        if self.backend == "default_hash":
            clusters = build_clusters_with_lsh(
                unique_jobs,
                d_sim_threshold=self.d_sim_threshold,
                max_cluster_distance_km=self.max_cluster_distance_km,
            )
        elif self.backend == "sklearn_hash":
            clusters = sklearn_hash_clusters(unique_jobs)
        elif self.backend == "faiss_hash":
            clusters = faiss_hash_clusters(
                unique_jobs,
                dim=128,
                max_neighbors=self.max_per_cluster_in_pool * 4,
                hamming_threshold=self.d_sim_threshold,
            )
        else:  # pragma: no cover
            raise ValueError(f"Unknown backend: {self.backend}")

        # 5) rank inside clusters by quality
        for C in clusters:
            C.sort(key=lambda j: j.quality, reverse=True)

        # 6) candidate pool
        pool: List[Job] = []
        for C in clusters:
            pool.extend(C[: self.max_per_cluster_in_pool])

        pool_dict: Dict[str, Job] = {j.id: j for j in pool}
        pool = list(pool_dict.values())
        if not pool:
            return []

        # 7) diversity-aware greedy selection
        pool.sort(key=lambda j: j.quality, reverse=True)
        selected: List[Job] = []

        first = pool.pop(0)
        selected.append(first)

        alpha = self.alpha

        while len(selected) < K and pool:
            dmins = []
            for x in pool:
                dmin = min(hamming_distance(x.signature, s.signature) for s in selected)
                dmins.append((x, dmin))

            dvals = [d for _, d in dmins]
            dmin_val, dmax_val = min(dvals), max(dvals)
            span = max(dmax_val - dmin_val, 1)

            best_x = None
            best_score = -1.0
            for x, d in dmins:
                diversity = (d - dmin_val) / span
                score = alpha * x.quality + (1 - alpha) * diversity
                if score > best_score:
                    best_score = score
                    best_x = x

            selected.append(best_x)
            pool.remove(best_x)

        if len(selected) < K and pool:
            for x in pool:
                if len(selected) >= K:
                    break
                selected.append(x)

        return selected
