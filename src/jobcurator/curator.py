from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
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


@dataclass
class JobCurator:
    """
    Main entrypoint for dedupe + compression.
    """
    ratio: float = 1.0
    alpha: float = 0.6
    max_per_cluster_in_pool: int = 3
    d_sim_threshold: int = 20
    max_cluster_distance_km: float = 150.0

    def dedupe_and_compress(self,
                            jobs: List[Job],
                            ratio: Optional[float] = None) -> List[Job]:
        r = self.ratio if ratio is None else ratio

        if r >= 1.0:
            return list(jobs)
        if r <= 0.0:
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

        # 4) clusters
        clusters = build_clusters_with_lsh(
            unique_jobs,
            d_sim_threshold=self.d_sim_threshold,
            max_cluster_distance_km=self.max_cluster_distance_km,
        )

        # 5) rank inside clusters by quality
        for C in clusters:
            C.sort(key=lambda j: j.quality, reverse=True)

        # 6) candidate pool
        pool: List[Job] = []
        for C in clusters:
            pool.extend(C[: self.max_per_cluster_in_pool])

        # dedup pool by id
        pool_dict: Dict[str, Job] = {j.id: j for j in pool}
        pool = list(pool_dict.values())

        # 7) diversity-aware greedy selection
        pool.sort(key=lambda j: j.quality, reverse=True)
        selected: List[Job] = []

        # bootstrap with best-quality job
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
