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

from .minhash_backends import minhash_hash_clusters
from .sklearn_backends import (
    sklearn_hash_clusters,
    filter_outliers,
)
from .faiss_backends import faiss_hash_clusters


@dataclass
class JobCurator:
    """
    Main entrypoint for job deduplication + compression with diversity.

    Backends
    --------
    `backend` controls which clustering / hashing strategy is used:

      - "default_hash"
          SimHash + LSH (with optional Multi-probe) on text, plus 3D geo distance
          and meta-hash (categories, salary, location). Pure Python.

      - "minhash_hash"
          MinHash + Jaccard LSH on shingles built from text, categories,
          coarse location and salary. Optional Multi-probe + 3D geo distance.
          Pure Python.

      - "sklearn_hash"
          HashingVectorizer + NearestNeighbors (cosine radius) over text +
          encoded 3D location + category tokens. Optional IsolationForest
          for outlier filtering. Requires scikit-learn.

      - "faiss_hash"
          FAISS IndexFlatL2 on composite vectors
          [signature bits + normalized (x,y,z) + category richness].
          Designed for large-scale catalogs. Requires faiss-cpu.


    Global parameters (all backends)
    --------------------------------
    ratio : float
        Target compression ratio in [0, 1].
        Example: ratio = 0.4 → keep ~40% of jobs after dedupe + selection.

    alpha : float
        Trade-off between quality and diversity in the final greedy selection:
            score = alpha * quality + (1 - alpha) * diversity
        where diversity is based on Hamming distance between signatures.

    max_per_cluster_in_pool : int
        Maximum number of jobs taken from each cluster into the global
        candidate pool before the diversity-aware selection.

    backend : {"default_hash", "minhash_hash", "sklearn_hash", "faiss_hash"}
        Which clustering / hashing backend to use.

    use_outlier_filter : bool
        When True (and scikit-learn is installed), runs an IsolationForest-based
        outlier filter on numeric features BEFORE clustering. Applies to any
        backend. When False, no outlier filtering is performed.

    outlier_contamination : float
        Proportion of expected outliers for IsolationForest when
        use_outlier_filter=True. Ignored otherwise.


    Backend-specific parameters
    ---------------------------

    d_sim_threshold : int
        Similarity / distance threshold for some backends:

          - "default_hash":
              Maximum Hamming distance on the SimHash (64-bit) part of
              the composite signature to consider two jobs as near-duplicates.

          - "faiss_hash":
              Approximate maximum L2 distance in FAISS space to connect jobs
              in the same cluster.

          - "minhash_hash", "sklearn_hash":
              Ignored.

    max_cluster_distance_km : float
        Maximum allowed 3D geo distance (in kilometers) between jobs in
        the same cluster:

          - used by "default_hash" and "minhash_hash",
          - ignored by "sklearn_hash" and "faiss_hash".

    jaccard_threshold : float
        ONLY used by the "minhash_hash" backend.
        Minimum Jaccard similarity (in [0, 1]) between two jobs’ shingle sets
        for them to be connected in the same cluster.
        Ignored by "default_hash", "sklearn_hash", and "faiss_hash".


    Multi-probe LSH (default_hash / minhash_hash)
    ---------------------------------------------

    use_multiprobe : bool
        When True, enables Multi-probe LSH:

          - "default_hash":
              Probes neighboring buckets in SimHash LSH by flipping bits
              in band keys.

          - "minhash_hash":
              Probes neighboring buckets in MinHash band hashes (hashed bands).

          - "sklearn_hash", "faiss_hash":
              Ignored.

    max_multiprobe_flips : int
        When use_multiprobe=True, controls how many bit flips are used to
        generate neighboring bucket keys (higher = more recall, more CPU).
        Used by "default_hash" and "minhash_hash".
        Ignored by "sklearn_hash" and "faiss_hash".

    
    For incremental SQL / local-file usage.
    ---------------------------------------------

    See also:
        jobcurator.storage.process_batch
        jobcurator.storage.SqlStoreDB
        jobcurator.storage.LocalFileStoreDB

    """

    # 🌍 Global parameters (all backends)
    ratio: float = 1.0
    alpha: float = 0.6
    max_per_cluster_in_pool: int = 3
    backend: Literal["default_hash", "minhash_hash", "sklearn_hash", "faiss_hash"] = "default_hash"
    use_outlier_filter: bool = False
    outlier_contamination: float = 0.05

    # 🎯 Backend-specific thresholds
    d_sim_threshold: int = 20
    max_cluster_distance_km: float = 50.0
    jaccard_threshold: float = 0.8  # ∈ [0,1], only for minhash_hash

    # 🔍 Multi-probe LSH controls
    use_multiprobe: bool = False
    max_multiprobe_flips: int = 1


    def dedupe_and_compress(
    self,
    jobs: List[Job],
    ratio: Optional[float] = None) -> List[Job]:
        """
        Deduplicate + cluster + compress with diversity-aware greedy selection.
        The effective keep-count is computed on the candidate pool (post-dedupe).
        """
        # -------- input & early guards --------
        r = self.ratio if ratio is None else ratio
        if not jobs:
            return []
        if r is None or math.isnan(r):
            r = 1.0
        if r <= 0.0:
            return []
        if r >= 1.0:
            # Still do exact-hash dedupe if you prefer; otherwise return as-is
            return list(jobs)

        # Optional outlier filtering (only if explicitly enabled)
        if self.use_outlier_filter:
            # filter_outliers already imported; if sklearn isn't available,
            # your filter_outliers impl should gracefully no-op or raise.
            jobs = filter_outliers(jobs, contamination=self.outlier_contamination)
            if not jobs:
                return []

        # -------- per-item stats & signatures --------
        lengths = [compute_token_length(j) for j in jobs]
        lengths_sorted = sorted(lengths)
        p10 = percentile(lengths_sorted, 0.10)
        p90 = percentile(lengths_sorted, 0.90)

        for job, l in zip(jobs, lengths):
            job.length_tokens = l
            job.length_score = length_score(l, p10, p90)
            job.completion_score_val = completion_score(job)
            job.quality = compute_quality(job)
            job.exact_hash = build_exact_hash(job)
            job.signature = composite_signature(job)

        # -------- exact dedup --------
        seen_exact: Dict[str, str] = {}
        unique_jobs: List[Job] = []
        for job in jobs:
            # skip duplicate exact hashes, keep the first / highest quality (we sort clusters later)
            if job.exact_hash in seen_exact:
                continue
            seen_exact[job.exact_hash] = job.id
            unique_jobs.append(job)

        if not unique_jobs:
            return []

        # -------- clustering (backend switch) --------
        if self.backend == "default_hash":
            clusters = build_clusters_with_lsh(
                unique_jobs,
                d_sim_threshold=self.d_sim_threshold,
                max_cluster_distance_km=self.max_cluster_distance_km,
                use_multiprobe=self.use_multiprobe,
                max_multiprobe_flips=self.max_multiprobe_flips,
            )
        elif self.backend == "minhash_hash":
            clusters = minhash_hash_clusters(
                unique_jobs,
                num_perm=64,               # number of MinHash permutations (signature length)
                bands=8,                   # number of LSH bands
                jaccard_threshold=self.jaccard_threshold,
                max_cluster_distance_km=self.max_cluster_distance_km,
                use_multiprobe=self.use_multiprobe,
                max_multiprobe_flips=self.max_multiprobe_flips,
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

        # If no clusters formed, bail early
        if not clusters:
            return []

        # -------- rank within clusters by quality --------
        for C in clusters:
            C.sort(key=lambda j: j.quality, reverse=True)

        # -------- candidate pool (cap per cluster) --------
        pool: List[Job] = []
        for C in clusters:
            if C:
                pool.extend(C[: self.max_per_cluster_in_pool])

        # Remove accidental dupes by id (can happen if backends overlap edges)
        if not pool:
            return []
        pool = list({j.id: j for j in pool}.values())

        # -------- determine K on the actual pool --------
        K = max(1, math.ceil(len(pool) * r))
        if K >= len(pool):
            # nothing to compress—return pool as-is in quality order
            return sorted(pool, key=lambda j: j.quality, reverse=True)

        # -------- diversity-aware greedy selection --------
        pool.sort(key=lambda j: j.quality, reverse=True)
        selected: List[Job] = []

        # seed with the best quality
        first = pool.pop(0)
        selected.append(first)

        # clamp alpha to [0,1] to avoid surprises
        alpha = self.alpha
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0

        while len(selected) < K and pool:
            # compute each candidate's min Hamming distance to any selected
            dmins = []
            for x in pool:
                dmin = min(hamming_distance(x.signature, s.signature) for s in selected)
                dmins.append((x, dmin))

            # normalize diversity to [0,1] for stability
            dvals = [d for _, d in dmins]
            dmin_val, dmax_val = min(dvals), max(dvals)
            span = max(dmax_val - dmin_val, 1)

            best_x = None
            best_score = -1.0
            for x, d in dmins:
                diversity = (d - dmin_val) / span
                score = alpha * x.quality + (1.0 - alpha) * diversity
                if score > best_score:
                    best_score = score
                    best_x = x

            # safety: best_x should exist since pool not empty
            selected.append(best_x)
            pool.remove(best_x)

        # if we still need to fill due to ties/rounding, top up by quality
        if len(selected) < K and pool:
            pool.sort(key=lambda j: j.quality, reverse=True)
            selected.extend(pool[: (K - len(selected))])

        return selected
