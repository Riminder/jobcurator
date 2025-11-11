from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Callable, Dict, List, Literal, Optional

from .faiss_backends import faiss_cosine_distance, faiss_hash_clusters
from .hash_utils import (
    build_clusters_with_lsh,
    build_exact_hash,
    completion_score,
    composite_signature,
    compute_quality,
    compute_token_length,
    hamming_normalized_distance,
    length_score,
    percentile,
)
from .minhash_backends import minhash_hash_clusters, minhash_jaccard_distance
from .models import Job
from .sklearn_backends import (
    filter_outliers,
    sklearn_cosine_distance,
    sklearn_hash_clusters,
)


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
    backend: Literal["default_hash", "minhash_hash", "sklearn_hash", "faiss_hash"] = (
        "default_hash"
    )
    use_outlier_filter: bool = False
    outlier_contamination: float = 0.05

    # 🎯 Backend-specific thresholds
    d_sim_threshold: int = 20
    max_cluster_distance_km: float = 50.0
    jaccard_threshold: float = 0.8  # ∈ [0,1], only for minhash_hash

    # 🔍 Multi-probe LSH controls
    use_multiprobe: bool = False
    max_multiprobe_flips: int = 1

    jobs: List[Job] = None  # all processed jobs after quality computation(in memory)
    selected_jobs: List[Job] = None  # selected jobs after dedupe_and_compress

    # --- seen-filter helpers (keep inside the curator) ---
    @staticmethod
    def _seen_contains(seen, key: str) -> bool:
        if seen is None or key is None:
            return False
        if hasattr(seen, "contains"):
            try:
                return bool(seen.contains(key))
            except Exception:
                return False
        try:
            return key in seen  # works for set-like
        except Exception:
            return False

    @staticmethod
    def _seen_add(seen, key: str) -> None:
        if seen is None or key is None:
            return
        if hasattr(seen, "add"):
            try:
                seen.add(key)  # Bloom/Cuckoo/Set-like
                return
            except Exception:
                pass
        # best-effort no-op otherwise
        return

    # --- robust percentiles instead of raw min/max ---
    @staticmethod
    def _quantile(xs, q: float) -> float:
        if not xs:
            return 0.0
        q = 0.0 if q < 0.0 else (1.0 if q > 1.0 else q)  # clamp
        s = sorted(xs)
        pos = (len(s) - 1) * q
        i = int(math.floor(pos))
        j = min(i + 1, len(s) - 1)
        frac = pos - i
        return s[i] * (1.0 - frac) + s[j] * frac

    def _diversity_distance(self, a: Job, b: Job) -> float:
        """Return a distance in [0,1] depending on backend:
        - default_hash: Hamming on 64-bit simhash/composite signature (int)
        - minhash_hash: 1 - Jaccard_estimate from MinHash signatures (tuple[int])
        - sklearn_hash: cosine distance on unit-normalized vectors (if available)
        - faiss_hash: L2 distance on vectors, normalized to [0,1] by a cap
        """
        if self.backend == "default_hash":
            return hamming_normalized_distance(a, b)
        if self.backend == "minhash_hash":
            return minhash_jaccard_distance(a, b)
        if self.backend == "sklearn_hash":
            return sklearn_cosine_distance(a, b)
        if self.backend == "faiss_hash":
            return faiss_cosine_distance(a, b)
        # fallback: treat as identical
        return 0.0


    def recompute_diversity_scores(
        self,
        selected_jobs: List[Job],
        alpha: float,
        distance_fn: Callable[[Job, Job], float],
        *,
        k_nn: int = 3,  # use average of k nearest neighbors (softer than min)
        q_lo: float = 0.10,  # robust scaling lower percentile
        q_hi: float = 0.90,  # robust scaling upper percentile
        tau: float = 0.15,  # temperature for optional soft-min (smaller = closer to hard min)
        label_eps: float = 0.02,  # label smoothing to avoid exact 0/1
        use_softmin: bool = False,  # set True to use soft-min instead of k-NN mean
    ) -> List[Job]:
        """
        Softer, robust diversity recompute:
        - diversity_i = robust_scale( avg_kNN_i or softmin_i )
        - LOO_i = robust_scale( drop_in_mean(avg_kNN) when removing i )
        - final = 0.5 * diversity_i + 0.5 * LOO_i
        - selection_score = alpha*quality + (1-alpha)*final
        All distances are assumed in [0,1].
        """

        n = len(selected_jobs)
        if n == 0:
            return []
        if n == 1:
            j = selected_jobs[0]
            j.diversity_score = 1.0 - label_eps
            q = float(getattr(j, "quality", 0.0) or 0.0)
            j.selection_score = alpha * q + (1.0 - alpha) * j.diversity_score
            return [j]

        # 1) Build symmetric distance matrix D (clamped to [0,1])
        D = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for k in range(i + 1, n):
                d = float(distance_fn(selected_jobs[i], selected_jobs[k]))
                d = min(max(d, 0.0), 1.0)
                D[i][k] = d
                D[k][i] = d

        # helper: k-NN mean or soft-min for each row i (excluding self)
        def local_div(i: int) -> float:
            row = [D[i][j] for j in range(n) if j != i]
            if not row:
                return 0.0
            row.sort()
            if use_softmin:
                # soft-min: sum_j w_j * d_j with w ∝ exp(-d_j/tau)
                ws = [math.exp(-x / max(tau, 1e-6)) for x in row]
                s = sum(ws)
                return sum(w * x for w, x in zip(ws, row)) / s if s > 0 else row[0]
            kk = min(k_nn, len(row))
            return sum(row[:kk]) / kk

        # 2) Per-item soft/avg-kNN diversity
        raw_local = [local_div(i) for i in range(n)]

        # robust scaling via quantiles

        lo = self._quantile(raw_local, q_lo)
        hi = self._quantile(raw_local, q_hi)
        span = max(hi - lo, 1e-6)
        local_scaled = [(x - lo) / span for x in raw_local]
        local_scaled = [min(max(x, 0.0), 1.0) for x in local_scaled]

        # 3) Leave-one-out on the same metric (avg kNN or soft-min)
        if n == 2:
            # symmetric case: both get the same diversity
            loo_scaled = [D[0][1], D[0][1]]
        else:
            # baseline: mean of local diversities
            base_mean = sum(local_scaled) / len(local_scaled)

            loo_contrib = []
            for i in range(n):
                # recompute local diversities for u != i without using i
                loc_wo_i = []
                for u in range(n):
                    if u == i:
                        continue
                    # neighbors of u excluding {u,i} / v != u and v != i
                    neighbors = [D[u][v] for v in range(n) if v not in (u, i)]
                    if not neighbors:
                        loc_wo_i.append(0.0)
                        continue
                    neighbors.sort()
                    if use_softmin:
                        ws = [math.exp(-x / max(tau, 1e-6)) for x in neighbors]
                        s = sum(ws)
                        v = (
                            sum(w * x for w, x in zip(ws, neighbors)) / s
                            if s > 0
                            else neighbors[0]
                        )
                    else:
                        kk = min(k_nn, len(neighbors))
                        v = sum(neighbors[:kk]) / kk
                    loc_wo_i.append(v)

                # robust-scale the temporary set like above
                lo2 = self._quantile(loc_wo_i, q_lo)
                hi2 = self._quantile(loc_wo_i, q_hi)
                span2 = max(hi2 - lo2, 1e-6)
                loc_wo_i_scaled = [(x - lo2) / span2 for x in loc_wo_i]
                loc_wo_i_scaled = [min(max(x, 0.0), 1.0) for x in loc_wo_i]

                mean_wo_i = (
                    sum(loc_wo_i_scaled) / len(loc_wo_i_scaled)
                    if loc_wo_i_scaled
                    else 0.0
                )
                drop = max(base_mean - mean_wo_i, 0.0)  # non-negative contribution
                loo_contrib.append(drop)

            # robust scale contributions
            lo_c = self._quantile(loo_contrib, q_lo)
            hi_c = self._quantile(loo_contrib, q_hi)
            span_c = max(hi_c - lo_c, 1e-6)
            loo_scaled = [(x - lo_c) / span_c for x in loo_contrib]
            loo_scaled = [min(max(x, 0.0), 1.0) for x in loo_scaled]

        # 4) Combine signals + label smoothing
        final_div = [
            min(max(0.5 * a + 0.5 * b, label_eps), 1.0 - label_eps)
            for a, b in zip(local_scaled, loo_scaled)
        ]

        # write back
        for j, d in zip(selected_jobs, final_div):
            j.diversity_score = d
            q = float(getattr(j, "quality", 0.0) or 0.0)
            j.selection_score = alpha * q + (1.0 - alpha) * d

        return selected_jobs

    def dedupe_and_compress(
        self,
        jobs: List[Job],
        ratio: Optional[float] = None,
        *,
        greedy_diversity: bool = False,
        seen_filter=None,
    ) -> List[Job]: # pylint: disable=too-many-return-statements
        """
        Deduplicate + cluster + compress with diversity-aware greedy selection.
        Uses an optional seen_filter (Bloom/Cuckoo/set) to skip already-seen jobs
        and updates it with the kept items at the end.
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
            return list(jobs)

        # -------- optional outlier filtering --------
        if self.use_outlier_filter:
            # filter_outliers already imported; if sklearn isn't available,
            # your filter_outliers impl should gracefully no-op or raise.
            jobs = filter_outliers(jobs, contamination=self.outlier_contamination)
            if not jobs:
                return []

        # -------- early skip via seen_filter --------
        if seen_filter is not None:
            fresh: List[Job] = []
            for j in jobs:
                # ensure we have exact_hash available for membership check
                h = getattr(j, "exact_hash", None) or build_exact_hash(j)
                j.exact_hash = h
                if not self._seen_contains(seen_filter, h):
                    fresh.append(j)
            jobs = fresh
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
            # exact_hash may already be set above; keep consistent
            if not getattr(job, "exact_hash", None):
                job.exact_hash = build_exact_hash(job)
            job.signature = composite_signature(job)

        self.jobs = jobs

        # -------- exact dedup --------
        seen_exact: Dict[object, Job] = {}
        for j in jobs:
            h = j.exact_hash

            prev_j = seen_exact.get(h)
            # skip duplicate exact hashes, keep the first / highest quality (we sort clusters later)
            if prev_j is None or (
                j.quality,
                j.length_tokens,
                j.completion_score_val,
                str(j.id),
            ) > (
                prev_j.quality,
                prev_j.length_tokens,
                prev_j.completion_score_val,
                str(prev_j.id),
            ):
                seen_exact[h] = j

        unique_jobs: List[Job] = list(seen_exact.values())
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
                num_perm=64,  # number of MinHash permutations (signature length)
                bands=8,  # number of LSH bands
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
        pool_jobs: List[Job] = []
        for C in clusters:
            if C:
                pool_jobs.extend(C[: self.max_per_cluster_in_pool])

        # dedupe by canonical_id in pool

        by_key = {}
        for j in pool_jobs:
            key = j.canonical_id
            prev_j = by_key.get(key)
            if prev_j is None or (
                j.quality,
                j.length_tokens,
                j.completion_score_val,
            ) > (prev_j.quality, prev_j.length_tokens, prev_j.completion_score_val):
                by_key[key] = j
        pool_jobs = list(by_key.values())

        if not pool_jobs:
            return []

        # -------- determine K on the actual pool_jobs --------
        K = max(1, math.ceil(len(pool_jobs) * r))  # number of items to select
        if K >= len(pool_jobs):
            selected_jobs = sorted(pool_jobs, key=lambda j: j.quality, reverse=True)
            # update seen_filter before returning
            if seen_filter is not None:
                for j in selected_jobs:
                    self._seen_add(seen_filter, j.exact_hash or build_exact_hash(j))
            return selected_jobs

        # -------- diversity-aware greedy selection --------
        alpha = min(
            max(self.alpha, 0.0), 1.0
        )  # clamp alpha to [0,1] to avoid surprises
        pool_jobs.sort(key=lambda j: j.quality, reverse=True)
        seed_job = pool_jobs.pop(0)  # best quality job as seed and remove from pool
        seed_job.diversity_score = 1.0  # better in the case of robust scaling than 0.5
        seed_job.selection_score = (
            alpha * seed_job.quality + (1.0 - alpha) * seed_job.diversity_score
        )

        selected_jobs: List[Job] = [
            seed_job
        ]  # seed selected_jobs with best quality and remove it from pool_jobs (already sorted desc)

        while len(selected_jobs) < K and pool_jobs:
            # compute min diversity distance to any selected_jobs
            dmins = []
            for j in pool_jobs:
                dmin = min(self._diversity_distance(j, s) for s in selected_jobs)
                dmins.append((j, dmin))

            # dmins is a list of (job, dmin) you already computed
            dvals = [d for _, d in dmins]

            # robust scaling via quantiles

            q_lo, q_hi = 0.10, 0.90
            lo = self._quantile(dvals, q_lo)
            hi = self._quantile(dvals, q_hi)
            span = max(hi - lo, 1e-6)  # no magic 1, just tiny epsilon

            # optional label smoothing to avoid exact 0/1
            eps = 0.02

            best_j = None
            best_score = -1.0
            for j, d in dmins:
                z = (d - lo) / span  # robust-scaled
                z = 0.0 if z < 0.0 else (1.0 if z > 1.0 else z)
                z = eps + (1.0 - 2.0 * eps) * z  # smooth to (eps, 1-eps)
                j.diversity_score = z
                j.selection_score = alpha * j.quality + (1.0 - alpha) * z
                if j.selection_score > best_score:
                    best_score = j.selection_score
                    best_j = j

            # safety: best_j should exist since pool not empty
            selected_jobs.append(best_j)
            pool_jobs.remove(best_j)

        # top-up if needed
        if len(selected_jobs) < K and pool_jobs:
            pool_jobs.sort(key=lambda j: j.quality, reverse=True)
            selected_jobs.extend(pool_jobs[: (K - len(selected_jobs))])

        # final recompute diversity properly on the FINAL selected set
        if greedy_diversity:
            self.recompute_diversity_scores(
                selected_jobs, alpha=self.alpha, distance_fn=self._diversity_distance
            )

        # -------- update seen_filter with kept items --------
        if seen_filter is not None:
            for j in selected_jobs:
                self._seen_add(seen_filter, j.exact_hash or build_exact_hash(j))

        self.selected_jobs = sorted(
            selected_jobs, key=lambda j: j.selection_score, reverse=True
        )
        return self.selected_jobs

    @staticmethod
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

    def print_compression_summary(self, n_preview: int = 0, t_ms: float = 0.0) -> None:
        total_jobs = len(getattr(self, "jobs", []) or [])
        selected = getattr(self, "selected_jobs", []) or []
        kept = len(selected)
        keep_ratio = (kept / total_jobs) if total_jobs else 0.0

        line = f" 🔎 preview: {n_preview} | 🎯 ratio: {keep_ratio:.2f} | 🧠 backend: {self.backend} | ⏱️  time: {t_ms:.1f} ms " # pylint: disable=line-too-long
        print("┌" + "─" * (len(line) + 2) + "┐")
        print("│" + line + "│")
        print("└" + "─" * (len(line) + 2) + "┘")
        print("")

        total_stats = self.compute_job_stats(self.jobs)
        comp_stats = self.compute_job_stats(selected)

        # --- Stdlib table ---
        def fmt(st):
            return [
                f"{st['count']}",
                f"{st['length_mean']:.2f}",
                f"{st['length_std']:.2f}",
                f"{st['quality_mean']:.3f}",
                f"{st['quality_std']:.3f}",
            ]

        print(f"Total jobs: {total_jobs} | Selected: {kept} ({keep_ratio:.1%} kept)")
        headers = ["Dataset", "Count", "Len μ", "Len σ", "Qual μ", "Qual σ"]
        rows = [
            ["All jobs", *fmt(total_stats)],
            ["Selected", *fmt(comp_stats)],
        ]

        col_widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]

        def fill_line(fill="-"):
            return "+" + "+".join(fill * (w + 2) for w in col_widths) + "+"

        def fmt_row(vals):
            return (
                "| "
                + " | ".join(str(v).rjust(w) for v, w in zip(vals, col_widths))
                + " |"
            )

        print(fill_line("-"))
        print(fmt_row(headers))
        print(fill_line("="))
        for r in rows:
            print(fmt_row(r))
        print(fill_line("-"))

    @staticmethod
    def print_jobs_summary(jobs, n_preview=10, label="jobs set"):
        n_show = max(0, min(n_preview, len(jobs)))
        rows = []
        for j in jobs[:n_show]:
            city = getattr(getattr(j, "location", None), "city", None) or "Unknown"
            h_str = (
                j.canonical_hash(4) if hasattr(j, "canonical_hash") else None
            ) or "NA"
            q = getattr(j, "quality", None)
            d = getattr(j, "diversity_score", None)
            s = getattr(j, "selection_score", None)
            q_str = f"{q:.3f}" if isinstance(q, (int, float)) else "NA"
            d_str = f"{d:.3f}" if isinstance(d, (int, float)) else "NA"
            s_str = f"{s:.3f}" if isinstance(s, (int, float)) else "NA"
            title = (getattr(j, "title", "") or "").strip()
            rows.append([str(j.id), title, city, q_str, d_str, s_str, h_str])

        # ASCII table
        headers = ["ID", "Title", "City", "Quality", "Diversity", "Selection", "Hash"]
        data = [headers] + rows
        widths = [max(len(str(x)) for x in col) for col in zip(*data)]

        def fill_line(ch="-", cross="+"):
            return cross + cross.join(ch * (w + 2) for w in widths) + cross

        def fmt_row(vals):
            return (
                "| "
                + " | ".join(
                    str(v).ljust(w) if i in (0, 1, 2, 6) else str(v).rjust(w)
                    for i, (v, w) in enumerate(zip(vals, widths))
                )
                + " |"
            )

        print(f"\n=== Top {n_show} jobs from {label} ===")
        print(fill_line("-"))
        print(fmt_row(headers))
        print(fill_line("="))
        for r in rows:
            print(fmt_row(r))
        print(fill_line("-"))
