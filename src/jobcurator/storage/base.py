from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Set

from jobcurator import JobCurator, CuckooFilter, Job
from jobcurator.hash_utils import hamming_distance


@dataclass
class LightJob:
    """
    Minimal metadata required for global selection.

    REQUIRED for dedup/compression:
      - id
      - quality
      - signature
    """
    id: str
    quality: float
    signature: int


class StoreDB(Protocol):
    """
    Abstract persistence layer for compressed jobs + CuckooFilter.

    Implementations:
      - SqlStoreDB  (see sql_store.py)
      - LocalFileStoreDB (see local_store.py)
      - or any custom backend you want.
    """

    # --- CuckooFilter state ---

    def load_or_create_cuckoo(self, capacity: int = 1_000_000) -> CuckooFilter:
        """Load existing CuckooFilter or create a new one."""
        ...

    def save_cuckoo(self, cf: CuckooFilter) -> None:
        """Persist CuckooFilter state."""
        ...

    # --- Compressed jobs ---

    def insert_compressed_jobs(self, compressed_jobs: List[Job], backend: str) -> None:
        """
        Persist compressed jobs.

        For the algorithm, only the following fields are strictly required:
          - job.id
          - job.quality
          - job.signature

        All other fields are optional metadata.
        """
        ...

    def load_all_light_jobs(self) -> List[LightJob]:
        """
        Load minimal per-job metadata needed for global selection:
          - id
          - quality
          - signature
        """
        ...

    def overwrite_with_selected(self, selected_ids: Set[str]) -> None:
        """
        Keep only jobs whose id is in selected_ids, remove all others
        from the backend (e.g., DELETE in SQL or rewrite JSONL).
        """
        ...


def global_reselect(
    jobs_meta: List[LightJob],
    ratio: float,
    alpha: float = 0.6,
) -> List[LightJob]:
    """
    Global selection across multiple batches, based on:
      - quality
      - diversity via Hamming distance on signature.

    Uses only:
      - id
      - quality
      - signature
    """
    if not jobs_meta or ratio <= 0.0:
        return []

    if ratio >= 1.0:
        return list(jobs_meta)

    import math
    N_original = len(jobs_meta)
    K = math.ceil(N_original * ratio)

    # 1) Sort by quality
    pool = sorted(jobs_meta, key=lambda j: j.quality, reverse=True)

    # 2) Greedy quality/diversity
    selected: List[LightJob] = []

    # Start with best-quality job
    first = pool.pop(0)
    selected.append(first)

    while len(selected) < K and pool:
        dmins = []
        for x in pool:
            dmin = min(
                hamming_distance(x.signature, s.signature) for s in selected
            )
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

    # Fill if needed (rare)
    if len(selected) < K and pool:
        for x in pool:
            if len(selected) >= K:
                break
            selected.append(x)

    return selected


def process_batch(
    store: StoreDB,
    jobs: List[Job],
    curator: JobCurator,
    cuckoo_capacity: int = 1_000_000,
) -> List[Job]:
    """
    Storage-agnostic incremental pipeline for a batch:

      1. Load (or create) the global CuckooFilter from the store.
      2. Deduplicate & compress jobs via JobCurator (approx_seen_filter).
      3. Persist compressed jobs via the store.
      4. Save updated CuckooFilter back to the store.
    """
    # 1) Load or create CuckooFilter
    seen_filter = store.load_or_create_cuckoo(capacity=cuckoo_capacity)

    # 2) Run JobCurator
    compressed = curator.dedupe_and_compress(
        jobs,
        seen_filter=seen_filter,
    )

    # 3) Persist compressed jobs
    store.insert_compressed_jobs(compressed, backend=curator.backend)

    # 4) Save CuckooFilter
    store.save_cuckoo(seen_filter)

    return compressed


def global_reselect_in_store(
    store: StoreDB,
    ratio: float,
    alpha: float = 0.6,
) -> None:
    """
    Storage-agnostic global re-selection:

      1. Load LightJob metadata from store.
      2. Compute global selection using quality + diversity.
      3. Instruct store to keep only the selected IDs.
    """
    jobs_meta = store.load_all_light_jobs()
    selected = global_reselect(jobs_meta, ratio=ratio, alpha=alpha)
    selected_ids = {j.id for j in selected}
    store.overwrite_with_selected(selected_ids)
