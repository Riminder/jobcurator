#!/usr/bin/env python
"""
test_incremental.py

Small demo script for the incremental pipeline:

- Generates synthetic batches of Job objects
- Runs incremental dedup + compression with JobCurator
- Uses a StoreDB implementation (local files or SQL)
- Optionally runs a global re-selection across ALL stored compressed jobs
"""

import argparse
import os
import random
from datetime import datetime, timedelta
from typing import List

from jobcurator import Category, Job, JobCurator, Location3DField, SalaryField
from jobcurator.storage import (
    LocalFileStoreDB,
    SqlStoreDB,
    global_reselect_in_store,
    process_batch,
)

# ---------------------------------------------------------------------------
# Synthetic job generator (for demo purposes)
# ---------------------------------------------------------------------------


def make_dummy_job(
    job_id: str, city: str, country: str, lat: float, lon: float, idx: int
) -> Job:
    """
    Create a synthetic Job with minimal realistic fields.
    The dedup/compression logic only strictly needs: id, quality, signature
    (which JobCurator will compute), but we fill some contextual fields.
    """
    title_variants = [
        "Senior Backend Engineer",
        "Backend Engineer",
        "Software Engineer Backend",
        "Lead Backend Developer",
    ]
    text_variants = [
        "We are looking for a backend engineer to build scalable APIs.",
        "Join our platform team to work on distributed services.",
        "Help us design and implement microservices in Python.",
        "Contribute to our data processing and API layers.",
    ]
    companies = ["HrFlow.ai", "Acme Corp", "Globex", "Initech"]
    contract_types = ["Full-time", "Part-time", "Contract"]

    title = random.choice(title_variants)
    text = random.choice(text_variants)
    company = random.choice(companies)
    contract_type = random.choice(contract_types)

    categories = {
        "job_function": [
            Category(
                id="backend",
                label="Backend",
                level=2,
                parent_id="software",
                level_path=["Engineering", "Software", "Backend"],
            )
        ]
    }

    location = Location3DField(
        lat=lat,
        lon=lon,
        alt_m=35.0,
        city=city,
        country_code=country,
    )

    salary = SalaryField(
        min_value=60000 + idx * 1000,
        max_value=80000 + idx * 1000,
        currency="EUR",
        period="year",
    )

    created_at = datetime.utcnow() - timedelta(days=random.randint(0, 30))

    return Job(
        id=job_id,
        title=title,
        text=text,
        categories=categories,
        location=location,
        salary=salary,
        company=company,
        contract_type=contract_type,
        source="synthetic",
        created_at=created_at,
    )


def make_batch(batch_idx: int, n_per_batch: int) -> List[Job]:
    """
    Create a batch of jobs.
    We intentionally add some overlapping IDs and similar content across batches
    by reusing a few job_ids.
    """
    jobs: List[Job] = []

    # Some cities for location diversity
    city_specs = [
        ("Paris", "FR", 48.8566, 2.3522),
        ("London", "GB", 51.5074, -0.1278),
        ("Berlin", "DE", 52.5200, 13.4050),
        ("New York", "US", 40.7128, -74.0060),
    ]

    for i in range(n_per_batch):
        city, country, lat, lon = random.choice(city_specs)

        # Create some overlapping IDs across batches for demo
        base_id = f"demo-job-{i}"
        if batch_idx > 0 and i < n_per_batch // 4:
            job_id = base_id  # re-used ID → will be considered exact duplicate
        else:
            job_id = f"{base_id}-b{batch_idx}"

        jobs.append(make_dummy_job(job_id, city, country, lat, lon, idx=i))

    return jobs


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test incremental JobCurator pipeline."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="default_hash",
        choices=["default_hash", "minhash_hash", "sklearn_hash", "faiss_hash"],
        help="Hashing backend for JobCurator.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Global compression ratio (keep ~ratio of jobs).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Quality vs diversity trade-off (0..1).",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="local",
        choices=["local", "sql"],
        help="Storage mode: 'local' (JSONL + pickle) or 'sql' (Postgres/MySQL).",
    )
    parser.add_argument(
        "--dsn",
        type=str,
        default="",
        help="SQL DSN/connection string (used only if --storage sql).",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=2,
        help="Number of synthetic batches to generate and process.",
    )
    parser.add_argument(
        "--n-per-batch",
        type=int,
        default=20,
        help="Number of jobs per synthetic batch.",
    )
    parser.add_argument(
        "--no-global-reselect",
        action="store_true",
        help="Disable final global re-selection step.",
    )
    parser.add_argument(
        "--clear-local",
        action="store_true",
        help="If using local storage, clear previous data files before running.",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Choose storage
    # -----------------------------------------------------------------------
    if args.storage == "local":
        from jobcurator.storage.local_store import (
            DEFAULT_CUCKOO_PATH,
            DEFAULT_JOBS_PATH,
        )

        if args.clear_local:
            for path in (DEFAULT_JOBS_PATH, DEFAULT_CUCKOO_PATH):
                if os.path.exists(path):
                    print(f"[info] Removing old local file: {path}")
                    os.remove(path)

        store = LocalFileStoreDB()
        print("[info] Using LocalFileStoreDB (JSONL + pickle).")

    else:  # sql
        if not args.dsn:
            raise SystemExit(
                "ERROR: --dsn is required when using --storage sql "
                "(e.g. 'dbname=... user=... password=... host=...')"
            )

        try:
            import psycopg2  # type: ignore # user is responsible for installing psycopg2

        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "psycopg2 is required for SQL storage. Install with: pip install psycopg2"
            ) from e

        conn = psycopg2.connect(args.dsn)
        store = SqlStoreDB(conn)
        print("[info] Using SqlStoreDB with DSN:", args.dsn)

    # -----------------------------------------------------------------------
    # Configure JobCurator
    # -----------------------------------------------------------------------
    curator = JobCurator(
        backend=args.backend,
        ratio=args.ratio,
        alpha=args.alpha,
        max_per_cluster_in_pool=3,
        d_sim_threshold=20,
        max_cluster_distance_km=50.0,
        jaccard_threshold=0.8,
        use_outlier_filter=False,
        outlier_contamination=0.05,
        use_multiprobe=True,
        max_multiprobe_flips=1,
    )

    print(f"[info] Backend       : {args.backend}")
    print(f"[info] Ratio         : {args.ratio}")
    print(f"[info] Alpha         : {args.alpha}")
    print(f"[info] Batches       : {args.batches}")
    print(f"[info] Jobs/batch    : {args.n_per_batch}")
    print()

    # -----------------------------------------------------------------------
    # Process batches incrementally
    # -----------------------------------------------------------------------
    total_raw = 0
    total_compressed = 0

    for b in range(args.batches):
        jobs = make_batch(b, args.n_per_batch)
        total_raw += len(jobs)

        print(f"[batch {b}] Raw jobs: {len(jobs)}")
        compressed = process_batch(store, jobs, curator)
        total_compressed += len(compressed)
        print(f"[batch {b}] Compressed jobs (this batch): {len(compressed)}")
        print()

    print(f"[summary] Total raw jobs across batches      : {total_raw}")
    print(f"[summary] Total compressed jobs (per-batch) : {total_compressed}")
    print()

    # -----------------------------------------------------------------------
    # Optional global re-selection across all stored compressed jobs
    # -----------------------------------------------------------------------
    if args.no_global_reselect:
        print("[info] Skipping global re-selection (as requested).")
        return

    print("[info] Running global re-selection across ALL compressed jobs in store...")
    global_reselect_in_store(store, ratio=args.ratio, alpha=args.alpha)
    print("[info] Global re-selection completed.")
    print("       The store now contains a globally compressed + diversified set.")


if __name__ == "__main__":
    main()
