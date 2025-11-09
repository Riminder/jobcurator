from __future__ import annotations

import os
import json
import pickle
from datetime import datetime
from typing import List, Set

from jobcurator import CuckooFilter, Job
from .base import LightJob

DATA_DIR = "data"
DEFAULT_JOBS_PATH = os.path.join(DATA_DIR, "compressed_jobs.jsonl")
DEFAULT_CUCKOO_PATH = os.path.join(DATA_DIR, "cuckoo_filter.pkl")


def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


class LocalFileStoreDB:
    """
    Local-file StoreDB.

    Files:
      - data/compressed_jobs.jsonl : all compressed jobs (one JSON per line)
      - data/cuckoo_filter.pkl     : pickled CuckooFilter

    For the algorithm, only id, quality, signature are required.
    """

    def __init__(
        self,
        jobs_path: str = DEFAULT_JOBS_PATH,
        cuckoo_path: str = DEFAULT_CUCKOO_PATH,
    ):
        self.jobs_path = jobs_path
        self.cuckoo_path = cuckoo_path

    # ----- CuckooFilter -----

    def load_or_create_cuckoo(self, capacity: int = 1_000_000) -> CuckooFilter:
        _ensure_data_dir()
        if not os.path.exists(self.cuckoo_path):
            cf = CuckooFilter(capacity=capacity)
            with open(self.cuckoo_path, "wb") as f:
                f.write(pickle.dumps(cf, protocol=pickle.HIGHEST_PROTOCOL))
            return cf

        with open(self.cuckoo_path, "rb") as f:
            data = f.read()
        return pickle.loads(data)

    def save_cuckoo(self, cf: CuckooFilter) -> None:
        _ensure_data_dir()
        with open(self.cuckoo_path, "wb") as f:
            f.write(pickle.dumps(cf, protocol=pickle.HIGHEST_PROTOCOL))

    # ----- Compressed jobs -----

    def _job_to_record(self, j: Job, backend: str) -> dict:
        """
        Convert a Job to JSON record.

        Required for dedup/compression:
          - id
          - quality
          - signature

        Others are optional metadata.
        """
        loc = j.location
        city = getattr(loc, "city", None) if loc else None
        country = getattr(loc, "country_code", None) if loc else None

        return {
            "id": j.id,                         # REQUIRED
            "title": j.title,                   # OPTIONAL
            "text": j.text,                     # OPTIONAL
            "company": j.company,               # OPTIONAL
            "location_city": city,              # OPTIONAL
            "location_country": country,        # OPTIONAL

            "quality": float(j.quality),        # REQUIRED
            "signature": str(j.signature),      # REQUIRED
            "created_at": (
                (j.created_at or datetime.utcnow()).isoformat() + "Z"
            ),                                  # RECOMMENDED
            "backend": backend,                 # RECOMMENDED
        }

    def insert_compressed_jobs(self, compressed_jobs: List[Job], backend: str) -> None:
        _ensure_data_dir()
        with open(self.jobs_path, "a", encoding="utf-8") as f:
            for j in compressed_jobs:
                record = self._job_to_record(j, backend=backend)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_all_light_jobs(self) -> List[LightJob]:
        if not os.path.exists(self.jobs_path):
            return []

        jobs_meta: List[LightJob] = []
        with open(self.jobs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                job_id = record["id"]
                quality = float(record["quality"])
                sig_int = int(record["signature"])
                jobs_meta.append(
                    LightJob(
                        id=job_id,
                        quality=quality,
                        signature=sig_int,
                    )
                )
        return jobs_meta

    def overwrite_with_selected(self, selected_ids: Set[str]) -> None:
        """
        Rewrite compressed_jobs.jsonl with only selected IDs.
        """
        if not os.path.exists(self.jobs_path):
            return

        _ensure_data_dir()

        kept_records = []
        with open(self.jobs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record["id"] in selected_ids:
                    kept_records.append(record)

        with open(self.jobs_path, "w", encoding="utf-8") as f:
            for rec in kept_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
