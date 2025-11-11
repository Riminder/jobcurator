from __future__ import annotations

import pickle
from typing import List, Set

from jobcurator import CuckooFilter, Job

from .base import LightJob


class SqlStoreDB:
    """
    SQL-backed StoreDB.

    Expects tables:
      - compressed_jobs(
            id TEXT PRIMARY KEY,
            quality DOUBLE PRECISION NOT NULL,
            signature TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            backend TEXT NOT NULL,
            -- any optional business fields (title, text, company, location_*, ...)
        )
      - dedupe_state(
            id TEXT PRIMARY KEY,
            cuckoo_blob BYTEA NOT NULL,
            updated_at TIMESTAMP NOT NULL
        )

    Only id, quality, signature are required for dedup/global selection.
    """

    def __init__(self, conn, filter_id: str = "global"):
        self.conn = conn
        self.filter_id = filter_id

    # ----- CuckooFilter -----

    def load_or_create_cuckoo(self, capacity: int = 1_000_000) -> CuckooFilter:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT cuckoo_blob FROM dedupe_state WHERE id = %s",
                (self.filter_id,),
            )
            row = cur.fetchone()
            if row is None:
                cf = CuckooFilter(capacity=capacity)
                blob = pickle.dumps(cf, protocol=pickle.HIGHEST_PROTOCOL)
                cur.execute(
                    """
                    INSERT INTO dedupe_state (id, cuckoo_blob, updated_at)
                    VALUES (%s, %s, NOW())
                    """,
                    (self.filter_id, blob),
                )
                self.conn.commit()
                return cf
            else:
                return pickle.loads(row[0])

    def save_cuckoo(self, cf: CuckooFilter) -> None:
        blob = pickle.dumps(cf, protocol=pickle.HIGHEST_PROTOCOL)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dedupe_state
                SET cuckoo_blob = %s, updated_at = NOW()
                WHERE id = %s
                """,
                (blob, self.filter_id),
            )
        self.conn.commit()

    # ----- Compressed jobs -----

    def insert_compressed_jobs(self, compressed_jobs: List[Job], backend: str) -> None:
        """
        For the algorithm, only id, quality, signature are required.
        Other fields (title, text, company, location_*, ...) are optional.
        """
        with self.conn.cursor() as cur:
            for j in compressed_jobs:
                sig_str = str(j.signature)  # 128-bit int → string
                city = getattr(j.location, "city", None) if j.location else None
                country = (
                    getattr(j.location, "country_code", None) if j.location else None
                )

                cur.execute(
                    """
                    INSERT INTO compressed_jobs (
                        id, title, text, company,
                        location_city, location_country,
                        quality, signature, created_at, backend
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s)
                    ON CONFLICT (id) DO UPDATE
                        SET quality = EXCLUDED.quality,
                            signature = EXCLUDED.signature,
                            title = EXCLUDED.title,
                            text = EXCLUDED.text,
                            company = EXCLUDED.company,
                            location_city = EXCLUDED.location_city,
                            location_country = EXCLUDED.location_country,
                            created_at = EXCLUDED.created_at,
                            backend = EXCLUDED.backend
                    """,
                    (
                        j.id,
                        j.title,
                        j.text,
                        j.company,
                        city,
                        country,
                        float(j.quality),
                        sig_str,
                        backend,
                    ),
                )
        self.conn.commit()

    def load_all_light_jobs(self) -> List[LightJob]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT id, quality, signature FROM compressed_jobs")
            rows = cur.fetchall()

        jobs_meta: List[LightJob] = []
        for job_id, quality, sig_str in rows:
            jobs_meta.append(
                LightJob(
                    id=job_id,
                    quality=float(quality),
                    signature=int(sig_str),
                )
            )
        return jobs_meta

    def overwrite_with_selected(self, selected_ids: Set[str]) -> None:
        with self.conn.cursor() as cur:
            if not selected_ids:
                cur.execute("DELETE FROM compressed_jobs")
            else:
                id_tuple = tuple(selected_ids)
                cur.execute(
                    "DELETE FROM compressed_jobs WHERE id NOT IN %s",
                    (id_tuple,),
                )
        self.conn.commit()
