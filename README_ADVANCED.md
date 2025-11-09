# 🧠 JobCurator – Incremental Pipelines & Data Model

This page is an **advanced user guide** for `jobcurator`.

- For installation, basic usage, and backend comparison → see `README.md`.
- Here we focus on:
  - how the **objects** (`Job`, `Category`, etc.) fit together, and  
  - how to build an **incremental deduplication pipeline** over time (batches, SQL or local files).

---

## 1. Core Data Model (Conceptual)

### `Job`

A `Job` is the main unit you pass to `JobCurator`.

Typical fields:

- **Required**
  - `id: str` – unique identifier for the job

- **Content / context (optional but recommended)**
  - `title: str | None`
  - `text: str | None`
  - `categories: dict[str, list[Category]]`
  - `location: Location3DField | None`
  - `salary: SalaryField | None`
  - `company: str | None`
  - `contract_type: str | None`
  - `source: str | None`
  - `created_at: datetime | None`

- **Computed by `JobCurator`**
  - `length_score: float`
  - `completion_score_val: float`
  - `quality: float`          ← used for ranking / selection
  - `exact_hash: int`
  - `signature: int`          ← 128-bit composite hash, used for diversity (Hamming)

You create `Job` objects, then `JobCurator` enriches them with quality + hash metadata.

---

### `Category`

Represents hierarchical category information (multi-level taxonomy).

Fields:

- `id: str`
- `label: str`
- `level: int` – depth (0 = root)
- `parent_id: str | None`
- `level_path: list[str]` – e.g. `["Engineering", "Software", "Backend"]`

A job can have multiple category dimensions at once:

```python
job.categories = {
    "job_function": [
        Category(
            id="backend",
            label="Backend",
            level=2,
            parent_id="software",
            level_path=["Engineering", "Software", "Backend"],
        )
    ],
    "industry": [
        Category(
            id="saas",
            label="SaaS",
            level=1,
            parent_id="tech",
            level_path=["Technology", "SaaS"],
        )
    ],
}
```

These categories are used in the hashing process (meta-hash, MinHash, FAISS vectors).

---

### `Location3DField`

A location with 3D coordinates for proper geo distance:

* **Inputs**

  * `lat: float`  – latitude (degrees)
  * `lon: float`  – longitude (degrees)
  * `alt_m: float` – altitude in meters (optional)
  * `city: str | None`
  * `country_code: str | None`

* **Internal**

  * `x, y, z: float` – Earth-centered 3D coordinates (computed once)

`JobCurator` uses these coordinates to:

* avoid merging jobs that are too far apart,
* or to add location into feature vectors (e.g. FAISS backend).

---

### `SalaryField`

Structured salary information:

* `min_value: float | None`
* `max_value: float | None`
* `currency: str` – e.g. `"EUR"`, `"USD"`
* `period: str` – e.g. `"year"`, `"month"`

Salary can be bucketized and used in the hashing / meta-hash steps.

---

## 2. What JobCurator Does (In Memory)

`JobCurator` works entirely in memory on a list of `Job` objects:

1. **Scores quality**
   Combines length, completion, freshness, etc. into a single `quality` score per job.

2. **Computes hashes & signatures**

   * exact hash (to remove strict duplicates)
   * SimHash / MinHash / FAISS vector signatures depending on backend

3. **Clusters similar jobs**
   Using LSH, MinHash, FAISS, etc.

4. **Selects a subset (compression)**
   Respects:

   * `ratio` (e.g. keep 50%),
   * `alpha` (quality vs diversity trade-off),
   * cluster-level pooling (`max_per_cluster_in_pool`).

Canonical call:

```python
compressed_jobs = curator.dedupe_and_compress(jobs)
```

---

## 3. Incremental Pipelines: Key Concepts

In many real-world setups:

* jobs arrive in **batches** (e.g. `jobs1`, `jobs2`, …),
* you want to dedupe against **past batches**,
* and you want to maintain a **global compressed view** over time.

The incremental strategy uses three additional pieces:

1. **CuckooFilter**

   * A compact “seen set” of exact hashes.
   * Lets you check “have we already seen something that looks exactly like this job?”
   * Updated each time you process a batch.

2. **StoreDB interface**

   * Abstracts where compressed jobs and the CuckooFilter are stored.
   * There are ready-made implementations for:

     * SQL (`SqlStoreDB`)
     * local files (`LocalFileStoreDB`)

3. **Helpers for incremental flows**

   * `process_batch(store, jobs, curator)`
   * `global_reselect_in_store(store, ratio, alpha)`

These live under:

```python
from jobcurator.storage import (
    StoreDB,
    SqlStoreDB,
    LocalFileStoreDB,
    process_batch,
    global_reselect_in_store,
)
```

---

## 4. StoreDB: The Storage Abstraction

Conceptually, `StoreDB` is:

> “Anything that can store compressed jobs + one global CuckooFilter,
> and can list minimal per-job metadata when we want to rebalance.”

It needs to support:

* **CuckooFilter state**

  * `load_or_create_cuckoo(capacity) -> CuckooFilter`
  * `save_cuckoo(cf) -> None`

* **Compressed jobs**

  * `insert_compressed_jobs(compressed_jobs, backend)`
  * `load_all_light_jobs() -> list[LightJob]`
  * `overwrite_with_selected(selected_ids)`

The algorithmic core only needs, for each job:

* `id`
* `quality`
* `signature`

Everything else (title, text, company, location, etc.) is for your own business needs.

---

## 5. Incremental Batch Processing

### 5.1 `process_batch`

Used for each new batch of raw jobs:

```python
from jobcurator.storage import process_batch

compressed_jobsN = process_batch(
    store=my_store_db,  # SqlStoreDB or LocalFileStoreDB
    jobs=jobsN,
    curator=my_curator,
)
```

What happens:

1. Load or create a **global CuckooFilter** from the store.
2. Run `curator.dedupe_and_compress(jobsN, approx_seen_filter=cuckoo_filter)`:

   * dedup + compress **inside** the batch,
   * drop jobs that seem already seen (exact hash), based on previous batches.
3. Insert the resulting `compressed_jobsN` into storage.
4. Save the updated CuckooFilter back to the store.

Result:

* You can process `jobs1`, `jobs2`, `jobs3`, … in order,
* without ever having to reload all previous compressed jobs into memory,
* while still avoiding re-inserted duplicates across batches.

---

## 6. Global Rebalancing (Quality + Diversity)

Over time, you may want to:

* keep only a certain fraction of all compressed jobs (e.g. 50%),
* while preserving **diversity** and **quality** across *all* batches.

You can use:

```python
from jobcurator.storage import global_reselect_in_store

global_reselect_in_store(
    store=my_store_db,
    ratio=0.5,   # keep ~50% of stored compressed jobs
    alpha=0.6,   # trade-off between quality and diversity
)
```

What happens:

1. `store.load_all_light_jobs()` returns a list of light objects (id, quality, signature).
2. A global greedy selection is run:

   * same quality + diversity logic as in `JobCurator`.
3. `store.overwrite_with_selected(selected_ids)` keeps only those jobs in storage.

This gives you a **globally consistent compressed set** over multiple batches:

* same similarity notion (Hamming on `signature`),
* same `alpha` trade-off,
* but applied to **everything you have stored**, not just one batch.

---

## 7. Concrete Examples

### 7.1 Incremental Pipeline with SQL

```python
from jobcurator import JobCurator
from jobcurator.storage import SqlStoreDB, process_batch, global_reselect_in_store
import psycopg2

# 1) Connect to your database
conn = psycopg2.connect("dbname=... user=... password=... host=...")

# 2) Choose a storage implementation
store = SqlStoreDB(conn)

# 3) Configure JobCurator
curator = JobCurator(
    backend="default_hash",
    ratio=0.5,
    alpha=0.6,
    max_per_cluster_in_pool=3,
    d_sim_threshold=20,
    max_cluster_distance_km=50.0,
    use_multiprobe=True,
)

# 4) Process batches incrementally
compressed_jobs1 = process_batch(store, jobs1, curator)
compressed_jobs2 = process_batch(store, jobs2, curator)
compressed_jobs3 = process_batch(store, jobs3, curator)
# ...

# 5) Periodically rebalance globally
global_reselect_in_store(store, ratio=0.5, alpha=0.6)
```

Test with **SQL storage** (Postgres):
```python
python3 test_incremental.py \
  --backend default_hash \
  --ratio 0.5 \
  --alpha 0.6 \
  --storage sql \
  --dsn "dbname=mydb user=myuser password=mypass host=localhost port=5432" \  
  --batches 3 \
  --n-per-batch 30 \
  # --no-global-reselect   # optional
```

### 7.2 Incremental Pipeline with Local Files

```python
from jobcurator import JobCurator
from jobcurator.storage import LocalFileStoreDB, process_batch, global_reselect_in_store

# 1) Use the local file-based store
store = LocalFileStoreDB()  # defaults to ./data/compressed_jobs.jsonl, ./data/cuckoo_filter.pkl

# 2) Configure JobCurator as usual
curator = JobCurator(
    backend="default_hash",
    ratio=0.5,
    alpha=0.6,
    max_per_cluster_in_pool=3,
    d_sim_threshold=20,
    max_cluster_distance_km=50.0,
    use_multiprobe=True,
)

# 3) Process incoming batches
compressed_jobs1 = process_batch(store, jobs1, curator)
compressed_jobs2 = process_batch(store, jobs2, curator)

# 4) Periodic global cleanup / rebalancing
global_reselect_in_store(store, ratio=0.5, alpha=0.6)
```
Test with **local storage**:
```python
python3 test_incremental.py \
  --backend default_hash \
  --ratio 0.5 \
  --alpha 0.6 \
  --storage local \
  --dsn "" \
  --batches 3 \
  --n-per-batch 20 \
  --clear-local \
  # --no-global-reselect   # (optional) add this flag if you want to skip final global rebalancing
```

---

## 8. When to Use the Incremental Approach

You should consider the incremental pipeline if:

* Jobs arrive **continuously** (daily/hourly feeds).
* You want to **avoid reprocessing** or reloading all historical jobs.
* You need a **bounded** global set of compressed jobs with:

  * controlled compression ratio,
  * stable diversity,
  * and consistent quality scoring.

If you just want to dedupe one big static snapshot once, you can call:

```python
compressed_jobs = curator.dedupe_and_compress(jobs)
```

directly and ignore the incremental API.

For long-running production feeds, the combination of:

* `JobCurator` (in-memory dedup/compression),
* `CuckooFilter` (seen set),
* `StoreDB` (persistence),
* `process_batch` + `global_reselect_in_store`

gives you a clean, reusable pattern to scale over time.