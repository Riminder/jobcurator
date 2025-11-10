<div style="width:260px; height:100px; overflow:hidden; border-radius:8px;">
  <img src="./logo.png"
       style="width:100%; height:auto; object-fit:cover; object-position:center 50%;" />
</div>



# 🤗 Welcome to the jobcurator library
`jobcurator` is an open-source Machine Learning library to clean, normalize, structure, compress, and sample large datasets & feeds of job offers.

## 📚 Table of Contents

- [About the jobcurator library](#-about-the-jobcurator-library)
- [Motivation: Why jobcurator?](#-motivation-why-jobcurator)
  - [Contact](#-contact)
  - [Available features](#-available-features)
  - [Backends comparison](#-backends-comparison)
  - [TODO](#-todo)
- [Repository structure](#-repository-structure)
- [Installation](#-installation)
- [Testing code](#-testing-code)
- [Public API](#-public-api)
  - [Basic example usage](#basic-example-usage)
  - [Incremental JobCurator](#incremental-jobcurator)
- [Core Concepts](#-core-concepts)
- [Advanced (High Level)](#-advanced-high-level)
  - [Incremental Jobcurator Approach](#incremental-jobcurator-approach)
- [Contributing](#-contributing)
  - [Getting Started](#-getting-started)
  - [Reporting Bugs](#-reporting-bugs)
  - [Suggesting Features](#-suggesting-features)
  - [Tests & Quality](#-tests--quality)
  - [Code Style & Guidelines](#-code-style--guidelines)
  - [Public API & Backward Compatibility](#-public-api--backward-compatibility)
  - [Pull Requests](#-pull-requests)

## 💡 Motivation: Why jobcurator?
This library exists because job feeds in the aggregator world and the programmatic job distribution world are extremely noisy, redundant, low quality, and not normalized. jobcurator was created to take raw job firehose feeds and turn them into high-quality, diversified and deduplicated structured job data — before they hit searching, ranking, matching or bidding engines.

### 📬 Contact

For questions, ideas, or coordination around larger changes:

**Primary maintainer**
📧 [mouhidine.seiv@hrflow.ai](mailto:mouhidine.seiv@hrflow.ai)

### ✨ Available features:
- `dedupe_and_compress` is a Hash-based job deduplication and compression fuction with  **quality** and **diversity preservation**. It takes a list of
  structured job objects and:
  - Deduplicates using **hashing** (exact hash + SimHash + LSH / sklearn / FAISS)
  - Scores jobs by **length & completion** (and optional freshness/source)
  - Preserves **variance** by keeping jobs that are **far apart** in hash/signature space
  - Respects a global **compression ratio** (e.g. keep 40% of jobs) while prioritizing quality and diversity
  - Greedy diversity selection: optional post-selection pass that recomputes diversity scores on the final set using robust quantile scaling and 
    (optionally) soft-min.
- Print helpers:
  - `print_compression_summary()` → compact, ASCII table with length/quality stats.
  - `print_jobs_summary()` → preview of top-N selected jobs with Quality / Diversity / Selection.
- `process_batch` Supports **incremental pipelines** (see the [Advanced documentation](README_ADVANCED.md):
  - process batches over time (jobs₁, jobs₂, …)
  - uses a CuckooFilter + pluggable storage (SQL or local files) to avoid re-ingesting old jobs
    

### ⚖️ Backends comparison

| Capability / Feature        | `default_hash`                                  | `minhash_hash`                                  | `sklearn_hash`                                       | `faiss_hash`                                              |
|----------------------------|--------------------------------------------------|-------------------------------------------------|------------------------------------------------------|-----------------------------------------------------------|
| **Algorithm**         | SimHash + LSH (± Multi-probe)                   | MinHash + Jaccard LSH (± Multi-probe)           | HashingVectorizer + NearestNeighbors (cosine)        | FAISS IndexFlatL2 on composite vectors                   |
| **Similarity**        | Hamming cosine                           | Jaccard on token/shingle sets                   | Cosine distance                                      | L2 distance                                              |
| **Use case**              | General-purpose hash+geo dedupe/compression     | Robust near-dupe on noisy / reordered text      | Text-heavy feeds + sklearn-based experimentation     | Huge catalogs, low latency, NN-heavy workloads           |
| **Dependencies**             | None                                            | None                                            | `scikit-learn`+`numpy`                                        | `faiss-cpu`+`numpy`                                    |
| **Dataset size**   | ~1k → ~200k                                     | ~1k → ~200k                                      | ~1k → ~100k                                          | ~50k → 1M+                                               |
| **Speed**         | Fast on CPU for small–medium datasets           | Slower than `default_hash` (more hashing work)  | Moderate, depends on sparse ops & RAM                | Very fast for large NN queries once indexed              |
| **Explicit geo constraint**| Yes (3D distance filter in clustering)          | Yes (3D distance filter in clustering)          | No (only via tokens)                                 | No (geo only affects L2 distance)                        |
| **3D location use**        | Hard geo radius (`max_cluster_distance_km`)     | Hard geo radius (`max_cluster_distance_km`)     | Encoded as coarse x/y/z tokens                       | Normalized (x,y,z) directly in vector                    |
| **Text encoder**               | SimHash on title + text                         | Word n-gram shingles on title + text            | Text to sparse hashed vector                         | Encoded via signature bits                               |
| **Categories encoder**         | In meta-hash (part of signature)                | Added as shingles in MinHash set                | As extra tokens to HashingVectorizer                 | As “richness” feature in vector                          |
| **Salary encoder**             | Bucketed into meta-hash                         | Bucketed tokens in MinHash set                  | Via numeric/features (quality, etc.)                 | Indirect (via signature / numeric features)              |
| **Main threshold**         | `d_sim_threshold` (Hamming on SimHash)          | `jaccard_threshold` (min Jaccard)               | Internal NN radius (not exposed in API)              | `d_sim_threshold` (max L2 in FAISS space)                |
| **Multi-probe support**    | Yes (`use_multiprobe`, `max_multiprobe_flips`)  | Yes (`use_multiprobe`, `max_multiprobe_flips`)  | No                                                   | No                                                       |
| **Outlier filter**         | Optional (`use_outlier_filter` + IsolationForest)| Optional (same)                                 | Optional (same)                                      | Optional (same)                                          |

No dense text embeddings. Hash-based + classic ML only.

### 📋 TODO
- publish package to PyPI:
- add Job Parsing
- add Job dynamic Tagging with Taxonomy
- add job auto-formating & Normalization

---

## 🗂️ Repository structure
```yaml
jobcurator/
├── pyproject.toml
├── README.md                 # main intro, installation, backends, examples
├── README_ADVANCED.md        # incremental pipelines + data model (advanced doc)
├── LICENSE
├── .gitignore
├── test.py                   # unit tests for JobCurator (single batch)
├── test_incremental.py       # CLI demo for incremental pipeline (local/sql)
│
├── src/
│   └── jobcurator/
│       ├── __init__.py        # exports JobCurator, Job, Category, etc.
│       ├── models.py          # Job, Category, Location3DField, SalaryField
│       ├── curator.py         # JobCurator class (dedupe + compression)
│       ├── hash_utils.py      # SimHash, MinHash, LSH, distances, signatures
│       ├── cuckoo_filter.py   # CuckooFilter implementation
│       ├── sklearn_backend.py # sklearn_hash backend helpers (if used)
│       ├── faiss_backend.py   # faiss_hash backend helpers (if used)
│       └── storage/
│           ├── __init__.py
│           ├── base.py        # LightJob, StoreDB, process_batch, global_reselect
│           ├── sql_store.py   # SqlStoreDB (compressed_jobs + dedupe_state tables)
│           └── local_store.py # LocalFileStoreDB (JSONL + pickle)
│
└── tests/
    └── __init__.py
```

---

## 🚀 Installation
To install for local Dev:
```bash
git clone https://github.com/<your-username>/jobcurator.git
cd jobcurator
pip install -e .
```
To reinstall for local Dev:
```bash
pip uninstall -y jobcurator  # ignore error if not installed
pip install -e .
```
(coming soon) To install the package once published to PyPI:
```bash
pip install jobcurator
```
Optional extras:
```bash
pip install scikit-learn faiss-cpu
```

## 🧪 Testing code
Run main folder run test.py
```bash
# 1) Default backend (SimHash + LSH + geo), keep ~50%, preview 10 jobs (capped to len(jobs))
python3 test.py                        # n-preview-jobs=10 (capped to len(jobs)), ratio=0.5

# 2) Default backend (SimHash + LSH + geo), more aggressive compression  keep ~30%, preview 8 jobs
python3 test.py --backend default_hash --ratio 0.3 --n-preview-jobs 8

# 3) Default backend (MinHash + Jaccard LSH + geo),  keep ~50%, preview 5
python3 test.py --backend minhash_hash --ratio 0.5 --n-preview-jobs 5

# 4) sklearn backend (HashingVectorizer + NearestNeighbors), keep ~50%, preview 5
#    (requires: pip install scikit-learn)
python3 test.py --backend sklearn_hash --ratio 0.5 --n-preview-jobs 5

# 5) FAISS backend (signature bits + 3D loc + categories), keep ~40%, preview 5
#    (requires: pip install faiss-cpu)
python3 test.py --backend faiss_hash --ratio 0.4 --n-preview-jobs 5

# 6) Short option for preview:
python3 test.py -n 5 --backend default_hash --ratio 0.5
python3 test.py -n 5 --backend minhash_hash  --ratio 0.4
```

---


## 🧩 Public API a Example usage

### Import

```python
from jobcurator import JobCurator, Job, Category, SalaryField, Location3DField
from datetime import datetime
```

### Basic Jobcurator


```python

# 1) Build some jobs

jobs = [
    Job(
        id="job-1",
        title="Senior Backend Engineer",
        text="Full description...",
        categories={
            "job_function": [
                Category(
                    id="backend",
                    label="Backend",
                    level=1,
                    parent_id="eng",
                    level_path=["Engineering", "Software", "Backend"],
                )
            ]
        },
        location=Location3DField(
            lat=48.8566,
            lon=2.3522,
            alt_m=35,
            city="Paris",
            country_code="FR",
        ),
        salary=SalaryField(
            min_value=60000,
            max_value=80000,
            currency="EUR",
            period="year",
        ),
        company="HrFlow.ai",
        contract_type="Full-time",
        source="direct",
        created_at=datetime.utcnow(),
    ),
]

# 2) Choose a backend

# ======================================================
# Option 1: "default_hash"
#      SimHash + LSH (+ optional Multi-probe) + geo distance
#      (no extra dependencies)
# ======================================================

curator_default = JobCurator(
    # Global parameters (used by all backends)
    ratio=0.5,                     # keep ~50% of jobs
    alpha=0.6,                     # quality vs diversity tradeoff
    max_per_cluster_in_pool=3,     # max jobs per cluster entering pool
    backend="default_hash",
    use_outlier_filter=False,      # set True to enable IsolationForest (if sklearn installed)
    outlier_contamination=0.05,    # only used when use_outlier_filter=True

    # Backend-specific: default_hash
    d_sim_threshold=20,            # max Hamming distance on SimHash
    max_cluster_distance_km=50.0,  # max geo distance (km) within a cluster
    # jaccard_threshold is ignored by default_hash

    # Multi-probe LSH (used by default_hash + minhash_hash)
    use_multiprobe=True,
    max_multiprobe_flips=1,        # small value = light extra recall
)


# ======================================================
# Option 2: "minhash_hash"
#      MinHash + Jaccard LSH on shingles (text + cats + coarse loc + salary)
#      + optional Multi-probe + geo distance
#      (no extra dependencies)
# ======================================================

curator_minhash = JobCurator(
    # Global parameters
    ratio=0.5,
    alpha=0.6,
    max_per_cluster_in_pool=3,
    backend="minhash_hash",
    use_outlier_filter=False,
    outlier_contamination=0.05,

    # Backend-specific: minhash_hash
    jaccard_threshold=0.8,         # min Jaccard similarity between jobs in a cluster
    max_cluster_distance_km=50.0,  # geo radius (km) for clusters
    # d_sim_threshold is ignored by minhash_hash

    # Multi-probe LSH for MinHash bands
    use_multiprobe=True,
    max_multiprobe_flips=1,
)


# ======================================================
# Option 3: "sklearn_hash"
#      HashingVectorizer + NearestNeighbors (cosine radius)
#      (requires scikit-learn)
# ======================================================

# pip install scikit-learn
curator_sklearn = JobCurator(
    # Global parameters
    ratio=0.5,
    alpha=0.6,
    max_per_cluster_in_pool=3,
    backend="sklearn_hash",
    use_outlier_filter=True,       # enable IsolationForest pre-filter
    outlier_contamination=0.05,    # proportion of jobs flagged as outliers

    # Backend-specific:
    # d_sim_threshold, max_cluster_distance_km, jaccard_threshold,
    # use_multiprobe, max_multiprobe_flips are all ignored by sklearn_hash
)


# ======================================================
# Option 4: "faiss_hash"
#      FAISS on [signature bits + 3D location + category richness]
#      (requires faiss-cpu)
# ======================================================

# pip install faiss-cpu
curator_faiss = JobCurator(
    # Global parameters
    ratio=0.5,
    alpha=0.6,
    max_per_cluster_in_pool=3,
    backend="faiss_hash",
    use_outlier_filter=False,
    outlier_contamination=0.05,

    # Backend-specific: faiss_hash
    d_sim_threshold=20,            # approx max L2 distance in FAISS space
    # max_cluster_distance_km, jaccard_threshold, use_multiprobe,
    # max_multiprobe_flips are ignored by faiss_hash
)


# 3) Compute the results

compressed_jobs = curator_default.dedupe_and_compress(jobs)

print(f"{len(jobs)} → {len(compressed_jobs)} jobs kept")
for j in compressed_jobs:
    print(j.id, j.title, j.location.city, f"quality={j.quality:.3f}")

```

### Print helpers
```python
curator_default.print_compression_summary(n_preview=10, t_ms=elapsed_ms)
curator_default.print_jobs_summary(selected, n_preview=10, label="Selected")
```

#### 1) `print_compression_summary(n_preview: int = 0, t_ms: float = 0.0)`

Shows the effective keep ratio, backend, timing, and an ASCII table of length/quality stats for **all** vs **selected**.

**Example output**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  🔎 preview: 10 | 🎯 ratio: 0.40 | 🧠 backend: default_hash | ⏱️  time: 82.4 ms │
└─────────────────────────────────────────────────────────────────────────────────┘

Total jobs: 12000 | Selected: 4800 (40.0% kept)
+----------+-------+--------+--------+--------+--------+
| Dataset  | Count |  Len μ |  Len σ | Qual μ | Qual σ |
+==========+=======+========+========+========+========+
| All jobs | 12000 |  92.14 |  37.21 | 0.644  | 0.112  |
| Selected |  4800 | 106.38 |  29.07 | 0.711  | 0.087  |
+----------+-------+--------+--------+--------+--------+
```

#### 2) `print_jobs_summary(jobs, num_selected_to_show=10, label="jobs set")`

Previews the top-N by **current order** (you can pass `curator.selected_jobs`) with per-row Quality / Diversity / Selection and a short canonical hash.

**Columns**: `ID | Title | City | Quality | Diversity | Selection | Hash`

### Incremental JobCURATOR

* SQL store
```python
from jobcurator import JobCurator
from jobcurator.storage import SqlStoreDB, process_batch, global_reselect_in_store
import psycopg2

conn = psycopg2.connect("dbname=... user=... password=... host=...")
store = SqlStoreDB(conn)

curator = JobCurator(backend="default_hash", ratio=0.5, alpha=0.6)

compressed_jobs1 = process_batch(store, jobs1, curator)
compressed_jobs2 = process_batch(store, jobs2, curator)

global_reselect_in_store(store, ratio=0.5, alpha=0.6)
```
* Local store
```python
from jobcurator import JobCurator
from jobcurator.storage import LocalFileStoreDB, process_batch, global_reselect_in_store

store = LocalFileStoreDB()

curator = JobCurator(backend="default_hash", ratio=0.5, alpha=0.6)

compressed_jobs1 = process_batch(store, jobs1, curator)
compressed_jobs2 = process_batch(store, jobs2, curator)

global_reselect_in_store(store, ratio=0.5, alpha=0.6)

```

---


## 🧱 Core Concepts

### Job schema

A `Job` is a structured object with:

- `id`: unique identifier
- `title`: job title (string)
- `text`: full job description (string)
- `categories`: hierarchical taxonomy per dimension (`dict[str, list[Category]]`)
- `location`: `Location3DField` with lat/lon/alt (internally converted to 3D x,y,z)
- `salary`: optional `SalaryField`
- Optional metadata: `company`, `contract_type`, `source`, `created_at`
- Internal fields (computed by `JobCurator`):
  - `length_tokens`, `length_score`
  - `completion_score_val`
  - `quality`
  - `exact_hash` (strict dedup key)
  - `signature` (128-bit composite hash used by backends)

### Category schema

A `Category` is a hierarchical node:

- `id`: unique taxonomy ID
- `label`: human-readable label
- `level`: depth in hierarchy (0 = root)
- `parent_id`: optional parent category id
- `level_path`: full path from root (e.g. `["Engineering", "Software", "Backend"]`)

Multiple dimensions (e.g. `job_function`, `industry`, `seniority`) can coexist in `categories`:

```python
job.categories = {
    "job_function": [Category(...), ...],
    "industry": [Category(...), ...],
}
```

### Location schema with 3D coordinates

`Location3DField`:

* `lat`, `lon`: degrees
* `alt_m`: altitude in meters
* `city`, `country_code`: human-readable metadata
* `x, y, z`: **Earth-centered 3D coordinates** (computed internally by `compute_xyz()`)

These 3D coordinates are used to compute **actual distances between cities** and avoid merging jobs that are geographically too far when clustering.

### Salary schema

`SalaryField`:

* `min_value`, `max_value`: numeric range (optional)
* `currency`: e.g. `"EUR"`, `"USD"`
* `period`: `"year"`, `"month"`, `"day"`, `"hour"`

Salary is used both in completion/quality scoring and in the exact/meta hashes (bucketed).

### CuckooFilter (approximate “seen before”)

The library includes a simple `CuckooFilter`:

* Used to **avoid re-processing jobs** that have already been seen across runs or batches.
* Works on `exact_hash` values:

  * If `exact_hash` is *probably* present → the job is skipped.
  * Otherwise → `add(exact_hash)` and the job is processed.
* Integrated via `seen_filter` parameter in:

```python
compressed = curator.dedupe_and_compress(jobs, seen_filter=seen_filter)
```

Where `seen_filter` is typically an instance of `jobcurator.CuckooFilter`.


### JobCurator parameters

```python
JobCurator(
    ratio: float = 1.0,              # default compression ratio
    alpha: float = 0.6,              # quality vs diversity weight
    max_per_cluster_in_pool: int = 3,
    d_sim_threshold: int = 20,       # SimHash Hamming threshold for clustering
    max_cluster_distance_km: float = 50.0,  # max distance between cities in same cluster
)
```

* `ratio = 1.0` → keep all jobs
* `ratio = 0.5` → keep ~50% of jobs (highest quality + diversity)
* `alpha` closer to 1 → prioritize quality; closer to 0 → prioritize diversity

More params: 
| Param                     | Where                                       | Type          | Default          | Description                                                                                        |
| ------------------------- | ------------------------------------------- | ------------- | ---------------- | -------------------------------------------------------------------------------------------------- |
| `ratio`                   | `JobCurator(...)` / `dedupe_and_compress()` | float ∈ [0,1] | `1.0`            | Target keep ratio after dedupe + selection.                                                        |
| `alpha`                   | `JobCurator(...)`                           | float ∈ [0,1] | `0.6`            | Trade-off in `selection_score`.                                                                    |
| `greedy_diversity`        | `dedupe_and_compress()`                     | bool          | `False`          | Recompute diversity on the final set with robust scaling (recommended for quality-sensitive runs). |
| `max_per_cluster_in_pool` | `JobCurator(...)`                           | int           | `3`              | Cap per cluster before global selection.                                                           |
| `backend`                 | `JobCurator(...)`                           | literal       | `"default_hash"` | Hashing/clustering strategy.                                                                       |
| `use_outlier_filter`      | `JobCurator(...)`                           | bool          | `False`          | Optional IsolationForest pre-filter.                                                               |
| `d_sim_threshold`         | `JobCurator(...)`                           | int           | `20`             | Hamming/L2 threshold (backend-specific).                                                           |
| `jaccard_threshold`       | `JobCurator(...)`                           | float         | `0.8`            | MinHash LSH threshold.                                                                             |


---

### JobCurator Backends

You choose the dedup clustering strategy via:

```python
JobCurator(backend=...)
```

Available backends:

* **`default_hash`**

  * SimHash + **LSH** (with optional **Multi-probe LSH**) on text.
  * Geo-aware: enforces a maximum **3D distance** between jobs in the same cluster.
  * Uses categories and salary in the composite signature.
  * Pure Python, no external dependencies.

* **`minhash_hash`**

  * MinHash over **shingles** built from:

    * text (word n-grams),
    * categories,
    * coarse location bucket,
    * salary bucket.
  * Jaccard similarity + LSH (banding) + optional Multi-probe.
  * Optional geo distance filter (same `max_cluster_distance_km` as `default_hash`).
  * Pure Python, no external deps.

* **`sklearn_hash`**

  * Uses `scikit-learn`:

    * `HashingVectorizer` on text + encoded 3D location + category tokens.
    * `NearestNeighbors` (cosine radius) to form clusters.
  * Compatible with `IsolationForest` for outlier filtering.
  * Requires: `pip install scikit-learn`.

* **`faiss_hash`**

  * Uses **FAISS** (`IndexFlatL2`) on composite vectors:

    [
    [\text{signature bits} + \text{normalized (x,y,z)} + \text{category richness}]
    ]

  * Designed for large-scale catalogs (fast nearest-neighbor search).

  * Requires: `pip install faiss-cpu`.

---


## ⚙️ How It Works (High Level)

### 1. **Preprocessing & scoring**

   - Compute token length of `title + text` → normalize to `length_score ∈ [0,1]` using p10/p90 percentiles.
   - Compute `completion_score` based on presence of key fields: title, text, location, salary, categories, company, contract_type.
   - Compute `freshness_score` (based on `created_at`) and `source_quality` (e.g. `direct` vs `job_board`).
   - Combine into a single quality score:

     ```text
     quality(j) = 0.3 * length_score
                + 0.4 * completion_score
                + 0.2 * freshness_score
                + 0.1 * source_quality
     ```

### 2. **Approximate “seen before” filter (CuckooFilter)**

   - Optionally use an internal **CuckooFilter** to track jobs across runs or batches.
   - For each job:
     - If `exact_hash(j)` is *probably* already in the filter → skip.
     - Otherwise → `add(exact_hash(j))` to the filter and keep the job.
   - This avoids re-processing jobs that have already been seen.

### 3. **Exact hash dedup (strict duplicates)**

   - Build a canonical string from:
     - normalized title,
     - flattened categories,
     - coarse location bucket,
     - salary bucket,
     - normalized full text.
   - Hash with `blake2b` into a 64-bit `exact_hash`.
   - Keep only one job per `exact_hash` (hard dedup).

### 4. **Composite signature (no embeddings)**

   For each job, build a 128-bit `signature`:

   - 64-bit **SimHash** on normalized `title + text`.
   - 64-bit **meta feature-hash** on categories, location (city, country, coords), salary.
   - Concatenate:

     ```text
     signature = (simhash << 64) | meta_bits
     ```

   This signature is used by the different backends.

### 5. **Clustering (backend-dependent)**

   Depending on `backend`:

   #### a. `backend="default_hash"` – SimHash + Multi-probe LSH + geo

   - Take the **SimHash** part of the signature (64 bits).
   - Split into bands → Locality Sensitive Hashing.
   - **Multi-probe LSH**:
     - For each band, also explore neighboring band keys by flipping a few bits (configurable `max_multiprobe_flips`).
     - This increases recall for near-duplicates that differ in a few bits.
   - Candidate pairs are accepted into the same cluster if:
     - Hamming distance on SimHash ≤ `d_sim_threshold`
     - 3D geo distance between locations ≤ `max_cluster_distance_km`
   - Use union–find to build clusters.

   #### b. `backend="sklearn_hash"` – HashingVectorizer + NearestNeighbors

   - Build text features with `HashingVectorizer` over:
     - title + text,
     - coarse 3D location tokens (x,y,z),
     - flattened category tokens.
   - Use `NearestNeighbors` (cosine radius) to connect jobs that are close in this hashed feature space.
   - Connected components form clusters.

   #### c. `backend="faiss_hash"` – FAISS on signature + 3D loc + categories

   - For each job, build a numeric vector:

     ```text
     [signature_bits (0/1), normalized (x,y,z), category_richness]
     ```

   - Index all vectors in **FAISS** (`IndexFlatL2`).
   - For each job, query its nearest neighbors; pairs with distance ≤ `d_sim_threshold` are connected.
   - Connected components become clusters.

### 6. **Intra-cluster ranking**

   - Inside each cluster, sort jobs by `quality` (descending).
   - For each cluster, keep only the top `max_per_cluster_in_pool` jobs as candidates.

### 7. **Global compression with diversity**

   - Merge all per-cluster candidates into a global pool and deduplicate by `cannonical_id` based the job `id`, `reference`, `company` .
   - Sort the pool by `quality` (descending).
   - Greedy selection:

     1. Start with the highest-quality job.
     2. Repeatedly pick the job `j` in the pool that maximizes:

        ```text
        selection_score(j) =
            alpha * quality(j)
          + (1 - alpha) * normalized_diversity_distance(j)
        ```
        where `normalized_diversity_distance(j)` depends on the backend:

        - `default_hash`   → ***normalized Hamming distance*** on 64-bit composite signature
        - `minhash_hash`   → ***1 - Jaccard estimate*** from MinHash signatures
        - `sklearn_hash`   → ***cosine distance*** on HashingVectorizer vector
        - `faiss_hash`     → ***L2 distance*** on FAISS composite vector

   - Stop when you’ve selected:

     ```text
     K = ceil(ratio * N_original)
     ```

   Result: you keep **fewer, higher-quality, and more diverse** jobs, while avoiding duplicates (strict + near-duplicates), and optionally skipping already-seen jobs via **CuckooFilter**.

---

 ## 🛠️ Advanced (High Level)

### 1. **Diversity–aware selection**

During compression we score each candidate:

```
selection_score = α * quality + (1 - α) * diversity
```

* `quality` is computed per job (length, completion, etc.).
* `diversity` is backend-aware:

  * `default_hash`: normalized Hamming on 64-bit SimHash/composite signature
  * `minhash_hash`: 1 − Jaccard (from MinHash)
  * `sklearn_hash`: cosine distance
  * `faiss_hash`: cosine distance on composite vectors

#### a. Greedy pass (fast)

While selecting, we compute each job’s **min distance to any already-selected** item, then **robust-scale** distances with quantiles *(q_lo=0.10, q_hi=0.90)* and **label smoothing** *(ε=0.02)* to avoid hard 0/1:

```
z = clamp01( (d - q10) / (q90 - q10 + 1e-6) )
diversity = ε + (1 - 2ε) * z
```

> Seed item gets `diversity_score = 1.0` (helps robust scaling).

#### b. Greedy diversity re-compute (optional, slower, more faithful)

If you pass `greedy_diversity=True`, we run `recompute_diversity_scores()` on the **final** selected set:

* Compute pairwise distance matrix `[0,1]`.
* Per item, compute either:

  * **k-NN mean** of the *k* closest distances *(default k=3)*, or
  * **soft-min** with temperature τ *(smaller τ → closer to hard min)*.
* Apply **robust quantile scaling** (q_lo, q_hi), then a **leave-one-out (LOO)** contribution:

  * `final_div = 0.5 * local_scaled + 0.5 * loo_scaled`
* Rebuild `selection_score` with same α.

**Knobs** (in `recompute_diversity_scores`):

* `k_nn=3`, `q_lo=0.10`, `q_hi=0.90`, `tau=0.15`, `label_eps=0.02`, `use_softmin=False`.

```python
selected = curator.dedupe_and_compress(
    jobs,
    ratio=0.4,                 # optional override
    greedy_diversity=True,     # ← new
    seen_filter=my_cuckoo,     # optional Bloom/Cuckoo/set-like
)

# Optional recalibration if you want to run it manually:
curator.recompute_diversity_scores(
    selected_jobs=selected,
    alpha=curator.alpha,
    distance_fn=curator._diversity_distance,
    k_nn=3,
    q_lo=0.10, q_hi=0.90,
    tau=0.15,
    label_eps=0.02,
    use_softmin=False,
)
```

 ### **Incremental Jobcurator Approach**

Problem:
You often receive **batches of jobs over time** (`jobs1`, `jobs2`, …) and want to:

* Avoid re-ingesting duplicates/near-duplicates from past batches.
* Maintain a **global compressed set** across all batches with a fixed or target ratio.
* Not reload all previous jobs into memory each time.

The solution is:

1. Use a global **CuckooFilter** to remember “seen” jobs (by exact hash).
2. Use a pluggable **`StoreDB`** to store compressed jobs + CuckooFilter state.
3. Use:

   * `process_batch(StoreDB, jobs, JobCurator)` for incremental batches
   * `global_reselect_in_store(StoreDB, ratio, alpha)` for global rebalancing

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

For more details, see the [Advanced documentation](README_ADVANCED.md).

---

## 🤝 Contributing

First off, thank you for taking the time to contribute! 🎉
This project aims to provide a robust, hash-based job deduplication & compression engine, and your help is highly appreciated.

### 🧭 Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:

   ```bash
   git clone https://github.com/<your-username>/jobcurator.git
   cd jobcurator
   ```
3. Install in editable / dev mode:

   ```bash
   pip install -e .
   ```
4. Create a feature branch:

   ```bash
   git checkout -b feat/my-feature
   ```

---

### 🐛 Reporting Bugs

Please use GitHub Issues and include:

* `jobcurator` version
* Python version
* OS
* Minimal reproducible example (code + data schema, no sensitive data)
* Expected vs actual behavior

For security-related or sensitive issues, you can also contact the maintainer directly:

**📧 [mouhidine.seiv@hrflow.ai](mailto:mouhidine.seiv@hrflow.ai)**

---

### 🌱 Suggesting Features

When opening a feature request:

* Clearly describe the **problem** you want to solve.
* Explain how it fits into `jobcurator`’s scope:

  * hash-based dedupe
  * compression ratio
  * quality scoring
  * diversity / variance preservation
* Optionally include:

  * Proposed API shape (function/class signature)
  * Example usage snippet
  * Notes on performance / complexity if relevant

---

### 🧪 Tests & Quality

Before submitting a PR:

1. Add or update tests (e.g. under `tests/`):

   * Edge cases: empty input, single job, all duplicates, all unique.
   * Typical cases: mixed locations, mixed sources, various compression ratios.
2. Run the test suite:

   ```bash
   pytest
   ```
3. Ensure all tests pass.

If your change touches deduplication, scoring, clustering, or selection logic, please add specific tests to cover the change and avoid regressions.

---

### 🧹 Code Style & Guidelines

* Target **Python 3.9+**.
* Use **type hints** for functions, methods, and dataclasses.
* Keep modules focused:

  * `models.py` → schema & dataclasses
  * `hash_utils.py` → hashing, signatures, clustering, quality scores
  * `curator.py` → `JobCurator` orchestration / public API
* Prefer:

  * `black` for formatting
  * `ruff` or `flake8` for linting

Naming conventions:

* Classes: `PascalCase` (`JobCurator`, `Location3DField`)
* Functions: `snake_case` (`build_exact_hash`, `geo_distance_km`)
* Constants: `UPPER_SNAKE_CASE`

Avoid introducing heavy dependencies—this library is intentionally lightweight and focused on hashing + simple math.

---

### 📦 Public API & Backward Compatibility

The main public API consists of:

* `jobcurator.JobCurator`
* `jobcurator.Job`
* `jobcurator.Category`
* `jobcurator.SalaryField`
* `jobcurator.Location3DField`

When changing their behavior or signatures:

* Consider backward compatibility.
* Document changes in:

  * PR description
  * `README.md` (if user-visible behavior changes)
* For breaking changes, propose a clear migration path and rationale.

---

### 📥 Pull Requests

1. Make sure your branch is up to date with `main`:

   ```bash
   git fetch origin
   git rebase origin/main
   ```
2. Push your branch to your fork:

   ```bash
   git push origin feat/my-feature
   ```
3. Open a PR and include:

   * A clear title (e.g. `Add salary band weighting to quality scoring`)
   * Description of what changed and **why**
   * Any performance considerations
   * Tests added or updated

PRs that are **small, focused, and well-tested** are more likely to be reviewed and merged quickly.

---
