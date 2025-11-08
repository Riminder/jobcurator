<div style="width:260px; height:100px; overflow:hidden; border-radius:8px;">
  <img src="./logo.png"
       style="width:100%; height:auto; object-fit:cover; object-position:center 50%;" />
</div>

# jobcurator
Open source Machine Learning library to clean, normalize, structure, compress, and sample large datasets & feeds of job offers.

### Available features:
- Hash-based job deduplication and compression with  **quality** and **diversity preservation**.
`jobcurator` takes a list of structured job objects and:
  - Deduplicates using **hashing** (exact hash + SimHash + LSH)
  - Scores jobs by **length & completion** (and optional freshness/source)
  - Preserves **variance** by keeping jobs that are **far apart** in hash space
  - Respects a global **compression ratio** (e.g. keep 40% of jobs)

No dense embeddings. Fully hashing + simple geometry (3D coordinates for cities).

### TODO
- publish package to PyPI:
- add Job Parsing
- add Job dynamic Tagging with Taxonomy
- add job auto-formating & Normalization

---

## Repository structure
```yaml
jobcurator/
├─ pyproject.toml
├─ README.md
└─ src/
   └─ jobcurator/
      ├─ __init__.py
      ├─ models.py
      ├─ hash_utils.py
      └─ curator.py
```

## 🚀 Installation
For local Dev:
```bash
git clone https://github.com/<your-username>/jobcurator.git
cd jobcurator
pip install -e .
```
(coming soon) Once published to PyPI:
```bash
pip install jobcurator
```

---


## 🧩 Public API

### Import

```python
from jobcurator import JobCurator, Job, Category, SalaryField, Location3DField
from datetime import datetime
```

### Example usage

```python
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
    # ... more jobs
]

curator = JobCurator(
    ratio=0.4,                 # keep 40% of jobs
    alpha=0.6,                 # quality vs diversity tradeoff
    max_per_cluster_in_pool=3, # max jobs per cluster entering the global pool
)

compressed_jobs = curator.dedupe_and_compress(jobs)
print(len(jobs), "→", len(compressed_jobs))
```

### JobCurator parameters

```python
JobCurator(
    ratio: float = 1.0,              # default compression ratio
    alpha: float = 0.6,              # quality vs diversity weight
    max_per_cluster_in_pool: int = 3,
    d_sim_threshold: int = 20,       # SimHash Hamming threshold for clustering
    max_cluster_distance_km: float = 150.0,  # max distance between cities in same cluster
)
```

* `ratio = 1.0` → keep all jobs
* `ratio = 0.5` → keep ~50% of jobs (highest quality + diversity)
* `alpha` closer to 1 → prioritize quality; closer to 0 → prioritize diversity

---

## 🧱 Core Concepts

### Job schema

A `Job` is a structured object with:

* `id`: unique identifier
* `title`: job title (string)
* `text`: full job description (string)
* `categories`: hierarchical taxonomy per dimension (`dict[str, list[Category]]`)
* `location`: `Location3DField` with lat/lon/alt (internally converted to 3D x,y,z)
* `salary`: optional `SalaryField`
* Optional: `company`, `contract_type`, `source`, `created_at`
* Internal fields: `length_score`, `completion_score_val`, `quality`, `exact_hash`, `signature` (computed by `JobCurator`)

### Category schema

A `Category` is a hierarchical node:

* `id`: unique taxonomy ID
* `label`: human-readable label
* `level`: depth in hierarchy (0 = root)
* `parent_id`: optional parent category id
* `level_path`: full path from root (e.g. `["Engineering", "Software", "Backend"]`)

Multiple dimensions (e.g. `job_function`, `industry`, `seniority`) can coexist in `categories`.

### Location schema with 3D coordinates

`Location3DField`:

* `lat`, `lon`: in degrees
* `alt_m`: altitude in meters
* `city`, `country_code`: metadata
* `x, y, z`: computed Earth-centered coordinates for **3D distance** (used to avoid merging jobs from very distant cities)

---

## ⚙️ How It Works (High Level)

1. **Preprocessing & scoring**

   * Compute token length → normalize to `length_score ∈ [0,1]` (using p10/p90 percentiles).
   * Compute `completion_score` based on presence of key fields (title, text, location, salary, categories, company, contract_type).
   * Optional `freshness_score` and `source_quality`.
   * Combine into:

     ```text
     quality(j) = 0.3 * length_score
                + 0.4 * completion_score
                + 0.2 * freshness_score
                + 0.1 * source_quality
     ```

2. **Exact hash**

   * Build a canonical string from title + categories + coarse location + salary bucket + text.
   * Use `blake2b` to get a 64-bit `exact_hash`.
   * Remove strict duplicates.

3. **Composite signature (no embeddings)**

   * 64-bit **SimHash** on `title + text`.
   * 64-bit **feature-hash** on categories, location, salary.
   * Concatenate into a 128-bit `signature = (simhash << 64) | meta_bits`.

4. **LSH clustering**

   * Use LSH on the SimHash part to find candidate near-duplicates.
   * Accept a pair as same cluster if:

     * Hamming distance on SimHash ≤ threshold
     * 3D geo distance between locations ≤ `max_cluster_distance_km`
   * Group jobs into clusters via union–find.

5. **Intra-cluster ranking**

   * Within each cluster, sort jobs by `quality` descending.

6. **Global compression with diversity**

   * Build a pool with the top N jobs per cluster.

   * Greedy selection:

     * Start from the highest-quality job.
     * Iteratively pick the job maximizing:

       ```text
       diversified_score = alpha * quality + (1 - alpha) * normalized_min_hamming_distance_to_selected
       ```

   * Stop when you’ve selected `ceil(ratio * N_original)` jobs.

Result: you keep **fewer, higher-quality, and more diverse** jobs.

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

## 📬 Contact

For questions, ideas, or coordination around larger changes:

**Primary maintainer**
📧 [mouhidine.seiv@hrflow.ai](mailto:mouhidine.seiv@hrflow.ai)



