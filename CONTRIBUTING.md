
## 📚 Table of Contents

- [Contributing](#-contributing)
  - [Getting Started](#-getting-started)
  - [Reporting Bugs](#-reporting-bugs)
  - [Suggesting Features](#-suggesting-features)
  - [Tests & Quality](#-tests--quality)
  - [Code Style & Guidelines](#-code-style--guidelines)
  - [Public API & Backward Compatibility](#-public-api--backward-compatibility)
  - [Pull Requests](#-pull-requests)

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
