<div style="width:260px; height:100px; overflow:hidden; border-radius:8px;">
  <img src="./logo.png"
       style="width:100%; height:auto; object-fit:cover; object-position:center 50%;" />
</div>

# jobcurator
Open source Machine Learning library to clean, normalize, structure, compress, and sample large datasets & feeds of job offers.

### Available features:
- Hash-based job deduplication and compression with diversity preservation

### TODO
- add Job Parsing
- add Job dynamic Tagging with Taxonomy
- add job auto-formating & Normalization


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

## Package installation via Pypi
```bash
pip install jobcurator
```

## Package installation via GitHub
1. Clone the repo
```bash
git clone https://github.com/Riminder/jobcurator.git
```
2. Once this is in your repo:
```bash
cd jobcurator
pip install .
# or, for dev
pip install -e .
```


## Code example

```python
from jobcurator import JobCurator, Job, Category, SalaryField, Location3DField
from datetime import datetime

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
            lat=48.8566, lon=2.3522, alt_m=35, city="Paris", country_code="FR"
        ),
        salary=SalaryField(min_value=60000, max_value=80000, currency="EUR", period="year"),
        company="HrFlow.ai",
        contract_type="Full-time",
        source="direct",
        created_at=datetime.utcnow(),
    ),
]

curator = JobCurator(ratio=0.4)
compressed_jobs = curator.dedupe_and_compress(jobs)
print(len(jobs), "→", len(compressed_jobs))
```

