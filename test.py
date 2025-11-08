from jobcurator import JobCurator, Job, Category, SalaryField, Location3DField
from datetime import datetime

jobs = [
    Job(
        id="job-1",
        title="Senior Backend Engineer",
        text="Full description for a senior backend engineer working on Python microservices and APIs.",
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

    # Near-duplicate of job-1 (same city, similar text, slightly different title)
    Job(
        id="job-2",
        title="Backend Engineer (Senior)",
        text="We are looking for a senior backend engineer to build Python microservices and scalable APIs.",
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
            alt_m=40,
            city="Paris",
            country_code="FR",
        ),
        salary=SalaryField(
            min_value=62000,
            max_value=82000,
            currency="EUR",
            period="year",
        ),
        company="HrFlow.ai",
        contract_type="Full-time",
        source="direct",
        created_at=datetime.utcnow(),
    ),

    # Same role, different city (should not cluster with Paris if distance threshold is strict)
    Job(
        id="job-3",
        title="Senior Backend Engineer",
        text="Senior backend engineer role working on distributed systems and REST APIs in Go and Python.",
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
            lat=51.5074,
            lon=-0.1278,
            alt_m=25,
            city="London",
            country_code="GB",
        ),
        salary=SalaryField(
            min_value=70000,
            max_value=90000,
            currency="GBP",
            period="year",
        ),
        company="TechFlow Ltd",
        contract_type="Full-time",
        source="job_board",
        created_at=datetime.utcnow(),
    ),

    # Different function: Data Scientist in Paris
    Job(
        id="job-4",
        title="Data Scientist",
        text="Data scientist working on machine learning models, experimentation and analytics.",
        categories={
            "job_function": [
                Category(
                    id="data",
                    label="Data Science",
                    level=1,
                    parent_id="eng",
                    level_path=["Engineering", "Data", "Data Science"],
                )
            ]
        },
        location=Location3DField(
            lat=48.8566,
            lon=2.3522,
            alt_m=33,
            city="Paris",
            country_code="FR",
        ),
        salary=SalaryField(
            min_value=55000,
            max_value=75000,
            currency="EUR",
            period="year",
        ),
        company="HrFlow.ai",
        contract_type="Full-time",
        source="direct",
        created_at=datetime.utcnow(),
    ),

    # Short / low completion job (should get lower quality)
    Job(
        id="job-5",
        title="Backend Dev",
        text="Backend dev wanted.",
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
            alt_m=30,
            city="Paris",
            country_code="FR",
        ),
        salary=None,
        company=None,
        contract_type=None,
        source="job_board",
        created_at=datetime.utcnow(),
    ),

    # Product Manager, New York
    Job(
        id="job-6",
        title="Product Manager",
        text="Product manager responsible for roadmap, stakeholder alignment, and discovery.",
        categories={
            "job_function": [
                Category(
                    id="product",
                    label="Product Management",
                    level=1,
                    parent_id="biz",
                    level_path=["Business", "Product", "Product Management"],
                )
            ]
        },
        location=Location3DField(
            lat=40.7128,
            lon=-74.0060,
            alt_m=10,
            city="New York",
            country_code="US",
        ),
        salary=SalaryField(
            min_value=90000,
            max_value=120000,
            currency="USD",
            period="year",
        ),
        company="GlobalTech",
        contract_type="Full-time",
        source="direct",
        created_at=datetime.utcnow(),
    ),

    # Near-duplicate of job-4 (Data Scientist in Paris, slightly different wording)
    Job(
        id="job-7",
        title="Data Scientist (ML)",
        text="We are hiring a data scientist to build and deploy machine learning models and run experiments.",
        categories={
            "job_function": [
                Category(
                    id="data",
                    label="Data Science",
                    level=1,
                    parent_id="eng",
                    level_path=["Engineering", "Data", "Data Science"],
                )
            ]
        },
        location=Location3DField(
            lat=48.8566,
            lon=2.3522,
            alt_m=32,
            city="Paris",
            country_code="FR",
        ),
        salary=SalaryField(
            min_value=58000,
            max_value=78000,
            currency="EUR",
            period="year",
        ),
        company="HrFlow.ai",
        contract_type="Full-time",
        source="job_board",
        created_at=datetime.utcnow(),
    ),

    # Very detailed long description (should get high length_score)
    Job(
        id="job-8",
        title="Full Stack Engineer",
        text=(
            "We are looking for a full stack engineer to work on our core web platform. "
            "You will design and implement frontend components in React, backend APIs in Python, "
            "collaborate with product and design, and help maintain CI/CD pipelines. "
            "Experience with cloud infrastructure, testing, and performance optimization is a plus."
        ),
        categories={
            "job_function": [
                Category(
                    id="fullstack",
                    label="Full Stack",
                    level=1,
                    parent_id="eng",
                    level_path=["Engineering", "Software", "Full Stack"],
                )
            ]
        },
        location=Location3DField(
            lat=52.5200,
            lon=13.4050,
            alt_m=34,
            city="Berlin",
            country_code="DE",
        ),
        salary=SalaryField(
            min_value=65000,
            max_value=90000,
            currency="EUR",
            period="year",
        ),
        company="CloudApps",
        contract_type="Full-time",
        source="direct",
        created_at=datetime.utcnow(),
    ),
]

curator = JobCurator(ratio=0.5)
compressed_jobs = curator.dedupe_and_compress(jobs)
print(len(jobs), "→", len(compressed_jobs))
for j in compressed:
    print(j.id, j.title, j.location.city)

