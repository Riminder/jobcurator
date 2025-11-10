import argparse
from datetime import datetime
from time import perf_counter

from jobcurator import JobCurator, Job, Category, SalaryField, Location3DField, CuckooFilter, curator


def build_jobs():
    return [
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test jobcurator: show first N preview jobs from the original list, "
            "then top N*ratio jobs from the compressed set."
        )
    )
    parser.add_argument(
        "-n",
        "--n-preview-jobs",
        type=int,
        default=10,
        help=(
            "Number of original jobs to preview (default: 10, "
            "capped at total number of jobs)."
        ),
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Compression ratio passed to JobCurator (default: 0.5).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="default_hash",
        choices=["default_hash", "minhash_hash", "sklearn_hash", "faiss_hash"],
        help="Backend to use for clustering/dedupe.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    jobs = build_jobs()
    total_jobs = len(jobs)

    if total_jobs == 0:
        print("No jobs available.")
        return

    n_preview = max(1, min(args.n_preview_jobs, total_jobs))

    # integrated CuckooFilter
    seen_filter = CuckooFilter(capacity=1_000_000)

    curator = JobCurator(
        ratio=args.ratio,
        backend=args.backend,
        use_multiprobe=True,
        max_multiprobe_flips=1,
    )

    # Dedupe & Compress
    from time import perf_counter
    t0 = perf_counter()
    compressed = curator.dedupe_and_compress(jobs, seen_filter=seen_filter)
    t_ms = (perf_counter() - t0) * 1000.0  # milliseconds

    # Print compression summary statistics
    curator.print_compression_summary(n_preview=n_preview, t_ms=t_ms)
    curator.print_jobs_summary(curator.jobs, n_preview=n_preview, label="All jobs set")
    curator.print_jobs_summary(compressed, n_preview=n_preview, label="compressed jobs set")


if __name__ == "__main__":
    main()
