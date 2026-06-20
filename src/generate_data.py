"""
Synthetic salary dataset generator for the Data Job Market Analysis case study.

WHY SYNTHETIC?
--------------
The analysis in this project mirrors the well-known Kaggle "Data Science Job
Salaries" dataset, but to keep the repository fully self-contained and free of
licensing/redistribution concerns, we GENERATE a realistic synthetic dataset
with the *same schema*. This is stated openly in the README and the notebook --
the goal of the case study is to demonstrate the analytical workflow, and the
synthetic data is built from documented, realistic assumptions.

The generator encodes real-world structure that the analysis later "discovers":
  * seniority premium       (EX > SE > MI > EN)
  * geographic differences  (US pays the most; emerging markets the least)
  * a modest remote premium
  * year-over-year salary growth
  * a company-size effect    (large > medium > small)

It also injects a SMALL amount of realistic messiness (a few duplicates, some
missing values, and a handful of data-entry outliers) on purpose, so that the
notebook's data-cleaning section is non-trivial and demonstrates real cleaning
skill. All randomness is seeded for full reproducibility -- the numbers are
identical on every run.

Output schema (matches the Kaggle dataset):
    work_year, experience_level, employment_type, job_title, salary_in_usd,
    employee_residence, remote_ratio, company_location, company_size
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Reproducibility: a fixed seed guarantees the dataset is identical on every run,
# which keeps every number quoted in the notebook and README consistent.
SEED = 42
N_ROWS = 9000

DEFAULT_RAW_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "data_jobs_salaries_raw.csv"
)

# ---------------------------------------------------------------------------
# Model parameters (documented, realistic assumptions)
# ---------------------------------------------------------------------------

YEARS = [2020, 2021, 2022, 2023, 2024]
# More recent years contribute more rows (data collection grew over time).
YEAR_WEIGHTS = [0.05, 0.12, 0.20, 0.30, 0.33]
# Year-over-year salary growth factor relative to 2020.
YEAR_GROWTH = {2020: 1.00, 2021: 1.05, 2022: 1.12, 2023: 1.18, 2024: 1.24}

# Experience levels (Kaggle codes) and their salary multiplier relative to MI.
EXPERIENCE_LEVELS = ["EN", "MI", "SE", "EX"]
EXPERIENCE_WEIGHTS = [0.20, 0.32, 0.40, 0.08]
EXPERIENCE_MULT = {"EN": 0.65, "MI": 1.00, "SE": 1.35, "EX": 1.85}

# Employment type: overwhelmingly full-time, like the real market.
EMPLOYMENT_TYPES = ["FT", "CT", "FL", "PT"]
EMPLOYMENT_WEIGHTS = [0.95, 0.02, 0.02, 0.01]
EMPLOYMENT_MULT = {"FT": 1.00, "CT": 1.05, "FL": 0.95, "PT": 0.55}

# Job titles with a US, mid-level, onsite baseline salary (USD) and a popularity
# weight. Bases are anchored to publicly reported market ranges.
JOB_TITLES = {
    "Data Analyst": {"base": 95_000, "weight": 0.16},
    "Data Scientist": {"base": 135_000, "weight": 0.18},
    "Data Engineer": {"base": 130_000, "weight": 0.18},
    "Machine Learning Engineer": {"base": 150_000, "weight": 0.12},
    "Analytics Engineer": {"base": 125_000, "weight": 0.08},
    "Business Intelligence Analyst": {"base": 100_000, "weight": 0.09},
    "Research Scientist": {"base": 140_000, "weight": 0.06},
    "AI Engineer": {"base": 155_000, "weight": 0.05},
    "Data Architect": {"base": 145_000, "weight": 0.04},
    "Data Science Manager": {"base": 165_000, "weight": 0.04},
}

# Country (ISO codes) with a cost/market factor relative to the US and a weight
# describing how common that residence is in the dataset.
COUNTRIES = {
    "US": {"factor": 1.00, "weight": 0.58},
    "GB": {"factor": 0.75, "weight": 0.07},
    "CA": {"factor": 0.85, "weight": 0.06},
    "IN": {"factor": 0.35, "weight": 0.05},
    "DE": {"factor": 0.72, "weight": 0.05},
    "FR": {"factor": 0.65, "weight": 0.03},
    "ES": {"factor": 0.55, "weight": 0.03},
    "AU": {"factor": 0.80, "weight": 0.02},
    "NL": {"factor": 0.70, "weight": 0.02},
    "BR": {"factor": 0.30, "weight": 0.02},
    "PT": {"factor": 0.45, "weight": 0.015},
    "PL": {"factor": 0.45, "weight": 0.015},
    "JP": {"factor": 0.60, "weight": 0.015},
    "CH": {"factor": 1.05, "weight": 0.015},
    "SG": {"factor": 0.78, "weight": 0.01},
}

# Company size codes, their popularity weight and salary multiplier.
COMPANY_SIZES = ["S", "M", "L"]
COMPANY_SIZE_WEIGHTS = [0.15, 0.55, 0.30]
COMPANY_SIZE_MULT = {"S": 0.90, "M": 1.00, "L": 1.08}

# Remote ratio distribution depends on the year (remote peaked in 2021-2022,
# then declined as return-to-office gathered pace). Order: [0, 50, 100].
REMOTE_BY_YEAR = {
    2020: [0.50, 0.15, 0.35],
    2021: [0.30, 0.15, 0.55],
    2022: [0.30, 0.20, 0.50],
    2023: [0.45, 0.22, 0.33],
    2024: [0.55, 0.22, 0.23],
}
REMOTE_OPTIONS = [0, 50, 100]
# A modest premium for fully-remote roles relative to onsite.
REMOTE_MULT = {0: 1.00, 50: 1.02, 100: 1.06}


def _normalize(weights: list[float]) -> list[float]:
    """Return weights rescaled to sum to exactly 1.0 (np.choice is picky)."""
    arr = np.array(weights, dtype=float)
    return (arr / arr.sum()).tolist()


def generate_clean_frame(rng: np.random.Generator) -> pd.DataFrame:
    """Build the 'true' clean dataset from the documented salary model."""
    country_codes = list(COUNTRIES.keys())
    country_weights = _normalize([c["weight"] for c in COUNTRIES.values()])

    title_names = list(JOB_TITLES.keys())
    title_weights = _normalize([t["weight"] for t in JOB_TITLES.values()])

    years = rng.choice(YEARS, size=N_ROWS, p=_normalize(YEAR_WEIGHTS))
    experience = rng.choice(
        EXPERIENCE_LEVELS, size=N_ROWS, p=_normalize(EXPERIENCE_WEIGHTS)
    )
    employment = rng.choice(
        EMPLOYMENT_TYPES, size=N_ROWS, p=_normalize(EMPLOYMENT_WEIGHTS)
    )
    titles = rng.choice(title_names, size=N_ROWS, p=title_weights)
    residence = rng.choice(country_codes, size=N_ROWS, p=country_weights)
    sizes = rng.choice(COMPANY_SIZES, size=N_ROWS, p=_normalize(COMPANY_SIZE_WEIGHTS))

    # Remote ratio is drawn per-row using that row's year distribution.
    remote = np.array(
        [
            rng.choice(REMOTE_OPTIONS, p=_normalize(REMOTE_BY_YEAR[int(y)]))
            for y in years
        ]
    )

    # Company location equals residence ~85% of the time; otherwise a random
    # different country (models people working for a company abroad).
    company_location = residence.copy()
    relocate_mask = rng.random(N_ROWS) < 0.15
    company_location[relocate_mask] = rng.choice(
        country_codes, size=relocate_mask.sum(), p=country_weights
    )

    # ---- Salary model -----------------------------------------------------
    base = np.array([JOB_TITLES[t]["base"] for t in titles], dtype=float)
    exp_mult = np.array([EXPERIENCE_MULT[e] for e in experience])
    emp_mult = np.array([EMPLOYMENT_MULT[e] for e in employment])
    # Salary tracks the *company* location market (where the role is paid).
    geo_mult = np.array([COUNTRIES[c]["factor"] for c in company_location])
    size_mult = np.array([COMPANY_SIZE_MULT[s] for s in sizes])
    year_mult = np.array([YEAR_GROWTH[int(y)] for y in years])
    remote_mult = np.array([REMOTE_MULT[int(r)] for r in remote])

    # Multiplicative log-normal noise gives a realistic right-skewed spread.
    noise = rng.lognormal(mean=0.0, sigma=0.15, size=N_ROWS)

    salary = (
        base * exp_mult * emp_mult * geo_mult * size_mult * year_mult * remote_mult * noise
    )
    # Round to the nearest $100, the way reported salaries usually look.
    salary_in_usd = (np.round(salary / 100) * 100).astype(int)

    df = pd.DataFrame(
        {
            "work_year": years.astype(int),
            "experience_level": experience,
            "employment_type": employment,
            "job_title": titles,
            "salary_in_usd": salary_in_usd,
            "employee_residence": residence,
            "remote_ratio": remote.astype(int),
            "company_location": company_location,
            "company_size": sizes,
        }
    )
    return df


def inject_messiness(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add a small, realistic amount of dirt so cleaning is meaningful.

    Injected on purpose (documented in the notebook):
      * duplicate rows,
      * missing values in a few columns,
      * a handful of impossible salary outliers (data-entry errors),
      * inconsistent text formatting in job_title.
    """
    df = df.copy()

    # 1) Inconsistent text formatting: stray whitespace / casing in some titles.
    fmt_idx = rng.choice(df.index, size=120, replace=False)
    df.loc[fmt_idx[:60], "job_title"] = df.loc[fmt_idx[:60], "job_title"] + "  "
    df.loc[fmt_idx[60:], "job_title"] = df.loc[fmt_idx[60:], "job_title"].str.lower()

    # 2) Missing values: a few exports lose company_size / remote_ratio / salary.
    miss_size = rng.choice(df.index, size=180, replace=False)       # ~2%
    df.loc[miss_size, "company_size"] = np.nan
    miss_remote = rng.choice(df.index, size=140, replace=False)     # ~1.5%
    df.loc[miss_remote, "remote_ratio"] = np.nan
    miss_salary = rng.choice(df.index, size=45, replace=False)      # ~0.5%
    df.loc[miss_salary, "salary_in_usd"] = np.nan

    # 3) Impossible salary outliers (typos: extra zero, or value entered in
    #    thousands instead of dollars).
    out_idx = rng.choice(df.dropna(subset=["salary_in_usd"]).index, size=15, replace=False)
    df.loc[out_idx[:8], "salary_in_usd"] = df.loc[out_idx[:8], "salary_in_usd"] * 10
    df.loc[out_idx[8:], "salary_in_usd"] = rng.integers(5, 60, size=len(out_idx[8:]))

    # 4) Duplicate rows (a classic export artefact).
    dup_idx = rng.choice(df.index, size=80, replace=False)
    duplicates = df.loc[dup_idx].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # Shuffle so the dirt is not all clustered at the end.
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df


def generate(path: Path | str = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Generate the messy raw dataset, write it to CSV, and return it."""
    rng = np.random.default_rng(SEED)
    clean = generate_clean_frame(rng)
    messy = inject_messiness(clean, rng)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    messy.to_csv(path, index=False)
    return messy


def main() -> None:
    df = generate()
    print(f"Generated {len(df):,} rows (raw, with intentional imperfections).")
    print(f"Saved to {DEFAULT_RAW_PATH}")
    print("\nColumn summary:")
    print(df.dtypes.to_string())
    print(f"\nMissing values per column:\n{df.isna().sum().to_string()}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print("\nPreview:")
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
