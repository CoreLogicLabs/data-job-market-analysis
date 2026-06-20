"""
Ethical web scraping demo for the Data Job Market Analysis case study.

WHY THIS SITE?
--------------
This module scrapes https://realpython.github.io/fake-jobs/ -- a sandbox site
that the Real Python team built *specifically* so people can practice web
scraping legally and safely. It contains fabricated job postings and is meant
to be hammered by learners, so scraping it does not harm a real business, does
not violate any Terms of Service, and does not touch private or personal data.

It is a deliberate choice: demonstrating the *technique* of scraping without the
ethical and legal grey areas that come with scraping a live commercial job board
(most of which forbid automated access in their ToS and rate-limit aggressively).

RESPECTFUL SCRAPING PRACTICES SHOWN HERE
----------------------------------------
1. A descriptive, honest User-Agent string (we identify ourselves rather than
   pretending to be a normal browser).
2. A polite delay (time.sleep) between requests to avoid hammering the server.
3. Timeouts and explicit error handling so we fail gracefully.
4. Scraping only the data we actually need (title, company, location, date).

The scraped result is a tidy pandas DataFrame, optionally written to
data/raw/ as a CSV so the rest of the pipeline can consume it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Optional: make Python's TLS use the operating system's trust store. This is
# helpful on machines behind a corporate proxy / antivirus that performs SSL
# inspection, where the standard certifi bundle may not recognise the injected
# root CA. It is a no-op (silently skipped) if `truststore` is not installed, so
# the scraper still works on a clean machine with plain requests + certifi.
try:  # pragma: no cover - environment dependent
    import truststore

    truststore.inject_into_ssl()
except Exception:  # truststore missing or injection failed -> use defaults
    pass

# Sandbox site explicitly built for scraping practice (see module docstring).
BASE_URL = "https://realpython.github.io/fake-jobs/"

# An honest User-Agent: we say who we are and why. Good scraping etiquette is to
# be identifiable rather than to disguise the bot as a regular browser.
HEADERS = {
    "User-Agent": (
        "DataJobMarketAnalysis/1.0 (portfolio case study; "
        "ethical scraping demo on a public sandbox site)"
    )
}

# Polite pause between HTTP requests, in seconds. Even though this static demo
# site has a single page, we keep the delay in the request helper to model the
# correct behaviour for multi-page / multi-request crawls.
REQUEST_DELAY_SECONDS = 1.0

# Default output location for the raw scraped data.
DEFAULT_RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "scraped_jobs.csv"


@dataclass
class JobPosting:
    """A single job posting scraped from the listings page."""

    title: str
    company: str
    location: str
    date_posted: str
    apply_url: str


def fetch_page(url: str, *, delay: float = REQUEST_DELAY_SECONDS) -> str:
    """Fetch a single page politely and return its HTML.

    Demonstrates respectful request behaviour:
      * sends a descriptive User-Agent header,
      * uses a timeout so a hung connection cannot block us forever,
      * raises a clear error on non-200 responses,
      * sleeps briefly after the request to avoid overloading the server.
    """
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()  # turn 4xx/5xx into an exception we can handle

    # Be a good citizen: pause before the caller issues the next request.
    time.sleep(delay)
    return response.text


def parse_jobs(html: str) -> list[JobPosting]:
    """Parse the listings HTML into a list of JobPosting records.

    The fake-jobs page wraps each posting in a ``div.card-content`` element.
    Inside each card we read the title, company, location, posting date and the
    "Apply" link. We guard every field with ``getattr``/conditionals so a single
    malformed card cannot crash the whole scrape.
    """
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("div.card-content")

    jobs: list[JobPosting] = []
    for card in cards:
        title_el = card.select_one("h2.title")
        company_el = card.select_one("h3.company")
        location_el = card.select_one("p.location")
        date_el = card.select_one("time")

        # The second footer link ("Apply") points to the job detail page.
        apply_el = card.select_one("a:-soup-contains('Apply')")

        title = title_el.get_text(strip=True) if title_el else ""
        company = company_el.get_text(strip=True) if company_el else ""
        location = location_el.get_text(strip=True) if location_el else ""
        date_posted = (
            date_el.get("datetime", "").strip()
            if date_el is not None
            else ""
        )
        apply_url = apply_el.get("href", "").strip() if apply_el is not None else ""

        # Skip empty / malformed cards defensively.
        if not title:
            continue

        jobs.append(
            JobPosting(
                title=title,
                company=company,
                location=location,
                date_posted=date_posted,
                apply_url=apply_url,
            )
        )

    return jobs


def scrape_jobs(url: str = BASE_URL) -> pd.DataFrame:
    """Scrape the sandbox job board and return a tidy DataFrame.

    Raises a RuntimeError with a friendly message if the network request fails,
    so callers (e.g. the notebook) can show a clean message instead of a stack
    trace.
    """
    try:
        html = fetch_page(url)
    except requests.RequestException as exc:  # network/HTTP problems
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc

    jobs = parse_jobs(html)
    df = pd.DataFrame([asdict(job) for job in jobs])

    # Parse the posting date into a proper datetime for downstream use.
    if not df.empty:
        df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")

    return df


def save_raw(df: pd.DataFrame, path: Path | str = DEFAULT_RAW_PATH) -> Path:
    """Persist the scraped DataFrame to ``data/raw`` as CSV and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    """Run the scraper from the command line and save the raw CSV."""
    print(f"Scraping job postings from {BASE_URL} ...")
    df = scrape_jobs()
    print(f"Scraped {len(df)} job postings.")

    out_path = save_raw(df)
    print(f"Saved raw data to {out_path}")

    # Show a small preview so a CLI user gets immediate feedback.
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
