"""Recreate the SEC source cache with a declared contact and courteous request rate."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class SecClient:
    def __init__(self, agent, delay=0.25):
        if not agent or "@" not in agent:
            raise ValueError("Set SEC_USER_AGENT to a truthful name and contact email before fetching")
        self.agent = agent
        self.delay = max(0.2, delay)
        self.last = 0.0

    def get(self, url):
        for attempt in range(5):
            time.sleep(max(0, self.delay - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            try:
                with urlopen(Request(url, headers={"User-Agent": self.agent, "Accept-Encoding": "identity"}), timeout=60) as r:
                    return r.read()
            except HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                    raise
            except URLError:
                if attempt == 4:
                    raise
            time.sleep(min(30, 2 ** attempt))


def flatten(recent):
    keys = list(recent)
    return [dict(zip(keys, values)) for values in zip(*(recent[k] for k in keys))]


def ingest(universe_path, out, agent, text_ciks=None, year_min=2012, year_max=2025):
    client = SecClient(agent)
    out.mkdir(parents=True, exist_ok=True)
    universe = list(csv.DictReader(universe_path.open()))
    full = []
    for item in universe:
        cik = item["cik"].zfill(10)
        fact_path, submission_path, filing_path = [out / sub / f"{cik}.{ext}" for sub, ext in
                                                  [("facts", "json.gz"), ("submissions", "json"), ("filings", "json")]]
        for p in (fact_path, submission_path, filing_path):
            p.parent.mkdir(parents=True, exist_ok=True)
        if not fact_path.exists():
            data = client.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
            json.loads(data)
            fact_path.write_bytes(gzip.compress(data, mtime=0))
        if not submission_path.exists():
            raw = client.get(f"https://data.sec.gov/submissions/CIK{cik}.json")
            json.loads(raw)
            submission_path.write_bytes(raw)
        if not filing_path.exists():
            submission = json.loads(submission_path.read_text())
            records = flatten(submission["filings"]["recent"])
            for prior in submission["filings"].get("files", []):
                contents = client.get("https://data.sec.gov/submissions/" + prior["name"])
                records.extend(flatten(json.loads(contents)))
            filing_path.write_text(json.dumps(records))
        records = json.loads(filing_path.read_text())
        if text_ciks is None or cik in text_ciks:
            for record in records:
                if record.get("form") != "10-K" or not record.get("reportDate") or not record.get("primaryDocument"):
                    continue
                if not year_min <= int(record["reportDate"][:4]) <= year_max:
                    continue
                accession = record["accessionNumber"]
                html = out / "html" / f"{accession}.html.gz"
                html.parent.mkdir(parents=True, exist_ok=True)
                if html.exists():
                    continue
                url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                       f"{accession.replace('-', '')}/{record['primaryDocument']}")
                body = client.get(url)
                html.write_bytes(gzip.compress(body, mtime=0))
                (out / "html" / f"{accession}.html.gz.meta.json").write_text(json.dumps({"url": url,
                    "filed": record["filingDate"], "accession": accession}, indent=2))
        full.append({**item, "cik": cik, "name": json.loads(gzip.open(fact_path, "rt").read()).get("entityName", ""), "status": "ok"})
        print(f"{item['ticker']}: cached {len(records)} filings", flush=True)
    (out / "universe_resolved.json").write_text(json.dumps(full, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--universe", type=Path, default=Path("configs/universe.csv"))
    p.add_argument("--text-cohort", type=Path, default=Path("configs/text_cohort.csv"))
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.add_argument("--all-text", action="store_true", help="Fetch 10-K HTML for all 100 companies")
    p.add_argument("--year-min", type=int, default=2012)
    p.add_argument("--year-max", type=int, default=2025)
    a = p.parse_args()
    cohort = None if a.all_text else {r["cik"].zfill(10) for r in csv.DictReader(a.text_cohort.open())}
    ingest(a.universe, a.out, os.getenv("SEC_USER_AGENT"), cohort, a.year_min, a.year_max)


if __name__ == "__main__":
    main()
