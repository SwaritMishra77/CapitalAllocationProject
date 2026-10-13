"""Build a point-in-time company-year panel from SEC Company Facts and 10-Ks."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "repurchase": ["PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity"],
    "dividend": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends", "PaymentsOfDividendsAndDividendEquivalentsOnCommonStockAndRestrictedStockUnits"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "rnd": ["ResearchAndDevelopmentExpense"],
}
KEYWORDS = re.compile(r"capital expenditur|capital investment|invest(?:ing|ments?) in (?:our |the )?(?:stores|facilities|equipment|capacity)|share repurchas|stock repurchas|buyback|dividend|research and development", re.I)


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "ix:header"):
            self.hidden += 1
        elif tag in ("p", "div", "tr", "br", "li"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "ix:header"):
            self.hidden = max(0, self.hidden - 1)
        elif tag in ("p", "div", "tr", "li"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def passages(path: Path) -> tuple[str, list[dict]]:
    """Deterministic excerpts from cached original 10-K HTML; never infer missing text."""
    if not path.exists():
        return "", []
    raw = gzip.open(path, "rb").read()
    parser = VisibleText()
    parser.feed(raw.decode("utf-8", errors="replace"))
    clean = re.sub(r"[ \t\u00a0]+", " ", "".join(parser.parts))
    clean = re.sub(r"\n\s*\n+", "\n", clean)
    clean = re.sub(r"([a-zA-Z])\n([a-zA-Z])", r"\1 \2", clean)
    hits = []
    for match in KEYWORDS.finditer(clean):
        left = max(0, match.start() - 110)
        right = min(len(clean), match.end() + 300)
        excerpt = " ".join(clean[left:right].split())
        if len(excerpt) < 100 or any(abs(match.start() - x[0]) < 320 for x in hits):
            continue
        topic = "capex" if re.search(r"capital expenditur|capital investment", match.group(), re.I) else "other"
        score = 2 * len(re.findall(r"expect|plan|intend|anticipat|will invest|year ahead|next year|fiscal year", excerpt, re.I))
        score += len(re.findall(r"\$\s?\d|\d+(?:\.\d+)?\s?(?:billion|million)", excerpt, re.I))
        score -= 6 * len(re.findall(r"forward.looking statements|table of contents|risk factors", excerpt, re.I))
        hits.append((match.start(), excerpt, left, right, topic, score))
    chosen = sorted(hits, key=lambda x: (-x[5], x[0]))
    capex = [x for x in chosen if x[4] == "capex"][:4]
    others = [x for x in chosen if x[4] != "capex"][:2]
    selected = capex + others if capex else chosen[:4]
    evidence = [{"offset_start": a, "offset_end": b, "excerpt": e, "selection_score": s}
                for _, e, a, b, _, s in selected]
    return " ".join(x["excerpt"] for x in evidence), evidence


def first_filings(records):
    """One original 10-K per report date. Amendments are excluded."""
    valid = [r for r in records if r.get("form") == "10-K" and r.get("reportDate") and r.get("filingDate")]
    chosen = {}
    for r in sorted(valid, key=lambda x: (x["filingDate"], x["accessionNumber"])):
        chosen.setdefault(r["reportDate"], r)
    return sorted(chosen.values(), key=lambda x: x["reportDate"])


def annual_fact(facts, category, filing):
    """Take the 12-month USD item in the filing's own accession, not later comparative restatements."""
    candidates = []
    for tag in TAGS[category]:
        units = facts.get("us-gaap", {}).get(tag, {}).get("units", {}).get("USD", [])
        found = [v for v in units if v.get("accn") == filing["accessionNumber"]
                 and v.get("end") == filing["reportDate"] and v.get("start")
                 and 330 <= (date.fromisoformat(v["end"]) - date.fromisoformat(v["start"])).days <= 400
                 and v.get("filed", "9999") <= filing["filingDate"]]
        if found:
            v = sorted(found, key=lambda x: (abs((date.fromisoformat(x["end"]) - date.fromisoformat(x["start"])).days - 365), x["start"]))[0]
            if category != "revenue":
                return float(v["val"]), tag
            if float(v["val"]) > 0:
                candidates.append((float(v["val"]), tag))
    if candidates:
        # Consolidated revenue should dominate narrower sales/discontinued-operation facts.
        # Retain the chosen tag so conflicts can be audited by accession.
        return max(candidates)
    return None, None


def make_panel(raw_dir: Path, out_dir: Path, max_year: int = 2025) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    universe = json.loads((raw_dir / "universe_resolved.json").read_text())
    rows, evidence, audit = [], [], {"companies": len(universe), "missing_revenue": 0, "missing_capex": 0, "missing_text": 0, "skipped_next_year": 0}
    for company in universe:
        cik = company["cik"]
        fact_path, filings_path = raw_dir / "facts" / f"{cik}.json.gz", raw_dir / "filings" / f"{cik}.json"
        if not fact_path.exists() or not filings_path.exists():
            raise FileNotFoundError(f"Incomplete source cache for CIK {cik}")
        with gzip.open(fact_path, "rt") as f:
            facts = json.load(f)["facts"]
        fs = first_filings(json.loads(filings_path.read_text()))
        annual = []
        for filing in fs:
            end = filing["reportDate"]
            if not (2012 <= int(end[:4]) <= max_year):
                continue
            values = {k: annual_fact(facts, k, filing) for k in TAGS}
            if values["revenue"][0] is None or values["revenue"][0] <= 0:
                audit["missing_revenue"] += 1
                continue
            if values["capex"][0] is None:
                audit["missing_capex"] += 1
                continue
            annual.append((filing, values))
        for i in range(len(annual) - 1):
            filing, v = annual[i]
            next_filing, nv = annual[i + 1]
            gap = (date.fromisoformat(next_filing["reportDate"]) - date.fromisoformat(filing["reportDate"])).days
            if not 330 <= gap <= 400:
                audit["skipped_next_year"] += 1
                continue
            accession = filing["accessionNumber"]
            html = raw_dir / "html" / f"{accession}.html.gz"
            narrative, excerpts = passages(html)
            if not narrative:
                audit["missing_text"] += 1
            ref = f"{cik}:{filing['reportDate']}"
            row = {"id": ref, "cik": cik, "ticker": company["ticker"], "sector": company["sector"],
                   "fiscal_end": filing["reportDate"], "year": int(filing["reportDate"][:4]),
                   "known_at": filing["filingDate"], "source_accession": accession,
                   "target_accession": next_filing["accessionNumber"], "target_known_at": next_filing["filingDate"],
                   "target_fiscal_end": next_filing["reportDate"],
                   "source_url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{filing.get('primaryDocument', '')}",
                   "source_sha256": hashlib.sha256(fact_path.read_bytes()).hexdigest(),
                   "html_sha256": hashlib.sha256(html.read_bytes()).hexdigest() if html.exists() else "",
                   "text": narrative, "has_text": int(bool(narrative))}
            for key in TAGS:
                row[key] = v[key][0]
                row[key + "_tag"] = v[key][1] or ""
            for key in ("revenue", "capex", "repurchase", "dividend", "cfo", "rnd"):
                row["next_" + key] = nv[key][0]
                row["next_" + key + "_tag"] = nv[key][1] or ""
            row["capex_rate"] = row["capex"] / row["revenue"]
            row["target_capex_rate"] = row["next_capex"] / row["next_revenue"]
            row["repurchase_rate"] = row["repurchase"] / row["revenue"] if row["repurchase"] is not None else None
            row["target_repurchase_rate"] = row["next_repurchase"] / row["next_revenue"] if row["next_repurchase"] is not None else None
            row["cfo_rate"] = row["cfo"] / row["revenue"] if row["cfo"] is not None else None
            row["rnd_rate"] = row["rnd"] / row["revenue"] if row["rnd"] is not None else None
            row["dividend_rate"] = row["dividend"] / row["revenue"] if row["dividend"] is not None else None
            if not (math.isfinite(row["capex_rate"]) and math.isfinite(row["target_capex_rate"])):
                continue
            rows.append(row)
            evidence.append({"id": ref, "html_file": html.name if html.exists() else "", "passages": excerpts})
    rows.sort(key=lambda r: (r["known_at"], r["cik"]))
    with (out_dir / "panel.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (out_dir / "evidence.jsonl").open("w") as f:
        for item in evidence:
            f.write(json.dumps(item) + "\n")
    audit.update({"rows": len(rows), "rows_with_text": sum(r["has_text"] for r in rows),
                  "first_known_at": rows[0]["known_at"], "last_known_at": rows[-1]["known_at"]})
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--max-year", type=int, default=2025)
    a = p.parse_args()
    print(json.dumps(make_panel(a.raw, a.out, a.max_year), indent=2))


if __name__ == "__main__":
    main()
