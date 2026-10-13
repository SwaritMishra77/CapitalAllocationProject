"""Compare a fixed sample of selected Company Facts against original inline XBRL."""

import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

SAMPLE = ("WMT", "TGT", "NKE", "BBY", "CAT", "DE", "GD", "NOC")


def inline_values(path, tag, fiscal_end):
    soup = BeautifulSoup(gzip.open(path, "rb").read(), "html.parser")
    context_ok = set()
    for ctx in soup.find_all(re.compile(r"(?:^|:)context$", re.I)):
        ident = ctx.get("id")
        dates = [x.get_text(" ", strip=True) for x in ctx.find_all(re.compile(r"(?:^|:)startdate$|(?:^|:)enddate$", re.I))]
        if ident and len(dates) == 2 and dates[-1] == fiscal_end and not ctx.find(re.compile(r"(?:^|:)(?:explicitmember|typedmember)$", re.I)):
            if 330 <= (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days <= 400:
                context_ok.add(ident)
    usd = set()
    for unit in soup.find_all(re.compile(r"(?:^|:)unit$", re.I)):
        if re.search(r"(?:^|:)USD\b", unit.get_text(" ", strip=True), re.I):
            usd.add(unit.get("id"))
    result = []
    for fact in soup.find_all(re.compile(r"(?:^|:)nonfraction$", re.I)):
        if fact.get("name", "").lower() != "us-gaap:" + tag.lower():
            continue
        if fact.get("contextref") not in context_ok or fact.get("unitref") not in usd:
            continue
        fmt = fact.get("format", "").lower()
        if fmt not in ("", "ixt:numdotdecimal", "ixt:num-dot-decimal"):
            continue
        raw = fact.get_text("", strip=True).replace(",", "").replace("$", "").replace(" ", "")
        if not re.fullmatch(r"\(?-?\d+(?:\.\d+)?\)?", raw):
            continue
        value = float(raw.strip("()")) * (10 ** int(fact.get("scale", "0")))
        if raw.startswith("(") or fact.get("sign") == "-":
            value = -abs(value)
        result.append(value)
    return result


def run(panel, raw, output):
    d = pd.read_csv(panel, dtype={"cik": str})
    checks = []
    for ticker in SAMPLE:
        sample = d[(d.ticker == ticker) & d.has_text.eq(1)].sort_values("fiscal_end")
        if len(sample) < 2:
            raise ValueError(f"Too few cached source filings for {ticker}")
        for row in sample.tail(2).itertuples():
            source = raw / "html" / f"{row.source_accession}.html.gz"
            for name in ("revenue", "capex"):
                target = float(getattr(row, name))
                tag = getattr(row, name + "_tag")
                values = inline_values(source, tag, row.fiscal_end)
                verified = any(abs(x - target) <= max(1, abs(target) * 1e-8) for x in values)
                checks.append({"ticker": ticker, "fiscal_end": row.fiscal_end,
                               "accession": row.source_accession, "field": name, "tag": tag,
                               "panel_usd": target, "matched_inline_xbrl": verified,
                               "candidate_count": len(values)})
    result = {"design": "eight fixed firms, two latest text-covered original filings each, revenue and capex",
              "checks": len(checks), "matched": sum(x["matched_inline_xbrl"] for x in checks),
              "rows": checks}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if result["matched"] != result["checks"]:
        raise ValueError(f"Inline XBRL reconciliation failed: {result['matched']}/{result['checks']}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=Path("data/processed/panel.csv"))
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("reports/filing_reconciliation.json"))
    a = p.parse_args()
    print(json.dumps(run(a.panel, a.raw, a.out), indent=2))


if __name__ == "__main__":
    main()
