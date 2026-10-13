"""Fail-fast structural checks for the point-in-time company-year panel."""

import argparse
import json
from pathlib import Path

import pandas as pd


def run(path, output):
    d = pd.read_csv(path, dtype={"cik": str})
    checks = {
        "unique_company_year": not d.duplicated(["cik", "fiscal_end"]).any(),
        "unique_id": d.id.is_unique,
        "positive_revenue": d.revenue.gt(0).all(),
        "positive_target_revenue": d.next_revenue.gt(0).all(),
        "nonnegative_capex": d.capex.ge(0).all() and d.next_capex.ge(0).all(),
        "source_precedes_target_filing": d.known_at.lt(d.target_known_at).all(),
        "source_fiscal_end_precedes_filing": d.fiscal_end.le(d.known_at).all(),
        "target_fiscal_end_precedes_filing": d.target_fiscal_end.le(d.target_known_at).all(),
        "annual_target_gap": (pd.to_datetime(d.target_fiscal_end)-pd.to_datetime(d.fiscal_end)).dt.days.between(330,400).all(),
        "capex_rate_reconciles": ((d.capex/d.revenue)-d.capex_rate).abs().lt(1e-12).all(),
        "target_rate_reconciles": ((d.next_capex/d.next_revenue)-d.target_capex_rate).abs().lt(1e-12).all(),
        "text_and_source_hash_coincide": d.loc[d.has_text.eq(1),"html_sha256"].notna().all(),
    }
    result = {"rows": len(d), "companies": int(d.cik.nunique()), "text_rows": int(d.has_text.sum()),
              "checks": {key: bool(value) for key, value in checks.items()},
              "revenue_source_tags": d.revenue_tag.value_counts().to_dict(),
              "capex_source_tags": d.capex_tag.value_counts().to_dict()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if not all(checks.values()):
        raise ValueError("Panel quality failed: " + ", ".join(k for k,v in checks.items() if not v))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=Path("data/processed/panel.csv"))
    p.add_argument("--out", type=Path, default=Path("reports/data_quality.json"))
    a = p.parse_args()
    print(json.dumps(run(a.panel, a.out), indent=2))


if __name__ == "__main__":
    main()
