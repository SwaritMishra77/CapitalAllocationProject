import gzip
import json

from capalloc.panel import annual_fact, first_filings, make_panel, passages


def filing(accn, end, filed):
    return {"accessionNumber": accn, "reportDate": end, "filingDate": filed, "form": "10-K"}


def test_original_filing_excludes_later_comparative_restatement():
    old = filing("original", "2022-12-31", "2023-02-01")
    facts = {"us-gaap": {"PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
        {"accn": "original", "end": "2022-12-31", "start": "2022-01-01", "filed": "2023-02-01", "val": 100},
        {"accn": "later", "end": "2022-12-31", "start": "2022-01-01", "filed": "2024-02-01", "val": 800},
    ]}}}}
    assert annual_fact(facts, "capex", old)[0] == 100
    assert annual_fact(facts, "repurchase", old)[0] is None


def test_first_10k_per_report_date_only():
    records = [filing("amended", "2022-12-31", "2023-03-01"),
               filing("original", "2022-12-31", "2023-02-01"),
               {**filing("10ka", "2022-12-31", "2023-02-10"), "form": "10-K/A"}]
    assert [r["accessionNumber"] for r in first_filings(records)] == ["original"]


def test_revenue_picks_consolidated_sales_over_small_other_revenue():
    f = filing("a", "2013-12-31", "2014-02-01")
    def value(n):
        return {"accn": "a", "end": "2013-12-31", "start": "2013-01-01", "filed": "2014-02-01", "val": n}
    facts = {"us-gaap": {"Revenues": {"units": {"USD": [value(80)]}},
                         "SalesRevenueGoodsNet": {"units": {"USD": [value(4300)]}}}}
    assert annual_fact(facts, "revenue", f) == (4300., "SalesRevenueGoodsNet")


def test_panel_links_next_year_without_treating_missing_buyback_as_zero(tmp_path):
    raw = tmp_path / "raw"
    for sub in ("facts", "filings", "html"):
        (raw / sub).mkdir(parents=True)
    cik = "0000000123"
    (raw / "universe_resolved.json").write_text(json.dumps([{"cik": cik, "ticker": "ABC", "sector": "retail"}]))
    fs = [filing("a", "2022-12-31", "2023-02-01"), filing("b", "2023-12-31", "2024-02-01")]
    (raw / "filings" / f"{cik}.json").write_text(json.dumps(fs))
    def entry(accn, start, end, filed, val):
        return {"accn": accn, "start": start, "end": end, "filed": filed, "val": val}
    facts = {"cik": 123, "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [entry("a", "2022-01-01", "2022-12-31", "2023-02-01", 1000),
                                                entry("b", "2023-01-01", "2023-12-31", "2024-02-01", 2000)]}},
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
            entry("a", "2022-01-01", "2022-12-31", "2023-02-01", 100),
            entry("b", "2023-01-01", "2023-12-31", "2024-02-01", 300)]}},
    }}}
    with gzip.open(raw / "facts" / f"{cik}.json.gz", "wt") as f:
        json.dump(facts, f)
    import pandas as pd
    make_panel(raw, tmp_path / "out")
    row = pd.read_csv(tmp_path / "out" / "panel.csv").iloc[0]
    assert row.capex_rate == .1 and row.target_capex_rate == .15
    assert pd.isna(row.repurchase) and pd.isna(row.next_repurchase)
    assert row.source_accession == "a" and row.target_accession == "b"


def test_passages_have_source_offsets(tmp_path):
    path = tmp_path / "a.html.gz"
    with gzip.open(path, "wt") as f:
        f.write("<html><script>capital expenditures hidden</script><p>We expect capital expenditures "
                "of about $300 million next year as we build more stores and increase our capacity.</p></html>")
    text, evidence = passages(path)
    assert "capital expenditures" in text
    assert len(evidence) == 1 and evidence[0]["offset_end"] > evidence[0]["offset_start"]
