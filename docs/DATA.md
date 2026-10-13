# Data dictionary and lineage

| File | Description | Git? |
| --- | --- | --- |
| `configs/universe.csv` | Pinned 100-name source universe, two sectors | Yes |
| `configs/snapshot_manifest.json` | SHA-256 and size of each of 408 downloaded input files | Yes |
| `data/raw/facts/<CIK>.json.gz` | SEC Company Facts JSON | Separate source archive |
| `data/raw/filings/<CIK>.json` | SEC filing submissions and original 10-K identifiers | Separate source archive |
| `data/raw/html/<accession>.html.gz` | Cached original 10-K text for the selected text firms | Separate source archive |
| `data/processed/panel.csv` | Eligible source/target pairs, numerical values and excerpts | Generated locally |
| `data/processed/evidence.jsonl` | Source offsets and selected passages | Generated locally |
| `reports/data_quality.json` | Structural checks and selected tag counts | Yes |
| `reports/filing_reconciliation.json` | Original inline XBRL spot checks | Yes |
| `reports/metrics.json`, `reports/test_predictions.csv` | Evaluation metadata and held-out row-level forecasts | Yes |

A row key `id` is CIK plus source fiscal end. `known_at`, `source_accession` and `source_url` identify the historical current filing. `source_sha256` hashes the cached compressed Company Facts response; `html_sha256` hashes the cached compressed source filing HTML when present. `target_fiscal_end`, `target_known_at`, `target_accession` identify the following filing, available for audit and scoring only. `revenue`, `capex`, `repurchase`, `dividend`, `cfo` and `rnd` are original-accession annual USD source facts; `*_tag` gives the selected US-GAAP taxonomy name. `next_*` fields are **future labels**, not model features. `capex_rate` is source capex/revenue; `target_capex_rate` is next-year capex/revenue. Other `*_rate` fields are source history. Blank amounts and ratios are missing, never inferred zero. `text` contains selected visible filing excerpts; `has_text` flags cached original HTML. Missing text is not neutral or reassuring language.

The snapshot was acquired in October 2026. Reproduction validates its manifest before modeling. A fresh download can alter data and results and should be treated as a separate study.
