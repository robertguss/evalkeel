---
name: evalkeel
description: Zero-Python Rust model-agnostic ML evaluation, calibration, conformal coverage, subgroup fairness parity & tamper-evident HTML Model Card CLI.
---

# `evalkeel` — Model-Agnostic ML Backtest, Calibration & Fairness Receipt Engine

## When to Use
- Evaluate any binary classifier (`CSV`, `TSV`, `JSONL`, `Parquet`) for **ROC-AUC**, **PR-AUC (Average Precision)**, **Brier score**, **Log-Loss**, **ECE** + 10-bin reliability curve, and **Precision/Recall/Lift @ Top-K%**.
- Evaluate regression and demand/capacity forecasters for **MAE**, **RMSE**, **WMAPE**, **R²**, **Conformal P10–P90 empirical coverage %**, and **section/capacity staffing accuracy** (`--cap 25`).
- Compare candidate models head-to-head against a baseline (`--compare baseline_col`), slice subgroup fairness parity (`--group program`), and gate CI merges with threshold flags (`--min-pr-auc`, `--max-brier`, `--max-wmape`).
- Emit a tamper-evident SHA-256 offline HTML Model Card (`--html model_card.html`).

## Command Matrix

| Task | Command |
| --- | --- |
| Binary classifier evaluation + fairness + HTML card | `evalkeel binary preds.parquet --y had_dfw --p risk_prob --compare baseline_prob --group program --html model_card.html` |
| CI binary regression gate | `evalkeel binary preds.csv --y y --p p --min-pr-auc 0.55 --max-brier 0.14 --max-ece 0.05 --format agent` |
| Regression + Conformal P10–P90 + section staffing | `evalkeel regress demand.csv --y actual --pred p50 --p10 p10 --p90 p90 --cap 25 --compare naive_pred --html demand_card.html` |
| CI regression gate | `evalkeel regress demand.parquet --y actual --pred p50 --max-wmape 0.18 --min-coverage 0.72 --format agent` |
| Print / install this skill | `evalkeel skill` / `evalkeel skill --install` |

## Exit Codes
- `0`: Evaluation completed and all configured quality/fairness gates passed.
- `1`: One or more CI quality, calibration, coverage, or fairness gates failed.
- `2`: I/O, column, or invocation error.
