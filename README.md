# `evalkeel`

**Part of the [Keel Data Stack](https://github.com/robertguss/keel-stack) (Tool #16).**

Zero-Python, single-binary Rust **model-agnostic ML backtest, calibration, conformal coverage, subgroup fairness parity & tamper-evident HTML Model Card engine** (DX + AX).

## Features
- **Binary Classification (`evalkeel binary`):**
  - Exact **ROC-AUC** (with exact tie handling), **PR-AUC / Average Precision** (exact step-function matching `scikit-learn`), **Brier Score**, **Brier Skill Score**, **Log-Loss**, and **10-bin Expected Calibration Error (ECE)** + full reliability curve.
  - **Operational Top-K% Capture:** Precision, Recall, and Lift at Top 5%, 10%, 20% (customizable via `--top-k`).
  - **Head-to-Head Baseline Comparison (`--compare`):** Exact `Δ PR-AUC`, `Δ ROC-AUC`, `Δ Brier` against heuristic rules or prior model versions.
  - **Subgroup Fairness Parity (`--group`):** Per-group base rate, mean score, ROC-AUC, PR-AUC, ECE, TPR @ threshold, plus max parity gaps (`max_tpr_gap`, `max_ece_gap`).
- **Regression & Capacity Forecasting (`evalkeel regress`):**
  - Exact **MAE**, **RMSE**, **MedAE**, **WMAPE**, **R²**, and signed **Bias**.
  - **Conformal Interval Coverage (`--p10`, `--p90`):** Empirical coverage %, mean interval width, and under/over miss rates.
  - **Operational Capacity / Section Sizing (`--cap 25`):** Exact section-count match %, within-±1 section %, under-staffed rate, and over-staffed rate.
- **CI Quality Gates & Tamper-Evident HTML Model Cards (`--html`):**
  - Gate model promotions in CI (`--min-pr-auc`, `--max-brier`, `--max-ece`, `--max-wmape`, `--min-coverage`, `--max-group-tpr-gap`) with exit code `1` on regression.
  - Emit zero-CDN offline HTML Model Cards with embedded SVG calibration curves and SHA-256 dataset+metric receipts.

## Quick Start

```bash
evalkeel binary retention_oof.parquet --y had_dfw --p risk_prob \
  --compare day21_rule --group program --min-pr-auc 0.55 --html retention_card.html

evalkeel regress demand_oof.csv --y actual_enroll --pred p50 \
  --p10 p10 --p90 p90 --cap 25 --compare last_year --html demand_card.html
```

## Install

```bash
cargo install --git https://github.com/robertguss/evalkeel --locked
```

## License
Dual-licensed under MIT or Apache-2.0.
