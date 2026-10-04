# Methodology and disclosure

## Data pipeline

For each futures symbol, intraday adjusted TWAP observations are grouped by
trading date. The first, last, maximum, and minimum observations become a daily
OHLC-style TWAP series. The public demo uses synthetic values with the same
schema; it does not contain or reconstruct the original research dataset.

## Factors

Four simple factors are calculated independently for every symbol:

1. **Trend:** 10-day moving average divided by the 40-day moving average, minus 1.
2. **Compression:** 5-day return volatility divided by 20-day return volatility.
3. **Range:** daily high-low TWAP range divided by opening TWAP.
4. **Z-score:** close TWAP relative to its rolling 20-day mean and standard deviation.

A long position requires all four positive-side conditions. A short position uses
the symmetric negative-side conditions for trend and z-score while retaining the
same compression and range filters.

## Validation design

- Parameters are searched inside the training sample only.
- Each sector shares one parameter set.
- When possible, both halves of the training year must have positive returns.
- Sectors are ranked by training Sharpe ratio, then by return metrics.
- The selected sectors are evaluated on a separate test sample.

This separation reduces, but does not eliminate, overfitting risk. A stronger
follow-up would use walk-forward validation, purged cross-validation, broader
cost stress tests, and multiple market regimes.

## Portfolio construction and costs

Active symbols are combined with equal weights or inverse-volatility weights.
The backtest deducts a configurable commission rate and optional slippage from
signal turnover. These are simplified proportional-cost assumptions.

## Public-repository sanitization

This portfolio edition excludes:

- organization-specific names and source-material wording;
- personal names, contact information, credentials, and local absolute paths;
- original market-data files and detailed signal/trade CSV files;
- duplicate archives, operating-system metadata, and LaTeX build artefacts;
- the private report source and compiled PDF.

Only general research code, documentation, and aggregate charts are published.
