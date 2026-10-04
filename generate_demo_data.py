"""Generate deterministic synthetic minute-style data for a public demo.

This file does not reproduce or approximate any private source dataset.  It only
creates the two columns expected by the backtest: ``tradeDate`` and ``hfq_twap``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hfq_twap_data_and_signals import DATA_DIR, SYMBOLS


def make_symbol_frame(symbol: str, seed: int) -> pd.DataFrame:
    """Create four synthetic intraday observations for every business day."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-10-01", "2023-12-29")

    # 使用平缓随机游走构造价格；固定种子保证每次运行结果一致。
    daily_shocks = rng.normal(loc=0.00015, scale=0.012, size=len(dates))
    daily_close = (80.0 + seed) * np.exp(np.cumsum(daily_shocks))

    rows: list[dict[str, object]] = []
    for trade_date, close_price in zip(dates, daily_close, strict=True):
        intraday_noise = rng.normal(0.0, 0.003, size=4)
        intraday_path = close_price * np.exp(np.cumsum(intraday_noise))
        for value in intraday_path:
            rows.append({"tradeDate": trade_date, "hfq_twap": float(value)})

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Feather files for the demo backtest.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed, symbol in enumerate(SYMBOLS, start=1):
        frame = make_symbol_frame(symbol, seed)
        output_path = args.output_dir / f"{symbol}_main.csv"
        frame.to_csv(output_path, index=False)
        print(f"created {output_path}")


if __name__ == "__main__":
    main()
