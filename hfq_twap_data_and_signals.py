"""Data preparation, factor construction, and signal utilities.

The public repository intentionally excludes the original minute-level dataset.
Use ``generate_demo_data.py`` to create a compatible synthetic dataset.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

# 默认读取仓库内的数据目录；真实研究数据不进入 Git 版本库。
DATA_DIR = Path("data/market_data_01min")
RESULTS_DIR = Path("results")
SYMBOLS = [
    "SP", "P", "BU", "CU", "V", "FG", "CF", "SC", "AU", "AG",
    "MA", "PG", "RM", "RU", "TA", "EB", "SR", "FU", "I", "SN",
    "OI", "ZN", "Y", "SA", "M", "RB", "EG", "PP", "AL", "NI", "HC",
]

# 把分钟级 TWAP 聚合成日频开、高、低、收。这里的“开收”都来自 TWAP 序列。
def load_daily_data(symbols: list[str], data_dir: Path) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        feather_path = data_dir / f"{symbol}_main.feather"
        csv_path = data_dir / f"{symbol}_main.csv"

        # Feather 适合真实研究数据；CSV 让公开演示无需额外二进制依赖也能运行。
        if feather_path.exists():
            df = pd.read_feather(feather_path, columns=["tradeDate", "hfq_twap"])
        elif csv_path.exists():
            df = pd.read_csv(csv_path, usecols=["tradeDate", "hfq_twap"])
        else:
            raise FileNotFoundError(
                f"Missing input for {symbol}: expected {feather_path.name} or {csv_path.name}"
            )
        daily = (
            df.groupby("tradeDate", sort=True)
            .agg(
                open_twap=("hfq_twap", "first"),
                close_twap=("hfq_twap", "last"),
                high_twap=("hfq_twap", "max"),
                low_twap=("hfq_twap", "min"),
            )
            .reset_index()
        )
        daily["symbol"] = symbol
        frames.append(daily)

    out = pd.concat(frames, ignore_index=True)
    out["tradeDate"] = pd.to_datetime(out["tradeDate"])
    return out.sort_values(["symbol", "tradeDate"]).reset_index(drop=True)

# 按品种分别计算趋势、波动率压缩、日内振幅和标准分数四类因子。
def add_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    by_symbol = out.groupby("symbol")

    out["ret1"] = by_symbol["close_twap"].pct_change()
    out["ma10"] = by_symbol["close_twap"].transform(lambda s: s.rolling(10).mean())
    out["ma40"] = by_symbol["close_twap"].transform(lambda s: s.rolling(40).mean())
    out["trend_factor"] = out["ma10"] / out["ma40"] - 1

    out["vol5"] = by_symbol["ret1"].transform(lambda s: s.rolling(5).std())
    out["vol20"] = by_symbol["ret1"].transform(lambda s: s.rolling(20).std())
    out["compression_factor"] = out["vol5"] / out["vol20"]

    out["range_factor"] = (out["high_twap"] - out["low_twap"]) / out["open_twap"]

    out["zscore_factor"] = by_symbol["close_twap"].transform(
        lambda s: (s - s.rolling(20).mean()) / s.rolling(20).std()
    )

    # 下一交易日开盘到收盘的收益，供回测阶段与滞后信号对齐。
    out["trade_return"] = by_symbol["close_twap"].shift(-1) / by_symbol["open_twap"].shift(-1) - 1
    return out

# 将四个因子条件合成为 -1 / 0 / 1 的空仓、观望、多仓信号。
def build_signals(
    df: pd.DataFrame,
    trend_threshold: float,
    compression_threshold: float,
    range_threshold: float,
    zscore_threshold: float,
    signal_lag_days: int,
) -> pd.DataFrame:
    out = df.copy()
    by_symbol = out.groupby("symbol")

    src_trend = by_symbol["trend_factor"].shift(signal_lag_days)
    src_compression = by_symbol["compression_factor"].shift(signal_lag_days)
    src_range = by_symbol["range_factor"].shift(signal_lag_days)
    src_zscore = by_symbol["zscore_factor"].shift(signal_lag_days)

    long_signal = (
        (src_trend > trend_threshold)
        & (src_compression < compression_threshold)
        & (src_range > range_threshold)
        & (src_zscore > zscore_threshold)
    )
    short_signal = (
        (src_trend < -trend_threshold)
        & (src_compression < compression_threshold)
        & (src_range > range_threshold)
        & (src_zscore < -zscore_threshold)
    )

    out["signal"] = 0
    out.loc[long_signal, "signal"] = 1
    out.loc[short_signal, "signal"] = -1

    out["source_trend"] = src_trend
    out["source_compression"] = src_compression
    out["source_range"] = src_range
    out["source_zscore"] = src_zscore
    return out


def run_backtest(
    df: pd.DataFrame,
    commission_rate: float,
    risk_free_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    signal_df = df.copy()
    signal_df.loc[signal_df["trade_return"].isna(), "signal"] = 0
    signal_df["gross_symbol_return"] = signal_df["signal"] * signal_df["trade_return"]

    # 换手次数乘以费率，得到简化后的交易成本。
    turnover = signal_df.groupby("symbol")["signal"].diff().abs().fillna(0.0)
    signal_df["turnover"] = turnover.where(signal_df["trade_return"].notna(), 0.0)

    daily_rows = []
    for trade_date, group in signal_df.groupby("tradeDate", sort=True):
        active = group[(group["signal"] != 0) & group["gross_symbol_return"].notna()].copy()
        if active.empty:
            daily_rows.append(
                {
                    "tradeDate": trade_date,
                    "active_symbols": 0,
                    "gross_return": 0.0,
                    "cost": group["turnover"].mean() * commission_rate,
                }
            )
            continue
        # 每个交易日只保存组合层面的汇总，不保存账户或投资者信息。
        daily_rows.append(
            {
                "tradeDate": trade_date,
                "active_symbols": int(len(active)),
                "gross_return": float(active["gross_symbol_return"].mean()),
                "cost": float(group["turnover"].mean() * commission_rate),
            }
        )

    daily_df = pd.DataFrame(daily_rows).sort_values("tradeDate").reset_index(drop=True)

    daily_df["net_return"] = daily_df["gross_return"] - daily_df["cost"]

    daily_df["equity"] = (1.0 + daily_df["net_return"]).cumprod()

    daily_returns = daily_df["net_return"]
    # 计算常见的年化收益、波动率、Sharpe 和回撤指标。
    annualized_return = float(daily_returns.mean() * 252)
    annualized_vol = float(daily_returns.std(ddof=1) * np.sqrt(252))
    sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0.0

    rolling_peak = daily_df["equity"].cummax()
    drawdown = daily_df["equity"] / rolling_peak - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    active_trades = signal_df[signal_df["signal"] != 0].copy()
    win_rate = float((active_trades["gross_symbol_return"] > 0).mean()) if len(active_trades) else 0.0
    wins = active_trades.loc[active_trades["gross_symbol_return"] > 0, "gross_symbol_return"]
    losses = active_trades.loc[active_trades["gross_symbol_return"] < 0, "gross_symbol_return"]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    win_loss_ratio = avg_win / abs(avg_loss) if avg_loss < 0 else 0.0

    summary = {
        "start": str(daily_df["tradeDate"].iloc[0].date()),
        "end": str(daily_df["tradeDate"].iloc[-1].date()),
        "days": int(len(daily_df)),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(calmar),
        "total_return": float(daily_df["equity"].iloc[-1] - 1.0),
        "average_active_symbols": float(daily_df["active_symbols"].mean()),
        "number_of_symbol_trades": int(len(active_trades)),
        "win_rate": win_rate,
        "win_loss_ratio": float(win_loss_ratio),
        "average_symbol_return": float(active_trades["gross_symbol_return"].mean()) if len(active_trades) else 0.0,
        "risk_free_rate": float(risk_free_rate),
        "commission_rate": float(commission_rate),
    }
    return daily_df, signal_df, summary


def plot_equity_curve(daily_df: pd.DataFrame, output_path: Path) -> None:
    # 绘图是可选功能，因此只在真正调用时导入 matplotlib。
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(daily_df["tradeDate"], daily_df["equity"], color="darkgreen", linewidth=1.2)
    ax.set_title("Multi-Factor HFQ TWAP Futures Portfolio")
    ax.set_xlabel("Trade Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
