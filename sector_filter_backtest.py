"""Train-only sector selection and out-of-sample futures backtest.

The script is research code, not a live-trading system or investment advice.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from hfq_twap_data_and_signals import DATA_DIR, RESULTS_DIR, SYMBOLS, add_factors, build_signals, load_daily_data


SECTORS = {
    "metals": ["CU", "AL", "ZN", "NI", "SN", "AU", "AG"],
    "black": ["I", "RB", "HC", "SA"],
    "energychem": ["BU", "SC", "PG", "FU", "MA", "TA", "EB", "EG", "V", "PP", "FG", "RU", "SP"],
    "agri": ["P", "CF", "RM", "SR", "OI", "Y", "M"],
}


def build_sector_signal_df(
    base_df: pd.DataFrame,
    symbols: list[str],
    params: dict[str, float | int],
    commission_rate: float,
    slippage_rate: float = 0.0,
) -> pd.DataFrame:
    """Create lagged signals and transaction-cost-adjusted returns for one sector."""
    df = base_df[base_df["symbol"].isin(symbols)].copy()
    df = build_signals(
        df=df,
        trend_threshold=float(params["trend_threshold"]),
        compression_threshold=float(params["compression_threshold"]),
        range_threshold=float(params["range_threshold"]),
        zscore_threshold=float(params["zscore_threshold"]),
        signal_lag_days=int(params["signal_lag_days"]),
    )
    df["execution_date"] = df.groupby("symbol")["tradeDate"].shift(-1)
    df = df[df["execution_date"].notna()].copy()
    df["execution_date"] = pd.to_datetime(df["execution_date"])
    df.loc[df["trade_return"].isna(), "signal"] = 0
    df["gross_symbol_return"] = df["signal"] * df["trade_return"]
    df["turnover"] = (
        df.groupby("symbol")["signal"].diff().abs().fillna(0.0).where(df["trade_return"].notna(), 0.0)
    )
    df["commission_cost"] = df["turnover"] * commission_rate
    df["slippage_cost"] = df["turnover"] * slippage_rate
    df["cost"] = df["commission_cost"] + df["slippage_cost"]
    df["net_symbol_return"] = df["gross_symbol_return"] - df["cost"]
    df["holding_days"] = np.where(df["signal"] != 0, 1.0, 0.0)
    return df


def compute_portfolio_weights(active: pd.DataFrame, weight_method: str, max_symbol_weight: float) -> pd.Series:
    """Return normalized equal or inverse-volatility weights for active symbols."""
    if active.empty:
        return pd.Series(dtype=float)

    if weight_method == "equal":
        raw = pd.Series(1.0, index=active.index)
    elif weight_method == "inverse_vol":
        vol = active["vol20"].replace([np.inf, -np.inf], np.nan)
        raw = 1.0 / vol.where(vol > 0)
        raw = raw.replace([np.inf, -np.inf], np.nan)
        if raw.notna().sum() == 0:
            raw = pd.Series(1.0, index=active.index)
        else:
            raw = raw.fillna(raw[raw.notna()].median())
    else:
        raise ValueError(f"Unsupported weight method: {weight_method}")

    weights = raw / raw.sum()
    if max_symbol_weight >= 1.0 or len(weights) <= 1:
        return weights

    cap = max(max_symbol_weight, 1.0 / len(weights))
    capped = weights.copy()
    for _ in range(len(capped)):
        over_cap = capped > cap
        if not over_cap.any():
            break
        capped.loc[over_cap] = cap
        remaining = ~over_cap
        remaining_weight = 1.0 - capped.loc[over_cap].sum()
        if not remaining.any() or remaining_weight <= 0:
            break
        capped.loc[remaining] = weights.loc[remaining] / weights.loc[remaining].sum() * remaining_weight
    return capped / capped.sum()


def summarize_active_only_period(
    signal_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    risk_free_rate: float,
    weight_method: str = "inverse_vol",
    max_symbol_weight: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    """Aggregate symbol-level returns into a daily portfolio and summary statistics."""
    period_df = signal_df[
        (signal_df["execution_date"] >= start) & (signal_df["execution_date"] <= end)
    ].copy()

    daily_rows = []
    active_trade_frames = []
    for execution_date, group in period_df.groupby("execution_date", sort=True):
        active = group[(group["signal"] != 0) & group["net_symbol_return"].notna()].copy()
        if len(active):
            active["portfolio_weight"] = compute_portfolio_weights(active, weight_method, max_symbol_weight)
            active["weighted_gross_return"] = active["portfolio_weight"] * active["gross_symbol_return"]
            active["weighted_cost"] = active["portfolio_weight"] * active["cost"]
            active["weighted_net_return"] = active["portfolio_weight"] * active["net_symbol_return"]
            active_trade_frames.append(active)
        daily_rows.append(
            {
                "execution_date": execution_date,
                "active_symbols": int(len(active)),
                "gross_return": float(active["weighted_gross_return"].sum()) if len(active) else 0.0,
                "cost": float(active["weighted_cost"].sum()) if len(active) else 0.0,
                "net_return": float(active["weighted_net_return"].sum()) if len(active) else 0.0,
            }
        )

    daily_df = pd.DataFrame(daily_rows).sort_values("execution_date").reset_index(drop=True)
    daily_df["equity"] = (1.0 + daily_df["net_return"]).cumprod() if len(daily_df) else pd.Series(dtype=float)

    daily_returns = daily_df["net_return"] if len(daily_df) else pd.Series(dtype=float)
    annualized_return = float(daily_returns.mean() * 252) if len(daily_returns) else 0.0
    annualized_volatility = float(daily_returns.std(ddof=1) * np.sqrt(252)) if len(daily_returns) > 1 else 0.0
    sharpe_ratio = (
        (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0.0
    )
    
    rolling_peak = daily_df["equity"].cummax() if len(daily_df) else pd.Series(dtype=float)
    drawdown = daily_df["equity"] / rolling_peak - 1.0 if len(daily_df) else pd.Series(dtype=float)
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    active_trades = pd.concat(active_trade_frames, ignore_index=True) if active_trade_frames else period_df.iloc[0:0].copy()
    wins = active_trades.loc[active_trades["net_symbol_return"] > 0, "net_symbol_return"]
    losses = active_trades.loc[active_trades["net_symbol_return"] < 0, "net_symbol_return"]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0

    summary = {
        "start": str(daily_df["execution_date"].iloc[0].date()) if len(daily_df) else "",
        "end": str(daily_df["execution_date"].iloc[-1].date()) if len(daily_df) else "",
        "days": int(len(daily_df)),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(calmar_ratio),
        "total_return": float(daily_df["equity"].iloc[-1] - 1.0) if len(daily_df) else 0.0,
        "average_active_symbols": float(daily_df["active_symbols"].mean()) if len(daily_df) else 0.0,
        "number_of_transactions": int(len(active_trades)),
        "win_rate": float((active_trades["net_symbol_return"] > 0).mean()) if len(active_trades) else 0.0,
        "win_loss_ratio": float(avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0,
        "average_holding_time": float(active_trades["holding_days"].mean()) if len(active_trades) else 0.0,
        "average_profit_loss_per_transaction": float(active_trades["net_symbol_return"].mean()) if len(active_trades) else 0.0,
        "weight_method": weight_method,
        "max_symbol_weight": float(max_symbol_weight),
    }
    return daily_df, active_trades, summary


def print_summary_block(title: str, summary: dict[str, object]) -> None:
    print(title)
    print(f"  Period: {summary['start']} -> {summary['end']}")
    print(f"  Days: {summary['days']}")
    print(f"  Annualized return: {summary['annualized_return']:.2%}")
    print(f"  Annualized volatility: {summary['annualized_volatility']:.2%}")
    print(f"  Sharpe ratio: {summary['sharpe_ratio']:.4f}")
    print(f"  Max drawdown: {summary['max_drawdown']:.2%}")
    print(f"  Calmar ratio: {summary['calmar_ratio']:.4f}")
    print(f"  Total return: {summary['total_return']:.2%}")
    print(f"  Average active symbols: {summary['average_active_symbols']:.2f}")
    print(f"  Number of transactions: {summary['number_of_transactions']}")
    print(f"  Win rate: {summary['win_rate']:.2%}")
    print(f"  Win/loss ratio: {summary['win_loss_ratio']:.4f}")
    print(f"  Average holding time: {summary['average_holding_time']:.2f} day(s)")
    print(f"  Average profit/loss per transaction: {summary['average_profit_loss_per_transaction']:.4%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sector-wise parameter training with train-only sector filter for hfq_twap futures portfolio."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--train-start", type=str, default="2022-01-01")
    parser.add_argument("--train-end", type=str, default="2022-12-31")
    parser.add_argument("--test-start", type=str, default="2023-01-01")
    parser.add_argument("--test-end", type=str, default="2023-12-31")
    parser.add_argument("--commission-rate", type=float, default=0.0002)
    parser.add_argument("--slippage-rate", type=float, default=0.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.02)
    parser.add_argument("--top-k-sectors", type=int, default=2)
    parser.add_argument("--weight-method", choices=["equal", "inverse_vol"], default="inverse_vol")
    parser.add_argument("--max-symbol-weight", type=float, default=1.0)
    parser.add_argument(
        "--sector-ranking-output",
        type=Path,
        default=RESULTS_DIR / "multifactor_sector_filter_sector_ranking.csv",
    )
    parser.add_argument(
        "--selected-sectors-output",
        type=Path,
        default=RESULTS_DIR / "multifactor_sector_filter_selected_sectors.csv",
    )
    parser.add_argument(
        "--train-portfolio-output",
        type=Path,
        default=RESULTS_DIR / "multifactor_sector_filter_train_daily.csv",
    )
    parser.add_argument(
        "--test-portfolio-output",
        type=Path,
        default=RESULTS_DIR / "multifactor_sector_filter_test_daily.csv",
    )
    parser.add_argument(
        "--test-trades-output",
        type=Path,
        default=RESULTS_DIR / "multifactor_sector_filter_test_trades.csv",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)

    base_df = add_factors(load_daily_data(SYMBOLS, args.data_dir))
    # 只在训练期内搜索参数，避免测试集信息反向渗入参数选择。
    grid = list(
        product(
            [0.003, 0.005, 0.007],
            [0.9, 1.0],
            [0.010, 0.012],
            [0.0, 0.5, 1.0],
            [1],
        )
    )

    sector_rows = []
    best_sector_payloads: dict[str, dict[str, object]] = {}

    for sector_name, sector_symbols in SECTORS.items():
        candidates = []
        for trend_threshold, compression_threshold, range_threshold, zscore_threshold, signal_lag_days in grid:
            params = {
                "trend_threshold": trend_threshold,
                "compression_threshold": compression_threshold,
                "range_threshold": range_threshold,
                "zscore_threshold": zscore_threshold,
                "signal_lag_days": signal_lag_days,
            }
            signal_df = build_sector_signal_df(
                base_df,
                sector_symbols,
                params,
                args.commission_rate,
                args.slippage_rate,
            )
            train_daily, train_trades, train_summary = summarize_active_only_period(
                signal_df,
                train_start,
                train_end,
                args.risk_free_rate,
                args.weight_method,
                args.max_symbol_weight,
            )
            test_daily, test_trades, test_summary = summarize_active_only_period(
                signal_df,
                test_start,
                test_end,
                args.risk_free_rate,
                args.weight_method,
                args.max_symbol_weight,
            )

            first_half_return = train_daily.loc[
                pd.to_datetime(train_daily["execution_date"]) < pd.Timestamp("2022-07-01"), "net_return"
            ].sum()
            second_half_return = train_daily.loc[
                pd.to_datetime(train_daily["execution_date"]) >= pd.Timestamp("2022-07-01"), "net_return"
            ].sum()

            payload = {
                "params": params,
                "signal_df": signal_df,
                "train_daily": train_daily,
                "train_trades": train_trades,
                "train_summary": train_summary,
                "test_daily": test_daily,
                "test_trades": test_trades,
                "test_summary": test_summary,
                "first_half_return": float(first_half_return),
                "second_half_return": float(second_half_return),
            }
            candidates.append(payload)

            sector_rows.append(
                {
                    "sector": sector_name,
                    **params,
                    "train_sharpe": train_summary["sharpe_ratio"],
                    "train_ann_return": train_summary["annualized_return"],
                    "train_total_return": train_summary["total_return"],
                    "train_max_dd": train_summary["max_drawdown"],
                    "train_h1_return": float(first_half_return),
                    "train_h2_return": float(second_half_return),
                    "test_sharpe": test_summary["sharpe_ratio"],
                    "test_ann_return": test_summary["annualized_return"],
                    "test_total_return": test_summary["total_return"],
                    "test_max_dd": test_summary["max_drawdown"],
                }
            )

        valid_candidates = [
            p for p in candidates if p["first_half_return"] > 0 and p["second_half_return"] > 0
        ]
        if not valid_candidates:
            valid_candidates = candidates

        best_payload = sorted(
            valid_candidates,
            key=lambda p: (
                p["train_summary"]["sharpe_ratio"],
                p["train_summary"]["annualized_return"],
                p["train_summary"]["total_return"],
            ),
            reverse=True,
        )[0]
        best_sector_payloads[sector_name] = best_payload

    sector_ranking_df = (
        pd.DataFrame(
            [
                {
                    "sector": sector_name,
                    **payload["params"],
                    "train_sharpe": payload["train_summary"]["sharpe_ratio"],
                    "train_ann_return": payload["train_summary"]["annualized_return"],
                    "train_total_return": payload["train_summary"]["total_return"],
                    "train_max_dd": payload["train_summary"]["max_drawdown"],
                    "train_h1_return": payload["first_half_return"],
                    "train_h2_return": payload["second_half_return"],
                    "test_sharpe": payload["test_summary"]["sharpe_ratio"],
                    "test_ann_return": payload["test_summary"]["annualized_return"],
                    "test_total_return": payload["test_summary"]["total_return"],
                    "test_max_dd": payload["test_summary"]["max_drawdown"],
                }
                for sector_name, payload in best_sector_payloads.items()
            ]
        )
        .sort_values(["train_sharpe", "train_ann_return"], ascending=[False, False])
        .reset_index(drop=True)
    )

    selected_sectors_df = sector_ranking_df.head(args.top_k_sectors).copy()
    selected_sector_names = selected_sectors_df["sector"].tolist()

    selected_signal_df = pd.concat(
        [best_sector_payloads[sector_name]["signal_df"] for sector_name in selected_sector_names],
        ignore_index=True,
    ).sort_values(["execution_date", "symbol"]).reset_index(drop=True)

    train_portfolio_daily, train_portfolio_trades, train_portfolio_summary = summarize_active_only_period(
        selected_signal_df,
        train_start,
        train_end,
        args.risk_free_rate,
        args.weight_method,
        args.max_symbol_weight,
    )
    test_portfolio_daily, test_portfolio_trades, test_portfolio_summary = summarize_active_only_period(
        selected_signal_df,
        test_start,
        test_end,
        args.risk_free_rate,
        args.weight_method,
        args.max_symbol_weight,
    )

    pd.DataFrame(sector_rows).to_csv(args.sector_ranking_output, index=False)
    selected_sectors_df.to_csv(args.selected_sectors_output, index=False)
    train_portfolio_daily.to_csv(args.train_portfolio_output, index=False)
    test_portfolio_daily.to_csv(args.test_portfolio_output, index=False)
    test_portfolio_trades.to_csv(args.test_trades_output, index=False)

    print("Assumption")
    print("  Training period: 2022 full year.")
    print("  Test period: 2023 full year.")
    print()
    print("Strategy Logic")
    print("  Step 1: Split the universe into sectors.")
    print("  Step 2: For each sector, search one shared parameter set using only 2022.")
    print("  Step 3: Require 2022H1 and 2022H2 to both be positive when possible.")
    print("  Step 4: Rank sectors by 2022 train Sharpe and keep the top sectors only.")
    print("  Step 5: On each day, allocate only to symbols with active signals inside the selected sectors.")
    print(f"  Step 6: Active symbols are combined with {args.weight_method} weights.")
    print(f"  Slippage rate: {args.slippage_rate:.4%}")
    print()
    print("Signal Logic")
    print("  open_twap and close_twap are execution prices, not separate signals.")
    print("  A long signal requires all four lagged conditions simultaneously.")
    print("  A short signal uses the exact symmetric opposite conditions.")
    print("  signal_lag_days = 1 means the T execution-day trade uses T-2 factor values.")
    print()
    print("Parameter Grid Per Sector")
    print(f"  combinations: {len(grid)}")
    print("  trend_threshold: [0.003, 0.005, 0.007]")
    print("  compression_threshold: [0.9, 1.0]")
    print("  range_threshold: [0.010, 0.012]")
    print("  zscore_threshold: [0.0, 0.5, 1.0]")
    print("  signal_lag_days: [1]")
    print()
    print("Sector Ranking By Train Sharpe")
    print(sector_ranking_df.to_string(index=False))
    print()
    print(f"Selected sectors (top {args.top_k_sectors} by train Sharpe)")
    print(selected_sectors_df.to_string(index=False))
    print()
    print_summary_block("In-Sample Portfolio Summary", train_portfolio_summary)
    print()
    print_summary_block("Out-of-Sample Portfolio Summary", test_portfolio_summary)
    print()
    print("Out-of-Sample Monthly Returns")
    monthly_returns = (
        test_portfolio_daily.assign(month=lambda x: pd.to_datetime(x["execution_date"]).dt.to_period("M"))
        .groupby("month")["net_return"]
        .sum()
    )
    print(monthly_returns.to_string())
    print()
    print(f"Sector ranking output: {args.sector_ranking_output}")
    print(f"Selected sectors output: {args.selected_sectors_output}")
    print(f"Train portfolio output: {args.train_portfolio_output}")
    print(f"Test portfolio output: {args.test_portfolio_output}")
    print(f"Test trades output: {args.test_trades_output}")


if __name__ == "__main__":
    main()
