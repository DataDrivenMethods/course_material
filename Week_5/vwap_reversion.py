"""
vwap_reversion.py
=================
Mean-reversion backtest of the VWAP-deviation signal on one day of minute bars
(data/minute_bars.h5), for every symbol and every lookback N, holding h = N minutes.
N and h count clock minutes: minutes with no trades carry the last close forward.

At every minute t with trades, for each symbol and lookback N:
    signal    s_t = (close_t - VWAP_t) / VWAP_t      VWAP over minutes t-N+1 .. t
    position  q_t = -DOLLARS_PER_BP * (s_t in bp)    short above VWAP, long below
    trade     enter q_t at close_t, exit at close_(t+N)
    cost      |q_t| * cost_bp / 1e4 on entry and again on exit   (--cost-bp)
    PnL       q_t * (close_(t+N) / close_t - 1) - cost

A new trade opens every minute that traded, so up to N trades are open at once.
Trades that would end after 15:59 are not opened.

Costs are charged per trade, with no netting of overlapping trades. 'edge' is
net PnL per dollar of round trip, in bp. Entry is at close_t, the same trade
the signal uses, which flatters the result.

Usage
-----
    python vwap_reversion.py                  # no costs
    python vwap_reversion.py --cost-bp 0.5    # 0.5 bp on entry + 0.5 bp on exit
"""

import argparse

import pandas as pd

from vwap_ic import add_vwap_dev, to_minute_grid

H5_PATH = "data/minute_bars.h5"
LOOKBACKS = [1, 2, 5, 10, 15, 25, 50, 100]   # N = signal lookback = holding period (minutes)
DOLLARS_PER_BP = 1_000                        # 10 bp from VWAP -> $10,000 position


def backtest(bars: pd.DataFrame, n: int, cost_bp: float = 0.0) -> pd.DataFrame:
    """One trade per traded minute, sized -signal and held n minutes. One row per trade."""
    grid = to_minute_grid(bars)            # n counts clock minutes, even if some have no trades
    signal_bp = add_vwap_dev(grid, mode="rolling", window=n)["vwap_dev"] * 1e4
    trades = pd.DataFrame({
        "position": -DOLLARS_PER_BP * signal_bp,
        "ret":      grid["close"].shift(-n) / grid["close"] - 1,
    }).loc[bars.index].dropna()            # open trades only in minutes that traded
    trades["gross"] = trades["position"] * trades["ret"]
    trades["cost"] = trades["position"].abs() * 2 * cost_bp / 1e4    # entry + exit
    trades["pnl"] = trades["gross"] - trades["cost"]
    return trades


def main():
    parser = argparse.ArgumentParser(description="VWAP mean-reversion backtest on minute bars.")
    parser.add_argument("--cost-bp", type=float, default=0.0,
                        help="Trading cost in bp of notional, charged on entry and again on exit")
    args = parser.parse_args()

    with pd.HDFStore(H5_PATH, mode="r") as store:
        all_bars = {key.split("/")[-1]: store[key] for key in store.keys()}

    rows = []
    for sym, bars in all_bars.items():
        for n in LOOKBACKS:
            t = backtest(bars, n, args.cost_bp)
            rows.append({"symbol": sym, "N": n, "trades": len(t), "gross": t["gross"].sum(),
                         "cost": t["cost"].sum(), "pnl": t["pnl"].sum(),
                         "notional": t["position"].abs().sum(), "wins": (t["pnl"] > 0).sum()})
    results = pd.DataFrame(rows)

    print(f"VWAP mean reversion, hold h = N minutes, ${DOLLARS_PER_BP:,} per bp of signal, "
          f"cost {args.cost_bp} bp per side\n")

    pnl = results.pivot(index="symbol", columns="N", values="pnl")
    pnl.loc["TOTAL"] = pnl.sum()
    print("Net PnL ($)   rows: symbol   columns: N = lookback = holding period (minutes)")
    print(pnl.round(0).to_string(), "\n")

    total = results.groupby("N").sum(numeric_only=True)
    print("All symbols   edge = net PnL per $ of round trip (bp)   hit rate = share of trades with net PnL > 0")
    print(pd.DataFrame({
        "trades":    total["trades"],
        "gross ($)": total["gross"].round(0),
        "cost ($)":  total["cost"].round(0),
        "net ($)":   total["pnl"].round(0),
        "edge (bp)": (total["pnl"] / total["notional"] * 1e4).round(2),
        "hit rate":  (total["wins"] / total["trades"]).round(3),
    }).to_string())


if __name__ == "__main__":
    main()
