"""
minute_bars.py
==============
Generate OHLCV + extended minute bars from NASDAQ ITCH HDF5 tick data.

The ITCH HDF5 store uses integer `stock_locate` codes (not ticker symbols).
We resolve symbols via the /R (stock directory) table.

Metrics per bar:
  open, high, low, close, volume, vwap, trade_count,
  dollar_volume, n_at_bid, n_at_ask

Usage
-----
    python minute_bars.py                    # default SYMBOLS
    python minute_bars.py --symbols AAPL MSFT
    python minute_bars.py --save             # write CSVs + HDF5
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "data" / "itch.h5"

SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "GOOGL",
    "TSLA", "NVDA", "JPM", "GS", "SPY",
]
# META was not in this Oct-2019 file (it was still FB)

MARKET_OPEN_NS  = 9  * 3_600 * 1_000_000_000 + 30 * 60 * 1_000_000_000
MARKET_CLOSE_NS = 16 * 3_600 * 1_000_000_000
PRICE_SCALE     = 10_000

DATE = "2019-10-30"

# ---------------------------------------------------------------------------
# Stock directory — symbol → stock_locate
# ---------------------------------------------------------------------------

_LOCATE_CACHE: dict = {}

def get_locate(symbol: str, h5path: Path = DATA_PATH) -> int | None:
    if symbol in _LOCATE_CACHE:
        return _LOCATE_CACHE[symbol]
    with pd.HDFStore(str(h5path), mode="r") as store:
        R = store["R"][["stock_locate", "stock"]]
    row = R[R["stock"] == symbol]
    if row.empty:
        return None
    loc = int(row["stock_locate"].iloc[0])
    _LOCATE_CACHE[symbol] = loc
    return loc


def build_locate_map(h5path: Path = DATA_PATH) -> dict:
    with pd.HDFStore(str(h5path), mode="r") as store:
        R = store["R"][["stock_locate", "stock"]]
    return dict(zip(R["stock"], R["stock_locate"].astype(int)))

# ---------------------------------------------------------------------------
# Load executed trades
# ---------------------------------------------------------------------------

def load_trades(symbol: str, h5path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Return executed trades for *symbol* with columns:
        timestamp (int ns), price (float $), shares (int),
        buy_sell_indicator (+1 buyer-initiated / -1 seller-initiated / 0 unknown)
    Filtered to regular market hours.
    """
    locate = get_locate(symbol, h5path)
    if locate is None:
        print(f"  ⚠  {symbol} not found in stock directory.")
        return pd.DataFrame(columns=["timestamp", "price", "shares", "buy_sell_indicator"])

    # --- Load Add Orders (A + F) for direction lookup --------------------
    orders = _load_orders(locate, h5path)

    frames = []

    # --- E: Order Executed ------------------------------------------------
    with pd.HDFStore(str(h5path), mode="r") as store:
        try:
            e = store.select("E", where=f"stock_locate={locate}")
        except Exception:
            e = pd.DataFrame()
    if not e.empty and not orders.empty:
        merged = e.rename(columns={"executed_shares": "shares"}).merge(
            orders[["order_reference_number", "buy_sell_indicator", "price"]],
            on="order_reference_number", how="inner"
        )
        merged["price"] = merged["price"] / PRICE_SCALE
        # The joined indicator is the RESTING order's side; the aggressor is the
        # contra side: resting sell (-1) hit at the ask -> buyer-initiated (+1).
        merged["buy_sell_indicator"] = -merged["buy_sell_indicator"]
        frames.append(merged[["timestamp", "price", "shares", "buy_sell_indicator"]])

    # --- C: Executed at Different Price -----------------------------------
    with pd.HDFStore(str(h5path), mode="r") as store:
        try:
            c = store.select("C", where=f"stock_locate={locate}")
        except Exception:
            c = pd.DataFrame()
    if not c.empty and not orders.empty:
        merged = c.rename(columns={"execution_price": "price",
                                   "executed_shares": "shares"}).merge(
            orders[["order_reference_number", "buy_sell_indicator"]],
            on="order_reference_number", how="inner"
        )
        if "price" not in merged.columns:
            # fallback: use price from orders
            merged = c.rename(columns={"executed_shares": "shares"}).merge(
                orders[["order_reference_number", "buy_sell_indicator", "price"]],
                on="order_reference_number", how="inner"
            )
        merged["price"] = merged["price"] / PRICE_SCALE
        # The joined indicator is the RESTING order's side; the aggressor is the
        # contra side: resting sell (-1) hit at the ask -> buyer-initiated (+1).
        merged["buy_sell_indicator"] = -merged["buy_sell_indicator"]
        frames.append(merged[["timestamp", "price", "shares", "buy_sell_indicator"]])

    # --- P: Trade Message (non-cross) ----------------------------------------
    with pd.HDFStore(str(h5path), mode="r") as store:
        try:
            p = store.select("P", where=f"stock_locate={locate}")
        except Exception:
            p = pd.DataFrame()
    if not p.empty:
        p = p[["timestamp", "price", "shares", "buy_sell_indicator"]].copy()
        p["price"] = p["price"] / PRICE_SCALE
        # ITCH 5.0 always reports "B" for P messages, so the side is unknown;
        # mark it 0 and let the tick rule classify these trades.
        p["buy_sell_indicator"] = 0
        frames.append(p)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "price", "shares", "buy_sell_indicator"])

    df = pd.concat(frames, ignore_index=True)

    # Normalise timestamp → int nanoseconds
    if pd.api.types.is_timedelta64_dtype(df["timestamp"]):
        df["timestamp"] = df["timestamp"].dt.total_seconds().mul(1e9).astype("int64")
    else:
        df["timestamp"] = df["timestamp"].astype("int64")

    df = df[(df["timestamp"] >= MARKET_OPEN_NS) & (df["timestamp"] < MARKET_CLOSE_NS)]
    return df.sort_values("timestamp").reset_index(drop=True)


def _load_orders(locate: int, h5path: Path) -> pd.DataFrame:
    """Load Add Order messages (A + F) for stock_locate *locate*."""
    frames = []
    with pd.HDFStore(str(h5path), mode="r") as store:
        for mtype in ("A", "F"):
            try:
                df = store.select(mtype, where=f"stock_locate={locate}")
                if not df.empty:
                    frames.append(df[["order_reference_number", "buy_sell_indicator", "price"]])
            except Exception:
                pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("order_reference_number")

# ---------------------------------------------------------------------------
# Tick rule for direction
# ---------------------------------------------------------------------------

def apply_tick_rule(trades: pd.DataFrame) -> pd.DataFrame:
    df = trades.copy()
    mask = df["buy_sell_indicator"] == 0
    if mask.any():
        diff = df["price"].diff()
        df.loc[mask, "buy_sell_indicator"] = np.where(diff[mask] >= 0, 1, -1)
    df["buy_sell_indicator"] = (df["buy_sell_indicator"]
                                  .replace(0, np.nan).ffill().fillna(1).astype(int))
    return df

# ---------------------------------------------------------------------------
# Minute bar construction
# ---------------------------------------------------------------------------

def build_minute_bars(trades: pd.DataFrame, date: str = DATE) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    trades = apply_tick_rule(trades)
    base   = pd.Timestamp(date)
    t = trades.copy()
    t["dt"] = base + pd.to_timedelta(t["timestamp"], unit="ns")
    t = t.set_index("dt").sort_index()

    t["dv"]     = t["price"] * t["shares"]
    t["at_ask"] = t["shares"].where(t["buy_sell_indicator"] ==  1, 0)
    t["at_bid"] = t["shares"].where(t["buy_sell_indicator"] == -1, 0)

    r = t.resample("1min")
    bars = pd.DataFrame({
        "open":          r["price"].first(),
        "high":          r["price"].max(),
        "low":           r["price"].min(),
        "close":         r["price"].last(),
        "volume":        r["shares"].sum(),
        "trade_count":   r["price"].count(),
        "dollar_volume": r["dv"].sum(),
        "n_at_ask":      r["at_ask"].sum(),
        "n_at_bid":      r["at_bid"].sum(),
    }).dropna(subset=["open"])

    bars["vwap"]          = bars["dollar_volume"] / bars["volume"]
    bars["volume"]        = bars["volume"].astype(int)
    bars["trade_count"]   = bars["trade_count"].astype(int)
    bars["n_at_ask"]      = bars["n_at_ask"].astype(int)
    bars["n_at_bid"]      = bars["n_at_bid"].astype(int)
    bars["dollar_volume"] = bars["dollar_volume"].round(2)
    bars.index.name = "minute"
    return bars

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_minute_bars(symbols=SYMBOLS, h5path=DATA_PATH) -> dict:
    results = {}
    for sym in symbols:
        print(f"\n{'─'*52}")
        print(f"  {sym}")
        try:
            trades = load_trades(sym, h5path)
            if trades.empty:
                print(f"  ⚠  No trades found.")
                continue
            print(f"  Loaded {len(trades):,} ticks")
            bars = build_minute_bars(trades)
            print(f"  → {len(bars)} minute bars  |  "
                  f"close: {bars['close'].iloc[-1]:.2f}  |  "
                  f"vol: {bars['volume'].sum():,.0f}")
            results[sym] = bars
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  ✗  {exc}")
    return results

# ---------------------------------------------------------------------------
# HDF5 persistence
# ---------------------------------------------------------------------------

def save_bars_hdf5(results: dict, out_path: Path = None):
    if out_path is None:
        out_path = DATA_PATH.parent / "minute_bars.h5"
    with pd.HDFStore(str(out_path), mode="w", complevel=5, complib="blosc") as store:
        for sym, bars in results.items():
            store.put(f"minute_bars/{sym}", bars, format="table", data_columns=True)
            print(f"  ✅  {sym} → {out_path}:/minute_bars/{sym}")
    print(f"\nAll bars saved to {out_path}")
    return out_path

def save_bars_csv(results: dict, out_dir: Path = None):
    if out_dir is None:
        out_dir = DATA_PATH.parent / "minute_bars"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sym, bars in results.items():
        bars.to_csv(out_dir / f"{sym}_1min.csv")

def load_bars_hdf5(sym: str, h5path: Path = None) -> pd.DataFrame:
    if h5path is None:
        h5path = DATA_PATH.parent / "minute_bars.h5"
    with pd.HDFStore(str(h5path), mode="r") as store:
        return store[f"minute_bars/{sym}"]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate minute bars from ITCH HDF5.")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    res = generate_minute_bars(symbols=args.symbols, h5path=args.data)

    if args.save and res:
        save_bars_hdf5(res)
        save_bars_csv(res)

    print("\n\n=== Summary ===")
    print(f"{'Symbol':8s}  {'Bars':>5s}  {'Last Close':>10s}  {'Total Vol':>14s}")
    print("─" * 44)
    for sym, bars in res.items():
        print(f"{sym:8s}  {len(bars):5d}  {bars['close'].iloc[-1]:10.2f}  "
              f"{bars['volume'].sum():14,.0f}")
