"""
plot_minute_bars.py
===================
Plot one day of 1-minute bars for a single ticker: candlesticks with session
VWAP, volume, and order-flow imbalance.

Bars are read from data/minute_bars.h5 (written by `python minute_bars.py --save`).
If the ticker is not there, they are built from data/itch.h5 on the fly.

Usage
-----
    python plot_minute_bars.py AAPL
    python plot_minute_bars.py MSFT --out msft.png      # save instead of showing
    python plot_minute_bars.py NVDA --start 09:30 --end 11:00
"""

from pathlib import Path
import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import minute_bars as mb

BARS_PATH = Path(__file__).parent / "data" / "minute_bars.h5"

# Colours: up/down is a polarity, so it uses a diverging blue/red pair;
# chrome is recessive gray so the data carries the contrast.
UP, DOWN = "#2a78d6", "#e34948"
VWAP     = "#0b0b0b"
SURFACE  = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def load_bars(sym: str) -> pd.DataFrame:
    """Saved bars if available, otherwise build them from the ITCH store."""
    if BARS_PATH.exists():
        with pd.HDFStore(str(BARS_PATH), mode="r") as store:
            key = f"/minute_bars/{sym}"
            if key in store.keys():
                return store[key]
    print(f"{sym} not in {BARS_PATH.name}; building bars from {mb.DATA_PATH} ...")
    trades = mb.load_trades(sym)
    if trades.empty:
        raise SystemExit(f"No trades found for {sym}.")
    return mb.build_minute_bars(trades)


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.yaxis.label.set_color(INK2)


def plot(sym: str, bars: pd.DataFrame, out: Path | None = None):
    t = mdates.date2num(bars.index.to_pydatetime())
    width = 0.7 / (24 * 60)                       # 70% of one minute, in days
    up = (bars["close"] >= bars["open"]).to_numpy()
    colors = np.where(up, UP, DOWN)

    session_vwap = bars["dollar_volume"].cumsum() / bars["volume"].cumsum()
    imbalance = (bars["n_at_ask"] - bars["n_at_bid"]) / bars["volume"].replace(0, np.nan)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True, figsize=(12, 7.5), facecolor=SURFACE,
        gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.22})

    # --- 1. Candlesticks + session VWAP ------------------------------------
    ax1.vlines(t, bars["low"], bars["high"], colors=colors, linewidth=0.8)
    body_lo = np.minimum(bars["open"], bars["close"])
    body_h = (bars["close"] - bars["open"]).abs()
    body_h = body_h.where(body_h > 0, bars["close"] * 1e-5)   # doji: hairline body
    ax1.bar(t, body_h, bottom=body_lo, width=width, color=colors, linewidth=0)
    ax1.plot(t, session_vwap, color=VWAP, linewidth=1.6)
    ax1.set_ylabel("Price ($)")
    ax1.legend(handles=[Patch(color=UP, label="Close ≥ open"),
                        Patch(color=DOWN, label="Close < open"),
                        Line2D([], [], color=VWAP, linewidth=1.6, label="Session VWAP")],
               loc="upper left", frameon=False, fontsize=8.5, ncol=3,
               labelcolor=INK2)

    # --- 2. Volume ----------------------------------------------------------
    ax2.bar(t, bars["volume"] / 1e3, width=width, color=colors, linewidth=0)
    ax2.set_ylabel("Volume (k sh)")

    # --- 3. Order-flow imbalance ---------------------------------------------
    imb = imbalance.fillna(0).to_numpy()
    ax3.bar(t, imb, width=width, color=np.where(imb >= 0, UP, DOWN), linewidth=0)
    ax3.axhline(0, color=AXIS, linewidth=0.8)
    ax3.set_ylim(-1, 1)
    ax3.set_ylabel("Imbalance")
    ax3.set_title("Order-flow imbalance: (volume at ask − volume at bid) / volume",
                  loc="left", fontsize=8.5, color=MUTED, pad=4)

    for ax in (ax1, ax2, ax3):
        style_axis(ax)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax3.xaxis.set_major_locator(locator)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax3.set_xlim(t[0] - 2 * width, t[-1] + 2 * width)

    day = bars.index[0].strftime("%Y-%m-%d")
    chg = bars["close"].iloc[-1] / bars["open"].iloc[0] - 1
    fig.suptitle(f"{sym}  ·  1-minute bars  ·  {day}",
                 x=0.07, y=0.975, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.07, 0.928,
             f"Open {bars['open'].iloc[0]:.2f}   Close {bars['close'].iloc[-1]:.2f} "
             f"({chg:+.2%})   Range {bars['low'].min():.2f}–{bars['high'].max():.2f}   "
             f"Volume {bars['volume'].sum() / 1e6:,.1f}M sh   "
             f"VWAP {session_vwap.iloc[-1]:.2f}",
             ha="left", fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.06)

    if out:
        fig.savefig(out, dpi=150, facecolor=SURFACE)
        print(f"Saved {out}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Plot 1-minute bars for one ticker.")
    p.add_argument("symbol", help="ticker, e.g. AAPL")
    p.add_argument("--start", help="first minute to show, HH:MM (default: open)")
    p.add_argument("--end", help="last minute to show, HH:MM (default: close)")
    p.add_argument("--out", type=Path, help="save to this file instead of showing")
    args = p.parse_args()

    sym = args.symbol.upper()
    bars = load_bars(sym)
    day = bars.index[0].strftime("%Y-%m-%d")
    if args.start:
        bars = bars[bars.index >= pd.Timestamp(f"{day} {args.start}")]
    if args.end:
        bars = bars[bars.index <= pd.Timestamp(f"{day} {args.end}")]
    if bars.empty:
        raise SystemExit("No bars in the requested time window.")
    plot(sym, bars, args.out)


if __name__ == "__main__":
    main()
