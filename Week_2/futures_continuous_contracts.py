#!/usr/bin/env python3
"""
Continuous futures series construction  --  IEOR E4576, Week 2.

ALL DATA IS SIMULATED.  The example uses NYMEX WTI crude oil (ticker root "CL")
labels and real CL last-trading days, but every price is generated here:

    spot_t      = random walk, s_0 = $80, daily sigma = $0.50
    F(t, T)     = spot_t + $6 * (1 - exp(-dte / 55 days)) + small noise,
                  where dte = days from t to the last trading day T.  The carry
                  is steepest at the front and flattens further out, the usual
                  shape of a contango curve, and F -> spot_t as dte -> 0.
    volume(t,T) = bell curve peaking 25 days before T

It shows:
  1. Individual contracts expire, so a long history must be "rolled"
  2. Two roll rules: time-based (days to expiry) vs volume-based (liquidity)
  3. Rolling puts the contract spread into the series; back-adjustment removes it
  4. Contango: the futures curve slopes up with days to expiry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday

SEED           = 123
SPOT_0         = 80.00   # $/bbl starting level of the simulated spot
SPOT_SIGMA     = 0.50    # $/day standard deviation of the spot random walk
CARRY_LEVEL    = 6.00    # $/bbl gap between spot and the far end of the curve
CARRY_TAU      = 55.0    # days; carry accrues fastest at the front of the curve
ROLL_DAYS      = 5       # time-based rule: roll this many days before expiry
VOL_PEAK_DTE   = 25      # volume peaks this many days before expiry
VOL_WIDTH      = 18

# categorical palette, validated for colour-vision deficiency
# Contracts own the blue/orange/aqua hues.  The derived series in panels 2-4
# are NOT contracts, so they use a separate pair and never borrow those hues.
C_CONTRACT = ["#2a78d6", "#eb6834", "#1baf7a"]   # blue, orange, aqua
C_RAW, C_ADJ, C_GRID = "#4a3aa7", "#2b2b2b", "#8a8a85"   # violet, near-black

# Three consecutive CL contracts, with their real 2023 last trading days.
# CME month codes: H = March, J = April, K = May.
CONTRACTS = [
    {"name": "CLH23", "label": "Mar 2023", "last_trade": "2023-02-21"},
    {"name": "CLJ23", "label": "Apr 2023", "last_trade": "2023-03-21"},
    {"name": "CLK23", "label": "May 2023", "last_trade": "2023-04-20"},
]
START = "2023-01-03"     # all three contracts are already listed on this day


class NYMEXHolidays(USFederalHolidayCalendar):
    """US federal holidays plus Good Friday, which the exchange also closes for."""
    rules = USFederalHolidayCalendar.rules + [GoodFriday]


TRADING_DAY = pd.offsets.CustomBusinessDay(calendar=NYMEXHolidays())


def carry(dte):
    """Contango premium over spot for a contract `dte` days from expiry.

    Concave in dte: most of the premium is earned in the first weeks and the
    curve flattens further out, and it vanishes at expiry, where the futures
    price must converge to spot.
    """
    return CARRY_LEVEL * (1 - np.exp(-np.asarray(dte, dtype=float) / CARRY_TAU))


def simulate_contracts():
    """Build the synthetic panel: one row per (trading day, live contract)."""
    rng = np.random.default_rng(SEED)

    expiry = {c["name"]: pd.Timestamp(c["last_trade"]) for c in CONTRACTS}
    days = pd.bdate_range(START, max(expiry.values()), freq=TRADING_DAY)

    # One spot path shared by every contract, so the contracts co-move.
    spot = pd.Series(SPOT_0 + np.cumsum(rng.normal(0, SPOT_SIGMA, len(days))),
                     index=days)

    rows = []
    for c in CONTRACTS:
        for d in days[days <= expiry[c["name"]]]:
            dte = (expiry[c["name"]] - d).days
            rows.append({
                "date": d,
                "contract": c["name"],
                "label": c["label"],
                "days_to_expiry": dte,
                "price": spot[d] + carry(dte) + rng.normal(0, 0.05),
                "volume": 200 + 5000 * np.exp(-((dte - VOL_PEAK_DTE) / VOL_WIDTH) ** 2),
            })
    return pd.DataFrame(rows)


def build_continuous(df, rule):
    """Pick one contract per day. rule='time' (days to expiry) or 'volume'.

    Only contracts more than ROLL_DAYS from expiry are ever held, so the
    series ends when the last contract reaches that threshold.
    """
    holdable = df[df["days_to_expiry"] > ROLL_DAYS]

    picks = []
    for date, day in holdable.groupby("date"):
        sel = (day.loc[day["days_to_expiry"].idxmin()] if rule == "time"
               else day.loc[day["volume"].idxmax()])
        picks.append({"date": date, "contract_used": sel["contract"],
                      "price": sel["price"], "volume": sel["volume"]})

    s = pd.DataFrame(picks).sort_values("date").reset_index(drop=True)
    s["return"] = s["price"].pct_change() * 100
    rolls = s.index[s["contract_used"] != s["contract_used"].shift(1)][1:]
    return s, list(rolls)


def back_adjust(series, rolls, df):
    """Back-adjustment: add each roll's spread to every price before the roll.

    The spread is measured between the two contracts on the roll day itself,
    when both are trading.  That is the part of the raw jump that is not a
    market move, so removing it leaves the day's true return intact.
    """
    price = df.set_index(["date", "contract"])["price"]

    s = series.copy()
    s["adjustment"] = 0.0
    spreads = []
    for i in rolls:
        day = s.at[i, "date"]
        spread = price[(day, s.at[i, "contract_used"])] - price[(day, s.at[i - 1, "contract_used"])]
        spreads.append(spread)
        s.loc[: i - 1, "adjustment"] += spread

    s["adjusted_price"] = s["price"] + s["adjustment"]
    s["adjusted_return"] = s["adjusted_price"].pct_change() * 100
    return s, spreads


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def _style(ax, dates=True):
    if dates:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.25, color=C_GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_rolling(df, time_s, vol_s, adj_s, rolls, vol_rolls):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Continuous Futures Construction — SIMULATED WTI Crude Oil (CL)",
                 fontsize=15, fontweight="bold")
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.set_title("1. Individual contracts", fontsize=11)
    for color, (_, g) in zip(C_CONTRACT, df.groupby("contract", sort=False)):
        ax1.plot(g["date"], g["price"], color=color, linewidth=2,
                 label=f"{g['contract'].iloc[0]} ({g['label'].iloc[0]}, "
                       f"last trade {g['date'].max():%b %d})")
        ax1.plot(g["date"].max(), g["price"].iloc[-1], "x",
                 color=color, markersize=9, markeredgewidth=2.5)
    ax1.set_ylabel("Simulated price ($/bbl)")
    ax1.legend(fontsize=8, frameon=False)
    _style(ax1)

    ax2.set_title("2. Roll rules: time vs. volume", fontsize=11)
    # The volume rule rolls first, so the two series differ only between the
    # two roll dates; shade that window so the disagreement is visible.
    for vi, ti in zip(vol_rolls, rolls):
        ax2.axvspan(vol_s.at[vi, "date"], time_s.at[ti, "date"],
                    color=C_GRID, alpha=0.15, zorder=0)
    ax2.plot(vol_s["date"], vol_s["price"], color=C_RAW, linewidth=2,
             linestyle="--", label="Volume-based roll")
    ax2.plot(time_s["date"], time_s["price"], color=C_ADJ, linewidth=2,
             label="Time-based roll")
    ax2.plot(vol_s.loc[vol_rolls, "date"], vol_s.loc[vol_rolls, "price"], "o",
             color=C_RAW, markersize=8, markeredgecolor="white", markeredgewidth=1.5,
             linestyle="none", label="Volume roll date")
    ax2.plot(time_s.loc[rolls, "date"], time_s.loc[rolls, "price"], "o",
             color=C_ADJ, markersize=8, markeredgecolor="white", markeredgewidth=1.5,
             linestyle="none", label="Time roll date")
    for vi, ti in zip(vol_rolls, rolls):
        ax2.annotate(f"vol → {vol_s.at[vi, 'contract_used']}",
                     (vol_s.at[vi, "date"], vol_s.at[vi, "price"]),
                     textcoords="offset points", xytext=(-12, 14),
                     ha="right", fontsize=8, color=C_RAW)
        ax2.annotate(f"time → {time_s.at[ti, 'contract_used']}",
                     (time_s.at[ti, "date"], time_s.at[ti, "price"]),
                     textcoords="offset points", xytext=(6, -16),
                     ha="left", fontsize=8, color=C_ADJ)
    ax2.set_ylabel("Simulated price ($/bbl)")
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax2)

    ax3.set_title("3. Unadjusted vs. back-adjusted price", fontsize=11)
    ax3.plot(time_s["date"], time_s["price"], color=C_RAW, linewidth=2,
             label="Unadjusted")
    ax3.plot(adj_s["date"], adj_s["adjusted_price"], color=C_ADJ, linewidth=2,
             label="Back-adjusted")
    for i in rolls:
        ax3.axvline(time_s.at[i, "date"], color=C_GRID, linestyle=":", alpha=0.8)
    ax3.set_ylabel("Simulated price ($/bbl)")
    ax3.legend(fontsize=9, frameon=False)
    _style(ax3)

    ax4.set_title("4. Daily returns", fontsize=11)
    ax4.axhline(0, color="#0b0b0b", linewidth=0.8, alpha=0.4)
    ax4.plot(time_s["date"], time_s["return"], color=C_RAW, linewidth=1.4,
             label="Unadjusted")
    ax4.plot(adj_s["date"], adj_s["adjusted_return"], color=C_ADJ, linewidth=1.4,
             label="Back-adjusted")
    ax4.scatter(time_s.loc[rolls, "date"], time_s.loc[rolls, "return"],
                color=C_RAW, s=60, zorder=5, edgecolor="white", linewidth=1.5)
    ax4.scatter(adj_s.loc[rolls, "date"], adj_s.loc[rolls, "adjusted_return"],
                color=C_ADJ, s=60, zorder=5, edgecolor="white", linewidth=1.5)
    for i in rolls:
        # label to the right of the roll date, clear of the neighbouring days
        ax4.axvline(time_s.at[i, "date"], color=C_GRID, linestyle=":", alpha=0.8, zorder=0)
        ax4.annotate(f"{time_s.at[i, 'return']:+.2f}% unadjusted\n"
                     f"{adj_s.at[i, 'adjusted_return']:+.2f}% adjusted",
                     (time_s.at[i, "date"], time_s.at[i, "return"]),
                     textcoords="offset points", xytext=(10, 0),
                     ha="left", va="center", fontsize=8, color="#52514e")
    ax4.margins(y=0.18)
    ax4.set_ylabel("Daily return (%)")
    ax4.legend(fontsize=9, frameon=False, loc="upper left")
    _style(ax4)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("futures_roll_and_backadjustment.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lifecycle(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9.5))
    fig.suptitle("Futures Contracts — SIMULATED WTI Crude Oil (CL)",
                 fontsize=15, fontweight="bold")

    ax1.set_title("1. A contract's life, in calendar time: volume builds, "
                  "peaks, then dies at expiry", fontsize=11)
    for color, (_, g) in zip(C_CONTRACT, df.groupby("contract", sort=False)):
        ax1.plot(g["date"], g["volume"], color=color, linewidth=2,
                 label=g["contract"].iloc[0])
        peak = g.loc[g["volume"].idxmax()]
        ax1.annotate(g["contract"].iloc[0], (peak["date"], peak["volume"]),
                     textcoords="offset points", xytext=(0, -16),
                     ha="center", fontsize=9, color="#52514e")
    ax1.set_xlabel("Calendar date (trading days)")
    ax1.set_ylabel("Simulated contracts traded")
    ax1.legend(fontsize=9, frameon=False, loc="upper left")
    _style(ax1)

    # Panel 2 is a CROSS-SECTION, not a time series: it holds one day fixed and
    # looks across the contracts trading that day.  So the x-axis is the
    # delivery month, one tick per contract -- never calendar time, and never a
    # continuous axis, both of which invite the line to be read as a price path.
    order = [c["name"] for c in CONTRACTS]
    months = [c["label"] for c in CONTRACTS]
    curve = df.pivot_table(index="date", columns="contract", values="price")[order]
    full = curve.dropna()          # days on which all three contracts trade
    snapshots = [full.index[int(f * (len(full) - 1))] for f in (0.1, 0.5, 0.9)]
    shades = ["#a8cbf0", "#5a9ae0", "#16437d"]      # early -> late
    x = np.arange(len(order))

    for color, day in zip(shades, snapshots):
        ax2.plot(x, curve.loc[day].values, "-o", color=color, linewidth=2,
                 markersize=9, markeredgecolor="white", markeredgewidth=1.5)
        ax2.annotate(f"{day:%b %d}", (x[-1], curve.at[day, order[-1]]),
                     textcoords="offset points", xytext=(12, 0),
                     ha="left", va="center", fontsize=9,
                     color=color, fontweight="bold")

    ax2.set_title("2. The futures curve on one day: later delivery, "
                  "higher price (contango)", fontsize=11)
    ax2.text(0.02, 0.97, "Each line is one trading day.",
             transform=ax2.transAxes, va="top", ha="left",
             fontsize=9, color="#52514e")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{m}\n{n}" for m, n in zip(months, order)])
    ax2.set_xlim(-0.3, len(order) - 1 + 0.6)
    ax2.margins(y=0.22)
    ax2.set_xlabel("Delivery month — each point is a different contract")
    ax2.set_ylabel("Simulated price ($/bbl)")
    _style(ax2, dates=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("futures_contract_lifecycle.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# real data: front-month continuous series from WRDS trdstrm via futspread
# --------------------------------------------------------------------------

WRDS_TICKERS = {"CL": "WTI crude ($/bbl)", "NG": "Natural gas ($/MMBtu)",
                "RB": "RBOB gasoline ($/gal)", "ZC": "Corn (c/bu)"}
WRDS_SINCE = "2015-01-01"
CURVE_DATE = "2020-06-01"   # one day on which to look across the curve
CURVE_YEARS = 5


def load_wrds_panel(ticker, since=WRDS_SINCE):
    """Real bars in the same shape as simulate_contracts(): one row per
    (trading day, live contract).  Only months that actually trade are kept,
    and `days_to_expiry` counts to the last holdable day - the last trading
    day, or the first notice day where that comes first (ZC)."""
    from futspread.codes import NUM_TO_MONTH_CODE
    from futspread.instances import liquid_months, load_contracts, load_prices
    from futspread.universe import contrcode_for

    info = load_contracts(ticker)
    info = info[info["contract_month"].map(NUM_TO_MONTH_CODE).isin(liquid_months(ticker))]
    px = load_prices(contrcode_for(ticker))
    rows = []
    for r in info[info["ltd"] >= since].itertuples():
        bars = px.get(int(r.futcode))
        if bars is None:
            continue
        bars = bars[(bars["date"] >= since) & (bars["date"] <= r.ltd)
                    & bars["settlement"].notna() & (bars["oi"] > 0)]
        for b in bars.itertuples():
            rows.append({"date": b.date,
                         "contract": f"{ticker}{NUM_TO_MONTH_CODE[r.contract_month]}{r.contract_year % 100:02d}",
                         "days_to_expiry": (r.ltd - b.date).days,
                         "price": b.settlement, "volume": b.volume})
    return pd.DataFrame(rows).sort_values(["date", "days_to_expiry"]).reset_index(drop=True)


def plot_front_month_wrds():
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Front-Month Continuous Series — WRDS trdstrm settlement prices",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.925,
             f"Roll rule: hold the nearest contract; roll to the next one {ROLL_DAYS} days "
             "before its last holdable day (last trading day, or first notice day for ZC). "
             "Unadjusted: each roll leaves the calendar spread in the series.",
             ha="center", fontsize=10, color="#52514e")

    for ax, (ticker, label) in zip(axes.ravel(), WRDS_TICKERS.items()):
        df = load_wrds_panel(ticker)
        s, rolls = build_continuous(df, "time")
        ax.plot(s["date"], s["price"], color=C_ADJ, linewidth=1.0, label="Front-month settlement")
        ax.plot(s.loc[rolls, "date"], s.loc[rolls, "price"], "|", color=C_CONTRACT[1],
                markersize=7, markeredgewidth=1.0, linestyle="none",
                label=f"Roll date ({len(rolls)} rolls)")
        ax.set_title(f"{ticker} — {label}: {s['contract_used'].iloc[0]} → "
                     f"{s['contract_used'].iloc[-1]}", fontsize=11, loc="left")
        ax.set_ylabel(label.split("(")[1].rstrip(")"))
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        _style(ax)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("futures_front_month_wrds.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_seasonality_wrds():
    """The futures curve on one day: price against delivery month.

    Seasonality shows up as a repeating shape along the curve - it is the
    market's forward price for each delivery month, all quoted on the same
    day, so trend and spot level are held fixed.
    """
    tickers = list(WRDS_TICKERS)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(f"Seasonality in the Futures Curve — every contract's settlement on "
                 f"{pd.Timestamp(CURVE_DATE):%d %b %Y}, next {CURVE_YEARS} years",
                 fontsize=14, fontweight="bold")

    for ax, ticker in zip(axes.ravel(), tickers):
        df = load_wrds_panel(ticker, since=CURVE_DATE)
        day = df[df["date"] == df["date"].min()]
        day = day[day["days_to_expiry"] <= 365 * CURVE_YEARS].sort_values("days_to_expiry")
        x = np.arange(len(day))
        ax.plot(x, day["price"], "-o", color=C_CONTRACT[0], linewidth=2,
                markersize=6, markeredgecolor="white", markeredgewidth=1.2)
        step = 3 if len(day) > 24 else 1        # monthly series: label every third contract
        ax.set_xticks(x[::step])
        ax.set_xticklabels(day["contract"].iloc[::step], rotation=90, fontsize=7)
        ax.set_title(f"{ticker} — {WRDS_TICKERS[ticker]}", fontsize=11, loc="left")
        ax.set_xlabel("Delivery month — each point is a different contract")
        _style(ax, dates=False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("futures_seasonality_wrds.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------

def main():
    df = simulate_contracts()
    time_s, time_rolls = build_continuous(df, "time")
    vol_s, vol_rolls = build_continuous(df, "volume")
    adj_s, spreads = back_adjust(time_s, time_rolls, df)

    print("\nCONTINUOUS FUTURES SERIES — SIMULATED WTI CRUDE OIL (CL)")
    print(f"Synthetic prices: spot random walk (s0=${SPOT_0:.0f}, "
          f"sigma=${SPOT_SIGMA:.2f}/day) plus a contango premium rising to "
          f"${CARRY_LEVEL:.0f} at the long end.\n")

    print("Contracts")
    print(f"  {'ticker':<7}{'delivery':<11}{'first day':<13}{'expiry':<13}{'days':>4}")
    for _, g in df.groupby("contract", sort=False):
        print(f"  {g['contract'].iloc[0]:<7}{g['label'].iloc[0]:<11}"
              f"{g['date'].min():%Y-%m-%d}   {g['date'].max():%Y-%m-%d}   {len(g):>4}")

    print(f"\nRolls  ({len(df)} contract-days -> {len(time_s)} continuous days)")
    for i, spread in zip(time_rolls, spreads):
        print(f"  time    {time_s.at[i, 'date']:%Y-%m-%d}  "
              f"{time_s.at[i - 1, 'contract_used']} -> {time_s.at[i, 'contract_used']}"
              f"   spread +${spread:.2f}")
    for i in vol_rolls:
        print(f"  volume  {vol_s.at[i, 'date']:%Y-%m-%d}  "
              f"{vol_s.at[i - 1, 'contract_used']} -> {vol_s.at[i, 'contract_used']}")

    print("\nBack-adjustment takes the spread out of the roll-day return, "
          "leaving the real move:")
    for i in time_rolls:
        print(f"  {time_s.at[i, 'date']:%Y-%m-%d}: "
              f"{time_s.at[i, 'return']:+.2f}% unadjusted -> "
              f"{adj_s.at[i, 'adjusted_return']:+.2f}% adjusted")

    plot_rolling(df, time_s, vol_s, adj_s, time_rolls, vol_rolls)
    plot_lifecycle(df)
    print("\nWrote futures_roll_and_backadjustment.png, futures_contract_lifecycle.png")

    plot_front_month_wrds()
    print("Wrote futures_front_month_wrds.png (real data, "
          f"{', '.join(WRDS_TICKERS)} since {WRDS_SINCE})")

    plot_seasonality_wrds()
    print(f"Wrote futures_seasonality_wrds.png (curves on {CURVE_DATE})\n")


if __name__ == "__main__":
    main()
