"""
kalman_filter.py
================
Apply a Kalman filter to a single stock's minute-bar close price series,
following the example in:
  "Machine Learning for Finance" — Chapter 4, p. 100–103
  (Stefan Jansen, Packt Publishing)

The Kalman filter is treated here as a dynamic signal smoother.
Rather than a fixed moving-average window, it maintains a probabilistic
estimate of the "true" hidden price state, updating it at each new
minute-bar observation.

Model (univariate random walk + noise)
---------------------------------------
  Hidden state:    x_{t+1} = x_t + w_t,       w_t ~ N(0, Q)   transition noise
  Observation:     z_t     = x_t + v_t,        v_t ~ N(0, R)   measurement noise

  Q (transition_covariance) : how fast the true price can drift
  R (observation_covariance): how noisy each price observation is

Usage
-----
    python kalman_filter.py                        # defaults: AAPL, from minute_bars.h5
    python kalman_filter.py --symbol MSFT
    python kalman_filter.py --symbol AAPL --q 0.005 --r 2.0
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pykalman import KalmanFilter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE       = Path(__file__).parent
BARS_H5    = HERE / "data" / "minute_bars.h5"
RESULTS_H5 = HERE / "data" / "kalman_results.h5"

# ---------------------------------------------------------------------------
# Load minute bars
# ---------------------------------------------------------------------------

def load_minute_bars(symbol: str, h5path: Path = BARS_H5) -> pd.DataFrame:
    """Load minute bars for *symbol* from the shared HDF5 store."""
    with pd.HDFStore(str(h5path), mode="r") as store:
        key = f"minute_bars/{symbol}"
        if key not in store:
            raise KeyError(f"Symbol '{symbol}' not found in {h5path}. "
                           f"Available: {store.keys()}")
        return store[key]

# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------

def fit_kalman(
    prices: pd.Series,
    transition_covariance: float = 0.01,
    observation_covariance: float = 1.0,
) -> pd.DataFrame:
    """
    Fit a univariate Kalman filter to *prices* (minute close prices).

    Parameters
    ----------
    prices                  : pd.Series indexed by minute datetime
    transition_covariance Q : process noise — how quickly the true state can change
    observation_covariance R: measurement noise — how noisy each observation is

    Returns
    -------
    DataFrame with columns: close, kalman, kalman_std, ma_20, ma_60, ma_120
    """
    kf = KalmanFilter(
        transition_matrices    = [1],      # x_{t+1} = 1 * x_t + noise
        observation_matrices   = [1],      # z_t     = 1 * x_t + noise
        initial_state_mean     = prices.iloc[0],
        initial_state_covariance = 1,
        observation_covariance = observation_covariance,
        transition_covariance  = transition_covariance,
    )

    state_means, state_covs = kf.filter(prices.values)

    result = prices.to_frame("close")
    result["kalman"]     = state_means.flatten()
    result["kalman_std"] = np.sqrt(state_covs.flatten())

    # Moving averages for comparison (20, 60, 120 minutes ~ 0.5h / 1.5h / 2h)
    for w, label in [(20, "ma_20"), (60, "ma_60"), (120, "ma_120")]:
        result[label] = prices.rolling(window=w).mean()

    return result


def fit_kalman_smooth(
    prices: pd.Series,
    transition_covariance: float = 0.01,
    observation_covariance: float = 1.0,
) -> pd.DataFrame:
    """
    Kalman *smoother* (two-pass: forward + backward).
    Gives a better retrospective estimate but cannot be used in real time.
    """
    kf = KalmanFilter(
        transition_matrices    = [1],
        observation_matrices   = [1],
        initial_state_mean     = prices.iloc[0],
        initial_state_covariance = 1,
        observation_covariance = observation_covariance,
        transition_covariance  = transition_covariance,
    )
    state_means, state_covs = kf.smooth(prices.values)

    result = prices.to_frame("close")
    result["kalman_smooth"]     = state_means.flatten()
    result["kalman_smooth_std"] = np.sqrt(state_covs.flatten())
    for w, label in [(20, "ma_20"), (60, "ma_60"), (120, "ma_120")]:
        result[label] = prices.rolling(window=w).mean()
    return result


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------

def sensitivity_analysis(prices: pd.Series) -> pd.DataFrame:
    """
    Try a grid of (Q, R) combinations and return a DataFrame of results
    showing how the filter responds.
    """
    rows = []
    for q in [0.001, 0.01, 0.05, 0.1]:
        for r in [0.5, 1.0, 2.0, 5.0]:
            kf = KalmanFilter(
                transition_matrices    = [1],
                observation_matrices   = [1],
                initial_state_mean     = prices.iloc[0],
                initial_state_covariance = 1,
                observation_covariance = r,
                transition_covariance  = q,
            )
            means, _ = kf.filter(prices.values)
            smoothed = means.flatten()
            lag = prices.values - smoothed
            rows.append({
                "Q (transition)":  q,
                "R (observation)": r,
                "mean_abs_lag":    np.abs(lag).mean(),
                "std_lag":         lag.std(),
                "responsiveness":  q / r,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_kalman(result: pd.DataFrame, symbol: str, save_path: Path = None):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{symbol} — Kalman Filter vs Moving Averages (1-min bars)",
                 fontsize=14, fontweight="bold")

    # ── Panel 1: price + filter + MAs ──────────────────────────────────────
    ax = axes[0]
    ax.plot(result.index, result["close"],   lw=0.6, color="gray",    alpha=0.7, label="close")
    ax.plot(result.index, result["kalman"],  lw=1.5, color="crimson", label="Kalman filter")
    if "kalman_std" in result.columns:
        ax.fill_between(
            result.index,
            result["kalman"] - 2 * result["kalman_std"],
            result["kalman"] + 2 * result["kalman_std"],
            alpha=0.15, color="crimson", label="±2σ"
        )
    for col, color, label in [
        ("ma_20",  "steelblue",  "MA 20m"),
        ("ma_60",  "darkorange", "MA 60m"),
        ("ma_120", "green",      "MA 120m"),
    ]:
        ax.plot(result.index, result[col], lw=1.0, color=color,
                linestyle="--", alpha=0.8, label=label)
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Kalman lag (close - kalman) ───────────────────────────────
    ax = axes[1]
    lag = result["close"] - result["kalman"]
    ax.fill_between(result.index, 0, lag,
                    where=(lag >= 0), color="green", alpha=0.5, label="above estimate")
    ax.fill_between(result.index, 0, lag,
                    where=(lag < 0),  color="red",   alpha=0.5, label="below estimate")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Close − Kalman ($)")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Plot saved → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Save / load results
# ---------------------------------------------------------------------------

def save_results(result: pd.DataFrame, symbol: str, h5path: Path = RESULTS_H5):
    """Persist Kalman filter results to HDF5."""
    h5path.parent.mkdir(parents=True, exist_ok=True)
    with pd.HDFStore(str(h5path), mode="a", complevel=5, complib="blosc") as store:
        store.put(f"kalman/{symbol}", result, format="table", data_columns=True)
    print(f"  Results saved → {h5path}:/kalman/{symbol}")


def load_results(symbol: str, h5path: Path = RESULTS_H5) -> pd.DataFrame:
    with pd.HDFStore(str(h5path), mode="r") as store:
        return store[f"kalman/{symbol}"]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(symbol: str = "AAPL", q: float = 0.01, r: float = 1.0):
    print(f"\n{'='*60}")
    print(f"  Kalman Filter — {symbol}  (Q={q}, R={r})")
    print(f"{'='*60}")

    bars   = load_minute_bars(symbol)
    prices = bars["close"].dropna()
    print(f"  Loaded {len(prices)} minute bars")

    result = fit_kalman(prices, transition_covariance=q, observation_covariance=r)
    print(f"  Kalman filter fitted")

    save_results(result, symbol)
    plot_save = HERE / f"{symbol}_kalman.png"
    plot_kalman(result, symbol, save_path=plot_save)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalman filter on minute bars.")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--q", type=float, default=0.01,
                        help="transition_covariance (process noise)")
    parser.add_argument("--r", type=float, default=1.0,
                        help="observation_covariance (measurement noise)")
    args = parser.parse_args()
    run(symbol=args.symbol, q=args.q, r=args.r)
