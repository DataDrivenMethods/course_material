"""
Momentum Strategy Backtester

This script implements portfolio optimization using momentum signals (UMD)
and various covariance matrix estimation methods (empirical, PCA).

It performs:
1. Downloading CRSP return data for momentum signal stocks
2. Forming covariance matrices on each form_date
3. Mean-variance optimization with constraints
4. Backtesting and P&L analysis
5. Performance reporting

Author: IEOR E4576
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Import optimizer
from optimizer_interface import optimize_portfolio

# Import covariance functions
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def download_crsp_returns(conn, permnos, start_date, end_date):
    """
    Download daily CRSP returns for specified permnos

    Args:
        conn: WRDS connection
        permnos: List or array of permnos
        start_date: Start date for returns
        end_date: End date for returns

    Returns:
        DataFrame with columns: permno, date, ret, prc, vol
    """
    print("\n" + "="*80)
    print("DOWNLOADING CRSP RETURNS")
    print("="*80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Number of unique permnos: {len(permnos)}")

    # Convert permnos to string for SQL
    permno_str = ','.join([str(int(p)) for p in permnos])

    # Download daily returns with market return and risk-free rate
    # Note: Risk-free rate is in ff.factors_daily table
    query = f"""
        SELECT a.permno, a.date, a.ret, a.prc, a.vol, a.shrout,
               b.vwretd as mkt, c.rf
        FROM crsp.dsf as a
        LEFT JOIN crsp.dsi as b ON a.date = b.date
        LEFT JOIN ff.factors_daily as c ON a.date = c.date
        WHERE a.permno IN ({permno_str})
        AND a.date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY a.date, a.permno
    """

    print("\nExecuting CRSP query...")
    crsp_daily = conn.raw_sql(query)

    print(f"Downloaded {len(crsp_daily)} daily return records")
    print(f"Date range: {crsp_daily['date'].min()} to {crsp_daily['date'].max()}")

    # Convert date to datetime
    crsp_daily['date'] = pd.to_datetime(crsp_daily['date'])

    # Calculate excess returns (over risk-free rate)
    crsp_daily['rf'] = crsp_daily['rf'].fillna(0.0)
    crsp_daily['exret'] = crsp_daily['ret'] - crsp_daily['rf']

    return crsp_daily


def compute_empirical_covariance_matrix(returns_data, permnos, current_date,
                                        lookback_days=252, verbose=False):
    """
    Compute empirical sample covariance matrix from historical returns

    Args:
        returns_data: DataFrame with daily returns (permno, date, exret)
        permnos: List of permnos for the universe
        current_date: The date for which to compute the covariance matrix
        lookback_days: Number of days to look back (default: 252)
        verbose: Print diagnostic information

    Returns:
        cov_matrix: Covariance matrix (n_stocks x n_stocks)
        permnos_out: List of permnos (same order as matrix rows/cols)
    """
    if isinstance(current_date, str):
        current_date = pd.to_datetime(current_date)

    if verbose:
        print(f"\nComputing empirical covariance for {current_date.date()}")
        print(f"Universe size: {len(permnos)} stocks")

    # Get historical data
    start_date = current_date - pd.Timedelta(days=lookback_days)
    historical_data = returns_data[
        (returns_data['date'] >= start_date) &
        (returns_data['date'] < current_date) &
        (returns_data['permno'].isin(permnos))
    ].copy()

    if len(historical_data) == 0:
        print(f"Warning: No historical data for {current_date.date()}")
        return None, None

    # Handle duplicates
    historical_data = historical_data.groupby(['date', 'permno'])['exret'].mean().reset_index()

    # Pivot to get returns matrix: dates x stocks
    returns_pivot = historical_data.pivot(index='date', columns='permno', values='exret')
    returns_pivot = returns_pivot.reindex(columns=permnos)

    if verbose:
        print(f"Historical data: {returns_pivot.shape[0]} days")
        print(f"Missing data: {returns_pivot.isna().sum().sum()} / {returns_pivot.size} cells")

    # Compute covariance matrix (handles NaN by pairwise deletion)
    cov_matrix = returns_pivot.cov().values

    if verbose:
        print(f"Mean variance: {np.diag(cov_matrix).mean():.6f}")

    return cov_matrix, permnos


def compute_pca_covariance_matrix(returns_data, permnos, current_date,
                                  n_factors=10, lookback_days=252,
                                  two_stage=True, include_idiosyncratic=True,
                                  verbose=False):
    """
    Compute PCA-based covariance matrix using statistical factors

    Args:
        returns_data: DataFrame with daily returns
        permnos: List of permnos for the universe
        current_date: Date for covariance estimation
        n_factors: Number of principal components (default: 10)
        lookback_days: Lookback period (default: 252)
        two_stage: Use two-stage weighted PCA (default: True)
        include_idiosyncratic: Add idiosyncratic variance (default: True)
        verbose: Print diagnostic information

    Returns:
        cov_matrix: PCA-based covariance matrix
        permnos_out: List of permnos
    """
    if isinstance(current_date, str):
        current_date = pd.to_datetime(current_date)

    if verbose:
        print(f"\nComputing PCA covariance for {current_date.date()}")
        print(f"Universe size: {len(permnos)} stocks")
        print(f"PCA factors: {n_factors}")

    # Get historical data
    start_date = current_date - pd.Timedelta(days=lookback_days)
    historical_data = returns_data[
        (returns_data['date'] >= start_date) &
        (returns_data['date'] < current_date) &
        (returns_data['permno'].isin(permnos))
    ].copy()

    if len(historical_data) == 0:
        print(f"Warning: No historical data for {current_date.date()}")
        return None, None

    # Handle duplicates
    historical_data = historical_data.groupby(['date', 'permno'])['exret'].mean().reset_index()

    # Pivot to get returns matrix
    returns_pivot = historical_data.pivot(index='date', columns='permno', values='exret')
    returns_pivot = returns_pivot.reindex(columns=permnos)

    # Remove stocks with too many missing values
    missing_pct = returns_pivot.isna().sum() / len(returns_pivot)
    valid_stocks = missing_pct[missing_pct < 0.5].index
    returns_clean = returns_pivot[valid_stocks].dropna(how='all', axis=0)

    if verbose:
        print(f"Historical data: {returns_clean.shape[0]} days")
        print(f"Stocks with sufficient data: {returns_clean.shape[1]}")

    # PCA requires n_components <= min(n_samples, n_features)
    n_samples, n_features = returns_clean.shape
    max_components = min(n_samples, n_features) - 1

    if n_factors > max_components:
        if verbose:
            print(f"Warning: Requested {n_factors} factors but only {max_components} available")
            print(f"  (n_samples={n_samples}, n_features={n_features})")
        n_factors = max(1, max_components)
        if verbose:
            print(f"Reducing to {n_factors} factors")

    if n_factors < 1:
        if verbose:
            print("Error: Cannot compute PCA with less than 1 factor")
        return None, None

    # Fill remaining NaN with 0
    returns_clean = returns_clean.fillna(0)

    # Standardize returns
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns_clean)
    returns_scaled = np.asarray(returns_scaled, dtype=np.float64)

    # Perform PCA
    if two_stage:
        # Stage 1: Initial PCA
        pca_stage1 = PCA(n_components=n_factors)
        pca_stage1.fit(returns_scaled)

        # Compute residuals using SVD
        U_full, S_full, Vt_full = np.linalg.svd(returns_scaled, full_matrices=False)
        U_m = U_full[:, :n_factors]
        S_m = np.diag(S_full[:n_factors])
        V_m = Vt_full[:n_factors, :].T

        # Reconstruct systematic component
        R_systematic = U_m @ S_m @ V_m.T

        # Compute residuals
        E = returns_scaled - R_systematic

        # Calculate idiosyncratic std dev
        idio_std = np.std(E, axis=0)
        idio_std = np.where(idio_std > 1e-10, idio_std, 1.0)

        # Stage 2: Weighted PCA
        W_diag = 1.0 / idio_std
        returns_weighted = returns_scaled * W_diag[np.newaxis, :]

        pca = PCA(n_components=n_factors)
        pca.fit(returns_weighted)

        # Transform loadings back
        loadings_weighted = pca.components_.T
        loadings = loadings_weighted * idio_std[:, np.newaxis]
    else:
        # Standard PCA
        pca = PCA(n_components=n_factors)
        pca.fit(returns_scaled)
        loadings = pca.components_.T

    if verbose:
        print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    # Create factor covariance matrix
    avg_return_var = returns_pivot[valid_stocks].var().mean()
    F = np.diag(pca.explained_variance_) * avg_return_var

    # Create full loadings matrix for all permnos
    n_stocks = len(permnos)
    B_full = np.zeros((n_stocks, n_factors))

    for i, permno in enumerate(permnos):
        if permno in valid_stocks:
            idx_clean = list(valid_stocks).index(permno)
            B_full[i, :] = loadings[idx_clean, :]

    # Compute systematic covariance
    systematic_cov = B_full @ F @ B_full.T

    # Add idiosyncratic variance
    if include_idiosyncratic:
        # Compute factor scores and residuals
        factors = returns_scaled @ loadings
        R_systematic = factors @ loadings.T
        E_scaled = returns_scaled - R_systematic
        idio_var_scaled = np.var(E_scaled, axis=0, ddof=1)

        # Transform back to original scale
        stock_stds = scaler.scale_
        idio_var_original = idio_var_scaled * (stock_stds ** 2)

        # Map to all stocks
        D_diag = np.zeros(n_stocks)
        for i, permno in enumerate(permnos):
            if permno in valid_stocks:
                idx_clean = list(valid_stocks).index(permno)
                D_diag[i] = idio_var_original[idx_clean]
            else:
                D_diag[i] = avg_return_var

        cov_matrix = systematic_cov + np.diag(D_diag)
    else:
        cov_matrix = systematic_cov

    if verbose:
        print(f"Mean variance: {np.diag(cov_matrix).mean():.6f}")

    return cov_matrix, permnos


def compute_portfolio_return(returns_data, permnos, weights, current_date):
    """
    Compute portfolio return for a given date

    Args:
        returns_data: DataFrame with returns
        permnos: List of permnos
        weights: Portfolio weights (aligned with permnos)
        current_date: Date to compute return for

    Returns:
        Portfolio return (scalar)
    """
    if isinstance(current_date, str):
        current_date = pd.to_datetime(current_date)

    # Get returns for this date
    date_returns = returns_data[returns_data['date'] == current_date]

    # Create mapping from permno to return
    ret_map = {}
    for _, row in date_returns.iterrows():
        ret_map[row['permno']] = row['ret']

    # Compute weighted return
    port_ret = 0.0
    for i, permno in enumerate(permnos):
        if permno in ret_map:
            ret = ret_map[permno]
            if not pd.isna(ret) and not np.isnan(weights[i]):
                port_ret += weights[i] * ret

    return port_ret


def stage_all(umd, conn, trade_end=None, cost_bps=0.0, stratified=True,
              n_factors=50, risk_aversion=5.0, long_only=False,
              l2_penalty=0.025, max_position=2.0, gross_leverage=5.0,
              lookback_days=252, verbose=False, pca_cov=True):
    """
    Run portfolio optimization backtest for momentum strategy with monthly rebalancing

    Args:
        umd: DataFrame with momentum signals (permno, form_date, momr)
        conn: WRDS connection

    Returns:
        results: Dictionary with portfolio weights, returns, and metadata
    """
    # Configuration (set from the command line, see main)
    FORM_STRATIFIED_PORTFOLIO = stratified
    N_FACTORS = n_factors
    RISK_AVERSION = risk_aversion
    LONG_ONLY = long_only
    L1_PENALTY = l2_penalty      # optimizer argument name; penalty is quadratic
    MAX_POSITION = max_position
    GROSS_LEVERAGE = gross_leverage
    LOOKBACK_DAYS = lookback_days
    VERBOSE = verbose
    PCA_BASED_COVARIANCE = pca_cov

    # Get unique permnos and date range
    all_permnos = umd['permno'].unique()
    form_dates = sorted(umd['form_date'].unique())

    print("\n" + "="*80)
    print("MOMENTUM BACKTEST CONFIGURATION")
    print("="*80)
    print(f"Number of stocks: {len(all_permnos)}")
    print(f"Number of form dates: {len(form_dates)}")
    print(f"Date range: {form_dates[0]} to {form_dates[-1]}")
    print(f"Portfolio type: {'Stratified (decile-weighted)' if FORM_STRATIFIED_PORTFOLIO else 'Optimized (MVO)'}")
    print(f"Trading cost: {cost_bps:.1f} bps one-way per unit of turnover")
    if not FORM_STRATIFIED_PORTFOLIO:
        print(f"Covariance method: {'PCA' if PCA_BASED_COVARIANCE else 'Empirical'}")
        print(f"Risk aversion: {RISK_AVERSION}")
        print(f"Long only: {LONG_ONLY}")
        print(f"Gross leverage: {GROSS_LEVERAGE}")
    print("="*80)

    # Download CRSP returns
    start_date = pd.to_datetime(form_dates[0]) - pd.Timedelta(days=LOOKBACK_DAYS + 30)
    end_date = pd.to_datetime(form_dates[-1]) + pd.offsets.MonthEnd(0) + pd.offsets.MonthEnd(1)
    if trade_end is not None:
        end_date = min(end_date, pd.Timestamp(trade_end))

    crsp_returns = download_crsp_returns(conn, all_permnos, start_date, end_date)

    # Initialize results
    summary = {
        'weights': {}, 'permnos': {}, 'daily_returns': [],
        'dates': [], 'w_a_corr': [], 'benchmark_returns': [], 'rf_returns': []
    }

    # Trading cost: one-way cost_bps per unit of turnover, charged on the
    # first day of each holding period. prev_w maps permno -> weight.
    prev_w = {}
    summary['turnover'] = {}

    # Process each form date
    total_dates = len(form_dates)
    print(f"\nProcessing {total_dates} form dates...")
    print("="*80)

    for i, form_date in enumerate(form_dates):
        dt = pd.to_datetime(form_date)
        dt_str = dt.strftime("%Y-%m-%d")

        # Show progress for every date
        print(f"[{i+1}/{total_dates}] Processing {dt_str}...", end='', flush=True)

        try:
            # Get momentum signals for this date
            date_signals = umd[umd['form_date'] == form_date].copy()

            if len(date_signals) < 10:
                print(f" SKIP (only {len(date_signals)} stocks)")
                continue

            # Get permnos and alphas (momr is the signal, 1=lowest, 10=highest)
            permnos = date_signals['permno'].values
            momr_raw = date_signals['momr'].values  # Raw momentum decile ranking

            if FORM_STRATIFIED_PORTFOLIO:
                # Form stratified portfolio: weight by decile (long winners, short losers)
                # Map deciles linearly: 1->-4.5, 2->-3.5, ..., 9->+3.5, 10->+4.5
                alphas = momr_raw - 5.5

                # Count stocks in each decile
                decile_counts = {}
                for decile in momr_raw:
                    decile_int = int(decile)
                    if decile_int not in decile_counts:
                        decile_counts[decile_int] = 0
                    decile_counts[decile_int] += 1

                # Assign weights: each stock gets (decile_weight / count_in_decile)
                weights = np.zeros(len(permnos))
                for j, decile in enumerate(momr_raw):
                    decile_int = int(decile)
                    decile_weight = alphas[j]  # Already centered
                    weights[j] = decile_weight / decile_counts[decile_int]

                # Normalize so sum of absolute weights = 1 (dollar neutral, equal long/short)
                total_abs_weight = np.abs(weights).sum()
                if total_abs_weight > 0:
                    weights = weights / total_abs_weight

                permnos_cov = permnos

            else:
                # Optimize portfolio using MVO
                # Alphas from -4.5 (decile 1, losers) to +4.5 (decile 10, winners)
                alphas = momr_raw - 5.5

                # Compute covariance matrix
                if PCA_BASED_COVARIANCE:
                    cov_matrix, permnos_cov = compute_pca_covariance_matrix(
                        crsp_returns, permnos, dt, n_factors=N_FACTORS,
                        lookback_days=LOOKBACK_DAYS,
                        two_stage=True, include_idiosyncratic=True, verbose=VERBOSE
                    )
                else:
                    cov_matrix, permnos_cov = compute_empirical_covariance_matrix(
                        crsp_returns, permnos, dt, lookback_days=LOOKBACK_DAYS, verbose=VERBOSE
                    )

                if cov_matrix is None:
                    print(f" SKIP (no covariance)")
                    continue

                # Optimize portfolio
                weights, opt_val, status = optimize_portfolio(
                    alphas, cov_matrix,
                    risk_aversion=RISK_AVERSION, long_only=LONG_ONLY,
                    max_position=MAX_POSITION, l1_penalty=L1_PENALTY,
                    gross_leverage=GROSS_LEVERAGE
                )

            # Store results
            summary['weights'][dt_str] = weights
            summary['permnos'][dt_str] = permnos_cov

            # Turnover vs previous weights (first period: building the book)
            new_w = dict(zip(permnos_cov, np.asarray(weights, dtype=float)))
            turnover = sum(abs(new_w.get(p, 0.0) - prev_w.get(p, 0.0))
                           for p in set(new_w) | set(prev_w))
            prev_w = new_w
            summary['turnover'][dt_str] = turnover
            cost = cost_bps / 1e4 * turnover

            # Compute portfolio returns until next form_date (monthly rebalancing)
            # Determine the end date for this holding period (next form_date or end of data)
            if i + 1 < total_dates:
                next_form_date = pd.to_datetime(form_dates[i + 1])
            else:
                # Last period: hold to the next month end, capped at trade_end
                next_form_date = dt + pd.offsets.MonthEnd(0) + pd.offsets.MonthEnd(1)
                if trade_end is not None:
                    next_form_date = min(next_form_date, pd.Timestamp(trade_end))

            # Returns after form_date through next form_date (inclusive)
            holding_period_data = crsp_returns[
                (crsp_returns['date'] > dt) &
                (crsp_returns['date'] <= next_form_date) &
                (crsp_returns['permno'].isin(permnos_cov))
            ]

            # Calculate returns for each day in holding period
            holding_dates = sorted(holding_period_data['date'].unique())

            for k, hold_date in enumerate(holding_dates):
                hold_date_str = hold_date.strftime("%Y-%m-%d")

                # Compute portfolio return for this holding date
                daily_ret = compute_portfolio_return(crsp_returns, permnos_cov, weights, hold_date)
                if k == 0:
                    daily_ret -= cost
                summary['daily_returns'].append(daily_ret)
                summary['dates'].append(hold_date_str)

                # Get benchmark and risk-free returns
                date_data = crsp_returns[crsp_returns['date'] == hold_date]
                if len(date_data) > 0:
                    rf_ret = date_data['rf'].iloc[0] if 'rf' in date_data.columns else 0.0
                    mkt_ret = date_data['mkt'].iloc[0]

                    # Use different benchmark based on portfolio type
                    if LONG_ONLY:
                        # Long-only: benchmark is market return
                        benchmark_ret = mkt_ret
                    else:
                        # Long-short (including stratified) is self-financing: its
                        # return is already an excess return. No rf, no benchmark.
                        benchmark_ret = 0.0
                        rf_ret = 0.0
                else:
                    benchmark_ret = 0.0
                    rf_ret = 0.0

                summary['benchmark_returns'].append(float(benchmark_ret) if not pd.isna(benchmark_ret) else 0.0)
                summary['rf_returns'].append(float(rf_ret) if not pd.isna(rf_ret) else 0.0)

            # Compute weight-alpha correlation
            weights_clean = np.array(weights, dtype=float)
            alphas_clean = np.array(alphas, dtype=float)
            mask = ~(np.isnan(weights_clean) | np.isnan(alphas_clean))
            if mask.sum() > 1:
                corr, _ = pearsonr(weights_clean[mask], alphas_clean[mask])
            else:
                corr = 0.0

            # Calculate return for the month (for reporting)
            if len(holding_dates) > 0:
                holding_period_returns = summary['daily_returns'][-len(holding_dates):]
                monthly_ret = np.prod([1 + r for r in holding_period_returns]) - 1
            else:
                monthly_ret = 0.0

            # Print completion for this date
            print(f" ({len(holding_dates)} days, ret={monthly_ret:.4f}, turnover={turnover:.3f}, cost={cost*1e4:.2f}bp)")

        except Exception as e:
            print(f" ERROR: {e}")
            summary['daily_returns'].append(0.0)
            summary['benchmark_returns'].append(0.0)
            summary['rf_returns'].append(0.0)
            summary['dates'].append(dt_str)

    print("="*80)
    print(f"Completed processing {len(summary['dates'])} dates")

    # Build results dictionary
    results = {
        'summary': {
            'weights': summary['weights'],
            'permnos': summary['permnos'],
            'daily_returns': summary['daily_returns'],
            'benchmark_returns': summary['benchmark_returns'],
            'rf_returns': summary['rf_returns'],
            'dates': summary['dates'],
            'turnover': summary['turnover'],
            'returns_ts': pd.DataFrame({
                'date': summary['dates'],
                'return': summary['daily_returns'],
                'benchmark': summary['benchmark_returns'],
                'rf': summary['rf_returns']
            })
        },
        'long_only': LONG_ONLY
    }

    return results


def calculate_information_ratio(portfolio_returns, benchmark_returns):
    """Calculate Information Ratio (IR)"""
    active_returns = portfolio_returns - benchmark_returns
    mean_active = active_returns.mean() * 252
    tracking_error = active_returns.std() * np.sqrt(252)
    ir = mean_active / tracking_error if tracking_error > 0 else 0
    return ir


def calculate_sharpe_ratio(portfolio_returns, benchmark_returns, long_only, rf_returns=None):
    """Calculate Sharpe Ratio

    For long-only: excess returns over benchmark (market)
    For long-short: excess returns over risk-free rate

    Args:
        portfolio_returns: Array of portfolio returns
        benchmark_returns: Array of benchmark returns (market)
        long_only: If True, use benchmark; if False, use risk-free rate
        rf_returns: Array of risk-free returns (used when long_only=False)
    """
    if long_only:
        excess_returns = portfolio_returns - benchmark_returns
    else:
        # For long-short, subtract risk-free rate
        if rf_returns is not None:
            excess_returns = portfolio_returns - rf_returns
        else:
            excess_returns = portfolio_returns

    mean_excess = excess_returns.mean() * 252
    std_excess = excess_returns.std() * np.sqrt(252)
    sharpe = mean_excess / std_excess if std_excess > 0 else 0
    return sharpe


def print_performance_summary(results, long_only=None):
    """
    Print performance summary statistics

    Args:
        results: Dictionary from stage_all (can include 'long_only' key) or wrapped results
        long_only: Whether the portfolio is long-only (affects Sharpe calculation).
                   If None, will try to extract from results dict.
    """
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    # Extract long_only from results if not provided
    if long_only is None and 'long_only' in results:
        long_only = results['long_only']
        # Remove long_only from results dict for iteration
        results = {k: v for k, v in results.items() if k != 'long_only'}

    if long_only is None:
        long_only = False  # Default to long-short

    summary_data = []

    for method, data in results.items():
        returns_ts = data['returns_ts']
        returns = returns_ts['return'].values

        # Get benchmark and risk-free returns
        benchmark_returns = returns_ts['benchmark'].values if 'benchmark' in returns_ts.columns else np.zeros_like(returns)
        rf_returns = returns_ts['rf'].values if 'rf' in returns_ts.columns else np.zeros_like(returns)

        # Compute statistics
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (252 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)

        # Information Ratio
        information_ratio = calculate_information_ratio(returns, benchmark_returns)

        # Sharpe Ratio
        sharpe_ratio = calculate_sharpe_ratio(returns, benchmark_returns, long_only, rf_returns)

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = ann_return / downside_vol if downside_vol > 0 else 0

        # Cumulative returns for drawdown
        cumulative_ret = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_ret)
        drawdown = (cumulative_ret - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 else 0

        # Win rate
        win_rate = (returns > 0).sum() / len(returns)

        summary_data.append({
            'Method': method,
            'Cumulative Ret': f"{total_return*100:.2f}%",
            'Annualized Ret': f"{ann_return*100:.2f}%",
            'Ann. StdDev': f"{ann_vol*100:.2f}%",
            'Sharpe': f"{sharpe_ratio:.3f}",
            'Info Ratio': f"{information_ratio:.3f}",
            'Sortino': f"{sortino:.3f}",
            'Calmar': f"{calmar:.3f}",
            'Max DD': f"{max_drawdown*100:.2f}%",
            'Win Rate': f"{win_rate*100:.1f}%"
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")

    # Monthly (calendar month) analysis
    for method, data in results.items():
        print("\n" + "="*80)
        print(f"MONTHLY ANALYSIS: {method}")
        print("="*80)

        returns_ts = data['returns_ts'].copy()
        returns_ts['date'] = pd.to_datetime(returns_ts['date'])
        returns_ts = returns_ts.sort_values('date').reset_index(drop=True)

        # Add year-month column
        returns_ts['year_month'] = returns_ts['date'].dt.to_period('M')

        # Get weights for turnover calculation
        weights_dict = data.get('weights', {})
        dates_list = data.get('dates', [])

        # Rebalance at the form date in month m opens holding month m+1.
        # summary['turnover'] holds sum|w_new - w_old| (two-sided). Reported
        # turnover is the fraction of the gross book replaced:
        # sum|dw| / (2 sum|w|). The first rebalance builds the book from cash
        # and is excluded.
        turnover_by_month = {}
        for k, fd in enumerate(sorted(data.get('turnover', {}))):
            if k == 0:
                continue
            gross = np.abs(np.asarray(weights_dict[fd], dtype=float)).sum()
            if gross > 0:
                turnover_by_month[pd.Period(fd, 'M') + 1] = data['turnover'][fd] / (2.0 * gross)

        # Group by month
        monthly_stats = []
        all_months = returns_ts['year_month'].unique()

        for month in all_months:
            month_data = returns_ts[returns_ts['year_month'] == month].copy()
            month_returns = month_data['return'].values
            month_dates = month_data['date'].values
            month_benchmark = month_data['benchmark'].values if 'benchmark' in month_data.columns else np.zeros_like(month_returns)
            month_rf = month_data['rf'].values if 'rf' in month_data.columns else np.zeros_like(month_returns)

            if len(month_returns) == 0:
                continue

            # Monthly return (cumulative)
            monthly_return = (1 + month_returns).prod() - 1

            # Monthly std dev
            monthly_std = month_returns.std()

            # Information Ratio (vs benchmark)
            monthly_ir = calculate_information_ratio(month_returns, month_benchmark)

            # Sharpe Ratio
            monthly_sharpe = calculate_sharpe_ratio(month_returns, month_benchmark, long_only, month_rf)

            # Sortino ratio (use downside deviation)
            month_downside_returns = month_returns[month_returns < 0]
            month_downside_std = month_downside_returns.std() if len(month_downside_returns) > 0 else 0
            monthly_sortino = (month_returns.mean() / month_downside_std * np.sqrt(252)) if month_downside_std > 0 else 0

            # Calmar Ratio = Annualized Return / |Max Drawdown|
            # Calculate drawdown for this month
            month_cum_ret = (1 + month_returns).cumprod()
            month_running_max = np.maximum.accumulate(month_cum_ret)
            month_drawdown = (month_cum_ret - month_running_max) / month_running_max
            month_max_dd = abs(month_drawdown.min())
            # Annualize the monthly return for Calmar
            ann_monthly_return = (1 + monthly_return) ** (252 / len(month_returns)) - 1 if len(month_returns) > 0 else 0
            monthly_calmar = (ann_monthly_return / month_max_dd) if month_max_dd > 0 else 0

            # Biggest up/down days
            best_day = month_returns.max()
            worst_day = month_returns.min()

            # Turnover of the rebalance that opened this holding month and the
            # implied average holding period (days / fraction replaced)
            month_turnover = turnover_by_month.get(month)
            if month_turnover:
                turnover_str = f"{month_turnover*100:.2f}%"
                holding_str = f"{len(month_returns) / month_turnover:.0f}"
            else:
                turnover_str = holding_str = "n/a"

            monthly_stats.append({
                'Month': str(month),
                'Return': f"{monthly_return*100:.2f}%",
                'Daily StdDev': f"{monthly_std*100:.2f}%",
                'Sharpe': f"{monthly_sharpe:.3f}",
                'Info Ratio': f"{monthly_ir:.3f}",
                'Sortino': f"{monthly_sortino:.3f}",
                'Calmar': f"{monthly_calmar:.3f}",
                'Monthly Turnover': turnover_str,
                'Holding Period (days)': holding_str,
                'Best Day': f"{best_day*100:.2f}%",
                'Worst Day': f"{worst_day*100:.2f}%",
                'Days': len(month_returns)
            })

        if len(monthly_stats) > 0:
            monthly_df = pd.DataFrame(monthly_stats)

            # Calculate cumulative row
            all_returns = returns_ts['return'].values
            all_benchmark = returns_ts['benchmark'].values if 'benchmark' in returns_ts.columns else np.zeros_like(all_returns)
            all_rf = returns_ts['rf'].values if 'rf' in returns_ts.columns else np.zeros_like(all_returns)

            cumulative_return = (1 + all_returns).prod() - 1
            cumulative_std = all_returns.std()
            cumulative_ann_std = cumulative_std * np.sqrt(252)

            # Information Ratio (vs benchmark)
            cumulative_ir = calculate_information_ratio(all_returns, all_benchmark)

            # Sharpe Ratio
            cumulative_sharpe = calculate_sharpe_ratio(all_returns, all_benchmark, long_only, all_rf)

            # Sortino for cumulative
            cumulative_downside_returns = all_returns[all_returns < 0]
            cumulative_downside_std = cumulative_downside_returns.std() if len(cumulative_downside_returns) > 0 else 0
            cumulative_sortino = (all_returns.mean() / cumulative_downside_std * np.sqrt(252)) if cumulative_downside_std > 0 else 0

            # Calmar Ratio = Annualized Return / |Max Drawdown|
            cum_ret_all = (1 + all_returns).cumprod()
            running_max_all = np.maximum.accumulate(cum_ret_all)
            drawdown_all = (cum_ret_all - running_max_all) / running_max_all
            max_dd_all = abs(drawdown_all.min())
            ann_return_all = (1 + cumulative_return) ** (252 / len(all_returns)) - 1 if len(all_returns) > 0 else 0
            cumulative_calmar = (ann_return_all / max_dd_all) if max_dd_all > 0 else 0

            # Average turnover per rebalance and implied holding period
            if turnover_by_month:
                cumulative_turnover = np.mean(list(turnover_by_month.values()))
                days_per_month = len(all_returns) / max(len(monthly_stats), 1)
                cum_turnover_str = f"{cumulative_turnover*100:.2f}%"
                cum_holding_str = f"{days_per_month / cumulative_turnover:.0f}"
            else:
                cum_turnover_str = cum_holding_str = "n/a"
            cumulative_best = all_returns.max()
            cumulative_worst = all_returns.min()

            # Add separator row
            separator_row = pd.DataFrame([{
                'Month': '=' * 10,
                'Return': '=' * 8,
                'Daily StdDev': '=' * 12,
                'Sharpe': '=' * 6,
                'Info Ratio': '=' * 10,
                'Sortino': '=' * 7,
                'Calmar': '=' * 6,
                'Monthly Turnover': '=' * 16,
                'Holding Period (days)': '=' * 20,
                'Best Day': '=' * 8,
                'Worst Day': '=' * 9,
                'Days': '=' * 4
            }])

            # Add cumulative row (use annualized std dev in Daily StdDev column)
            cumulative_row = pd.DataFrame([{
                'Month': 'CUMULATIVE',
                'Return': f"{cumulative_return*100:.2f}%",
                'Daily StdDev': f"{cumulative_ann_std*100:.2f}%",
                'Sharpe': f"{cumulative_sharpe:.3f}",
                'Info Ratio': f"{cumulative_ir:.3f}",
                'Sortino': f"{cumulative_sortino:.3f}",
                'Calmar': f"{cumulative_calmar:.3f}",
                'Monthly Turnover': cum_turnover_str,
                'Holding Period (days)': cum_holding_str,
                'Best Day': f"{cumulative_best*100:.2f}%",
                'Worst Day': f"{cumulative_worst*100:.2f}%",
                'Days': len(all_returns)
            }])

            monthly_df = pd.concat([monthly_df, separator_row, cumulative_row], ignore_index=True)

            # Reorder columns to group risk-adjusted metrics together
            column_order = ['Month', 'Return', 'Daily StdDev', 'Sharpe', 'Info Ratio', 'Sortino', 'Calmar',
                          'Monthly Turnover', 'Holding Period (days)', 'Best Day', 'Worst Day', 'Days']
            monthly_df = monthly_df[column_order]

            print("\n" + monthly_df.to_string(index=False))
            print("="*80 + "\n")


def plot_performance(results, figsize=(14, 10)):
    """
    Plot cumulative returns and performance metrics

    Args:
        results: Dictionary from stage_all (can include 'long_only' key) or wrapped results
        figsize: Figure size
    """
    # Filter out long_only key if present
    if 'long_only' in results:
        results = {k: v for k, v in results.items() if k != 'long_only'}

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Compute cumulative returns
    cumulative_returns = {}
    for method, data in results.items():
        returns_ts = data['returns_ts']
        returns_ts = returns_ts.sort_values('date')
        cumulative_ret = (1 + returns_ts['return']).cumprod()
        cumulative_returns[method] = cumulative_ret.values
        dates = pd.to_datetime(returns_ts['date'].values)

    # Plot 1: Cumulative returns
    ax1 = axes[0, 0]
    for method, cumulative_ret in cumulative_returns.items():
        ax1.plot(dates, cumulative_ret, label=method, linewidth=2, alpha=0.8)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title('Cumulative Returns by Covariance Method')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Daily returns histogram
    ax2 = axes[0, 1]
    for method, data in results.items():
        returns = data['returns_ts']['return'].values
        ax2.hist(returns, bins=50, alpha=0.5, label=method)
    ax2.set_xlabel('Daily Return')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Daily Returns')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Rolling Sharpe ratio (21-day)
    ax3 = axes[1, 0]
    for method, data in results.items():
        returns_ts = data['returns_ts'].copy()
        returns_ts = returns_ts.sort_values('date')
        returns = returns_ts['return'].values
        rolling_sharpe = []
        window = 21
        for i in range(window, len(returns)):
            ret_window = returns[i-window:i]
            sharpe = ret_window.mean() / ret_window.std() * np.sqrt(252) if ret_window.std() > 0 else 0
            rolling_sharpe.append(sharpe)
        ax3.plot(dates[window:], rolling_sharpe, label=method, linewidth=2, alpha=0.8)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Rolling 21-Day Sharpe Ratio')
    ax3.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Drawdown
    ax4 = axes[1, 1]
    for method, cumulative_ret in cumulative_returns.items():
        running_max = np.maximum.accumulate(cumulative_ret)
        drawdown = (cumulative_ret - running_max) / running_max
        ax4.plot(dates, drawdown * 100, label=method, linewidth=2, alpha=0.8)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Drawdown (%)')
    ax4.set_title('Drawdown')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('momentum_mvo_performance.png', dpi=300, bbox_inches='tight')
    print("Saved performance plot to 'momentum_mvo_performance.png'")
    plt.show()


def main(start='1995-01-01', end='1999-12-31', cost_bps=0.0, **config):
    """Run the momentum backtest over the trading period [start, end]"""
    # Load momentum signal (momentum.py builds `umd` and opens the WRDS `conn`).
    # exec() needs an explicit namespace: names it creates inside a function
    # are not visible as locals.
    print("Loading momentum signal from momentum.py...")
    ns = {'TRADE_START': start, 'TRADE_END': end}
    with open('momentum.py') as f:
        exec(f.read(), ns)
    umd = ns['umd']
    conn = ns['conn']

    # Run backtest (currently set to PCA in stage_all configuration)
    print("\n" + "="*80)
    print("RUNNING MOMENTUM BACKTEST")
    print("="*80)
    results = stage_all(umd, conn, trade_end=end, cost_bps=cost_bps, **config)

    conn.close()

    # Wrap results in expected format for reporting functions
    results_dict = {
        'Momentum Portfolio': results['summary']
    }

    # Print performance
    print_performance_summary(results_dict, long_only=results['long_only'])

    # Plot performance
    plot_performance(results_dict)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Momentum backtest")
    parser.add_argument('--start', default='1995-01-01',
                        help="first day of the trading period (YYYY-MM-DD)")
    parser.add_argument('--end', default='1999-12-31',
                        help="last day of the trading period (YYYY-MM-DD)")
    parser.add_argument('--cost', type=float, default=0.0,
                        help="one-way trading cost in bps per unit of turnover (default 0)")

    opt = parser.add_argument_group('MVO portfolio (used with --mvo)')
    opt.add_argument('--mvo', action='store_true',
                     help="mean-variance optimizer instead of the decile-weighted portfolio")
    opt.add_argument('--long-only', action='store_true',
                     help="long-only, sum(w) = 1, benchmarked to the market (default: dollar-neutral long-short)")
    opt.add_argument('--risk-aversion', type=float, default=5.0,
                     help="risk aversion gamma in max alpha'w - (gamma/2) w'Sigma w; larger = more risk-averse (default 5.0)")
    opt.add_argument('--gross-leverage', type=float, default=5.0,
                     help="cap on sum |w| for long-short (default 5.0)")
    opt.add_argument('--max-position', type=float, default=2.0,
                     help="cap on |w_i| (default 2.0)")
    opt.add_argument('--l2-penalty', type=float, default=0.025,
                     help="concentration penalty on sum w_i^2 (default 0.025)")
    opt.add_argument('--empirical-cov', action='store_true',
                     help="sample covariance instead of the PCA factor model")
    opt.add_argument('--n-factors', type=int, default=50,
                     help="PCA factors in the covariance model (default 50)")
    opt.add_argument('--lookback', type=int, default=252,
                     help="covariance lookback in calendar days (default 252)")
    parser.add_argument('--verbose', action='store_true', help="print diagnostics")

    args = parser.parse_args()
    main(start=args.start, end=args.end, cost_bps=args.cost,
         stratified=not args.mvo, long_only=args.long_only,
         risk_aversion=args.risk_aversion, gross_leverage=args.gross_leverage,
         max_position=args.max_position, l2_penalty=args.l2_penalty,
         pca_cov=not args.empirical_cov, n_factors=args.n_factors,
         lookback_days=args.lookback, verbose=args.verbose)
