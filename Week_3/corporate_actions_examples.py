#!/usr/bin/env python3
"""
Corporate Actions Examples for Students

This script demonstrates different types of corporate actions and their impact
on stock prices using real CRSP data.
"""

import wrds
import pandas as pd
import numpy as np
from price_adjustment_example import (get_unadjusted_prices, get_corporate_actions,
                                      calculate_adjustment_factors, build_adjustment_table,
                                      adjust_prices)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def get_split_adjusted_prices(conn, permnos, start_date, end_date):
    """Daily prices with a backward split-adjusted column, plus the distributions used.
    
    Returns (prices, distributions); prices has prc (unadjusted) and prc_adj (split-adjusted).
    """
    prices = get_unadjusted_prices(conn, permnos, start_date, end_date)
    distributions = get_corporate_actions(conn, permnos, start_date, end_date)
    adjustment_table = build_adjustment_table(calculate_adjustment_factors(distributions), prices)
    adjusted = adjust_prices(prices, adjustment_table)
    # CRSP's negative sign only flags a bid/ask-average close; the plots need price levels
    adjusted['prc'] = adjusted['prc'].abs()
    adjusted['prc_adj'] = adjusted['split_adj_prc']
    return adjusted, distributions

def is_cash_dividend(distributions):
    """Regular cash dividends: CIZ distype 'CD' (the old SIZ distcd 1232)."""
    return distributions['distype'].astype(str).str.strip().str.upper() == 'CD'

def create_stock_split_visualization(adjusted_prices, distributions, company_name, split_date):
    """Create visualization showing unadjusted vs adjusted prices around a stock split."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'{company_name}: Stock Split Impact on Prices', fontsize=16, fontweight='bold')
    
    # Convert dates
    adjusted_prices['date'] = pd.to_datetime(adjusted_prices['date'])
    split_dt = pd.to_datetime(split_date)
    
    # Plot 1: Raw vs Adjusted Prices
    ax1.set_title('Unadjusted vs Adjusted Prices (Shows Split Impact)', fontsize=14)
    
    ax1.plot(adjusted_prices['date'], adjusted_prices['prc'], 
            color='red', linewidth=2, alpha=0.8,
            label='Unadjusted Prices (shows split jump)')
    
    ax1.plot(adjusted_prices['date'], adjusted_prices['prc_adj'], 
            color='blue', linewidth=2, alpha=0.9,
            label='Adjusted Prices (smooth series)')
    
    # Mark split date
    ax1.axvline(split_dt, color='green', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Split Date: {split_dt.strftime("%Y-%m-%d")}')
    
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Adjustment Factor Over Time
    ax2.set_title('Price Adjustment Factor (Shows Cumulative Effect)', fontsize=14)
    
    # Calculate adjustment factor
    adjusted_prices['adj_factor'] = adjusted_prices['prc_adj'] / adjusted_prices['prc']
    
    ax2.plot(adjusted_prices['date'], adjusted_prices['adj_factor'], 
            color='purple', linewidth=2, alpha=0.9,
            label='Adjustment Factor (Adj/Unadj)')
    
    ax2.axvline(split_dt, color='green', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Split Date: {split_dt.strftime("%Y-%m-%d")}')
    
    # Add horizontal line at 1.0 for reference
    ax2.axhline(y=1.0, color='black', linestyle='-', alpha=0.3, 
               label='Factor = 1.0 (no adjustment)')
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Adjustment Factor', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Add explanatory text
    split_ratio = None
    if not distributions.empty:
        split_events = distributions[
            (pd.to_datetime(distributions['ex_date']) == split_dt) & 
            (distributions['facpr'] > 0)
        ]
        if not split_events.empty:
            split_ratio = 1.0 + split_events.iloc[0]['facpr']
    
    if split_ratio:
        ax2.text(0.02, 0.98, f'Split Ratio: {split_ratio:g}:1\nPre-split prices ÷ {split_ratio:g}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    safe_company_name = company_name.replace(' ', '_').replace('(', '').replace(')', '').replace(':', '')
    plt.savefig(f'stock_split_{safe_company_name}.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved stock split visualization as 'stock_split_{safe_company_name}.png'")
    
def create_dividend_visualization(distributions, adjusted_prices, company_name):
    """Create visualization showing dividend payments and their impact."""
    
    # Filter for dividends
    dividends = distributions[
        is_cash_dividend(distributions) & 
        (distributions['divamt'] > 0)
    ].copy()
    
    if dividends.empty or adjusted_prices.empty:
        print(f"No dividend data available for {company_name}")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'{company_name}: Dividend Analysis', fontsize=16, fontweight='bold')
    
    # Convert dates
    adjusted_prices['date'] = pd.to_datetime(adjusted_prices['date'])
    dividends['ex_date'] = pd.to_datetime(dividends['ex_date'])
    
    # Plot 1: Stock Price with Dividend Dates
    ax1.set_title('Stock Price with Dividend Payment Dates', fontsize=14)
    
    ax1.plot(adjusted_prices['date'], adjusted_prices['prc'], 
            color='blue', linewidth=2, alpha=0.8,
            label='Stock Price (Unadjusted)')
    
    # Mark dividend dates
    for _, div in dividends.iterrows():
        ax1.axvline(div['ex_date'], color='green', linestyle=':', alpha=0.6)
        # Add dividend amount as text
        price_at_div = adjusted_prices[adjusted_prices['date'] <= div['ex_date']]
        if not price_at_div.empty:
            y_pos = price_at_div.iloc[-1]['prc'] * 1.05  # Place text above price
            ax1.text(div['ex_date'], y_pos, f'${div["divamt"]:.2f}', 
                    rotation=90, va='bottom', ha='center', fontsize=8, 
                    color='green', alpha=0.8)
    
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Dividend History
    ax2.set_title('Dividend Payment History', fontsize=14)
    
    # Sort dividends by date
    dividends_sorted = dividends.sort_values('ex_date')
    
    # Create bar chart of dividends
    ax2.bar(dividends_sorted['ex_date'], dividends_sorted['divamt'], 
           color='green', alpha=0.7, width=30,  # width in days
           label='Dividend Amount')
    
    # Calculate and show trend
    if len(dividends_sorted) > 1:
        x_numeric = mdates.date2num(dividends_sorted['ex_date'])
        z = np.polyfit(x_numeric, dividends_sorted['divamt'], 1)
        p = np.poly1d(z)
        ax2.plot(dividends_sorted['ex_date'], p(x_numeric), 
                'r--', alpha=0.8, linewidth=2, label='Trend')
        
        # Show trend direction
        slope = z[0] * 365  # Convert to annual change
        trend_text = f'Annual Change: ${slope:.3f}' if slope != 0 else 'Stable'
        ax2.text(0.02, 0.98, trend_text, transform=ax2.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Dividend Amount ($)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Add summary statistics
    if len(dividends) > 0:
        total_annual = dividends.groupby(dividends['ex_date'].dt.year)['divamt'].sum()
        avg_annual = total_annual.mean() if len(total_annual) > 0 else 0
        recent_price = adjusted_prices['prc'].iloc[-1] if not adjusted_prices.empty else 0
        yield_est = (avg_annual / recent_price * 100) if recent_price > 0 else 0
        
        stats_text = f'Avg Annual Dividend: ${avg_annual:.2f}\nEst. Yield: {yield_est:.1f}%'
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    safe_company_name = company_name.replace(' ', '_').replace('(', '').replace(')', '').replace(':', '')
    plt.savefig(f'dividends_{safe_company_name}.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved dividend visualization as 'dividends_{safe_company_name}.png'")

# Example companies with interesting corporate actions
EXAMPLES = {
    'stock_splits': {
        'Apple (4:1 split)': {'permno': 14593, 'period': ('2020-06-01', '2020-09-30'), 'split_date': '2020-08-31'},
        'Google (20:1 split)': {'permno': 14542, 'period': ('2022-06-01', '2022-08-31'), 'split_date': '2022-07-18'},
        'Amazon (20:1 split)': {'permno': 84788, 'period': ('2022-05-01', '2022-07-31'), 'split_date': '2022-06-06'},
    },
    'high_dividend': {
        'Alexanders Inc': {'permno': 49330, 'period': ('2020-01-01', '2023-12-31')},
        'BlackRock Inc': {'permno': 87267, 'period': ('2020-01-01', '2023-12-31')},
        'Broadcom Inc': {'permno': 93002, 'period': ('2020-01-01', '2023-12-31')},
    }
}

def demonstrate_stock_split(conn, company_name, permno, start_date, end_date, split_date):
    """Demonstrate the impact of a stock split on prices."""
    print(f"\n{'='*60}")
    print(f"STOCK SPLIT EXAMPLE: {company_name}")
    print(f"PERMNO: {permno}, Split Date: {split_date}")
    print(f"{'='*60}")
    
    # Get data
    adjusted_prices, distributions = get_split_adjusted_prices(
        conn=conn,
        permnos=[permno],
        start_date=start_date,
        end_date=end_date
    )
    
    if adjusted_prices.empty:
        print("No price data found")
        return
    
    # Show the split event
    distributions['ex_date'] = pd.to_datetime(distributions['ex_date'])
    split_dt = pd.to_datetime(split_date)
    split_events = distributions[
        (distributions['ex_date'] == split_dt) & 
        (distributions['facpr'] > 0)
    ]
    
    if not split_events.empty:
        split_ratio = 1.0 + split_events.iloc[0]['facpr']
        print(f"Stock Split Ratio: {split_ratio:g}:1")
        print(f"Split Details:")
        print(split_events[['ex_date', 'distype', 'disdetailtype', 'facpr', 'facshr']])
    else:
        print("No split event found on specified date")
        # Look for any splits in the period
        any_splits = distributions[distributions['facpr'] > 0]
        if not any_splits.empty:
            print("Splits found in period:")
            print(any_splits[['ex_date', 'distype', 'disdetailtype', 'facpr', 'facshr']])
            split_ratio = 1.0 + any_splits.iloc[0]['facpr']
        else:
            print("No splits found in the data")
            return
    
    # Show price impact
    print(f"\nPrice Impact Analysis:")
    print(f"Before split prices are adjusted by factor of {1/split_ratio:.6f}")
    
    # Get prices around split date
    split_dt = pd.to_datetime(split_date)
    adjusted_prices['date'] = pd.to_datetime(adjusted_prices['date'])
    before_split = adjusted_prices[adjusted_prices['date'] < split_dt].tail(5)
    after_split = adjusted_prices[adjusted_prices['date'] >= split_dt].head(5)
    
    print(f"\nPrices BEFORE split (adjusted vs unadjusted):")
    for _, row in before_split.iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d')}: "
              f"Unadj=${row['prc']:8.2f}, Adj=${row['prc_adj']:8.2f}, "
              f"Ratio={row['prc_adj']/row['prc']:6.4f}")
    
    print(f"\nPrices AFTER split (should be similar):")
    for _, row in after_split.iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d')}: "
              f"Unadj=${row['prc']:8.2f}, Adj=${row['prc_adj']:8.2f}, "
              f"Ratio={row['prc_adj']/row['prc']:6.4f}")
    
    # Calculate actual vs theoretical post-split price
    if not before_split.empty and not after_split.empty:
        last_pre_split_adj = before_split.iloc[-1]['prc_adj']
        first_post_split_unadj = after_split.iloc[0]['prc']
        theoretical_post_split = last_pre_split_adj
        
        print(f"\nSplit Verification:")
        print(f"  Last adjusted price before split: ${last_pre_split_adj:.2f}")
        print(f"  First actual price after split:   ${first_post_split_unadj:.2f}")
        print(f"  Difference: ${abs(theoretical_post_split - first_post_split_unadj):.2f}")
    
    # Create visualization
    print(f"\n📊 Creating split visualization...")
    create_stock_split_visualization(adjusted_prices, distributions, company_name, split_date)

def demonstrate_dividends(conn, company_name, permno, start_date, end_date):
    """Demonstrate dividend payments and their frequency."""
    print(f"\n{'='*60}")
    print(f"DIVIDEND EXAMPLE: {company_name}")
    print(f"PERMNO: {permno}")
    print(f"{'='*60}")
    
    # Get distributions only
    distributions = get_corporate_actions(
        conn=conn,
        permnos=[permno],
        start_date=start_date,
        end_date=end_date
    )
    
    if distributions.empty:
        print("No distributions found")
        return
    
    # Filter for regular cash dividends (distype CD)
    dividends = distributions[
        is_cash_dividend(distributions) & 
        (distributions['divamt'] > 0)
    ].copy()
    
    if dividends.empty:
        print("No cash dividends found")
        return
    
    print(f"Found {len(dividends)} dividend payments:")
    
    # Sort by date and show dividend history
    dividends['ex_date'] = pd.to_datetime(dividends['ex_date'])
    dividends = dividends.sort_values('ex_date')
    dividends['year'] = dividends['ex_date'].dt.year
    
    print(f"\nDividend History:")
    for _, div in dividends.iterrows():
        print(f"  {div['ex_date'].strftime('%Y-%m-%d')}: ${div['divamt']:.3f}")
    
    # Annual summary
    annual_dividends = dividends.groupby('year')['divamt'].agg(['count', 'sum']).reset_index()
    annual_dividends.columns = ['Year', 'Payments', 'Total_Dividends']
    
    print(f"\nAnnual Dividend Summary:")
    print(annual_dividends.to_string(index=False))
    
    # Calculate dividend yield (approximate)
    recent_dividends = dividends[dividends['year'] == dividends['year'].max()]
    if len(recent_dividends) >= 4:  # Quarterly dividends
        quarterly_div = recent_dividends['divamt'].mean()
        annual_div = quarterly_div * 4
        print(f"\nEstimated Annual Dividend: ${annual_div:.2f}")
        print(f"Based on recent quarterly average: ${quarterly_div:.3f}")
    
    # Create dividend visualization
    print(f"\n📊 Creating dividend visualization...")
    # Get price data for dividend visualization
    adjusted_prices, _ = get_split_adjusted_prices(
        conn=conn,
        permnos=[permno],
        start_date=start_date,
        end_date=end_date
    )
    create_dividend_visualization(distributions, adjusted_prices, company_name)

def create_summary_visualization():
    """Create a summary visualization showing key corporate actions concepts."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Corporate Actions: Key Concepts for Students', fontsize=16, fontweight='bold')
    
    # Simulated data for educational purposes. One price path is shared by all four graphs.
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Corporate actions: a 4:1 split on June 1st and quarterly cash dividends.
    # Dividend amounts are per share AS PAID, so post-split dividends are per new share.
    split_date = pd.to_datetime('2023-06-01')
    split_ratio = 4
    dividends = pd.Series(
        [6.00, 6.00, 1.60, 1.60],
        index=pd.to_datetime(['2023-02-15', '2023-05-15', '2023-08-15', '2023-11-15'])
    )
    
    # Unadjusted (actually traded) price: random walk that drops by the dividend
    # on each ex-date and by the split ratio on the split date
    daily_ret = np.random.normal(0, 0.012, len(dates))
    unadjusted = np.empty(len(dates))
    unadjusted[0] = 400
    for i in range(1, len(dates)):
        p = unadjusted[i - 1] * (1 + daily_ret[i])
        if dates[i] == split_date:
            p /= split_ratio
        if dates[i] in dividends.index:
            p -= dividends[dates[i]]
        unadjusted[i] = p
    
    # Cumulative backward adjustment factors (1.0 today, applied to earlier prices):
    #   split factor    = 1/ratio for every split after the date (what CRSP's cfacpr does)
    #   dividend factor = product of (1 - D / prior close) for every ex-date after the date
    split_factor = np.ones(len(dates))
    div_factor = np.ones(len(dates))
    s_f, d_f = 1.0, 1.0
    for i in range(len(dates) - 1, -1, -1):
        split_factor[i], div_factor[i] = s_f, d_f
        if dates[i] == split_date:
            s_f /= split_ratio
        if dates[i] in dividends.index:
            d_f *= 1 - dividends[dates[i]] / unadjusted[i - 1]
    
    split_adjusted = unadjusted * split_factor               # price return series
    total_adjusted = unadjusted * split_factor * div_factor  # total return series
    
    def mark_events(ax, split_label='4:1 Split Date', div_label='Ex-dividend Dates'):
        ax.axvline(split_date, color='green', linestyle='--', linewidth=2, alpha=0.7, label=split_label)
        for j, ex in enumerate(dividends.index):
            ax.axvline(ex, color='darkorange', linestyle=':', linewidth=1.5, alpha=0.8,
                       label=div_label if j == 0 else None)
    
    # Plot 1: Stock Split Example
    ax1.set_title('1. Stock Split Impact', fontsize=14, fontweight='bold')
    ax1.plot(dates, unadjusted, 'r-', linewidth=2, alpha=0.8, label='Unadjusted (shows split jump)')
    ax1.plot(dates, split_adjusted, 'b-', linewidth=2, alpha=0.9, label='Split-adjusted (continuous)')
    mark_events(ax1)
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    # Plot 2: Dividend Pattern
    ax2.set_title('2. Quarterly Dividend Payments', fontsize=14, fontweight='bold')
    
    # Dividends as paid vs restated per post-split share (divided by the split ratio)
    div_split_adj = dividends * split_factor[dates.get_indexer(dividends.index)]
    ax2.bar(dividends.index, dividends.values, width=20, color='green', alpha=0.7,
            label='As paid (per share at the time)')
    ax2.plot(div_split_adj.index, div_split_adj.values, 'ro-', markersize=8, linewidth=2,
             label='Split-adjusted (per current share)')
    for ex, d in dividends.items():
        ax2.annotate(f'${d:.2f}', (ex, d), xytext=(0, 12), textcoords='offset points',
                     ha='center', va='bottom', fontsize=9)
    ax2.axvline(split_date, color='green', linestyle='--', linewidth=2, alpha=0.7, label='4:1 Split Date')
    ax2.set_xlim(dates[0], dates[-1])
    ax2.set_ylim(0, dividends.max() * 1.15)
    ax2.set_ylabel('Dividend ($)')
    ax2.set_xlabel('Quarter')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    # Plot 3: Adjustment Factors - all on ONE axis so their sizes are directly comparable
    ax3.set_title('3. Cumulative Adjustment Factors', fontsize=14, fontweight='bold')
    
    total_factor = split_factor * div_factor
    ax3.plot(dates, split_factor, 'purple', linewidth=3, alpha=0.9,
             label='Split factor (CRSP cfacpr)')
    ax3.plot(dates, div_factor, color='darkorange', linewidth=2.5, alpha=0.9,
             label=f'Dividend factor (Jan: {div_factor[0]:.3f})')
    ax3.plot(dates, total_factor, 'k--', linewidth=1.5, alpha=0.9,
             label=f'Total-return factor = split × dividend (Jan: {total_factor[0]:.3f})')
    mark_events(ax3)
    ax3.axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel('Adjustment Factor')
    ax3.set_xlabel('Date')
    ax3.legend(loc='center right')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    # Add explanatory text
    div_yields = [dividends[ex] / unadjusted[dates.get_loc(ex) - 1] for ex in dividends.index]
    ax3.text(0.02, 0.72,
             'Split: pre-split prices × 0.25\n'
             'Dividend: prices before each ex-date\n'
             '    × (1 − D / prior close)\n'
             f'    each dividend ≈ {np.mean(div_yields):.1%} of price,\n'
             '    so each step is small',
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 4: Backward-adjusted price series and the returns they imply.
    # All series equal the actual traded price on the last date (factor = 1.0, as in Graph 3)
    # and differ only in how earlier prices are restated.
    ax4.set_title('4. Returns: Unadjusted vs Split-adjusted vs Total', fontsize=14, fontweight='bold')
    
    def period_return(series):
        return series[-1] / series[0] - 1
    
    ax4.plot(dates, unadjusted, 'r-', alpha=0.8, linewidth=1.5,
             label=f'Unadjusted price (return {period_return(unadjusted):+.0%})')
    ax4.plot(dates, split_adjusted, 'b-', alpha=0.9, linewidth=1.5,
             label=f'Split-adjusted price (return {period_return(split_adjusted):+.1%})')
    ax4.plot(dates, total_adjusted, 'k-', alpha=0.9, linewidth=1.5,
             label=f'Split + dividend adjusted (total return {period_return(total_adjusted):+.1%})')
    mark_events(ax4)
    
    # Log scale: a constant % gap (the dividend adjustment) is a constant vertical distance
    ax4.set_yscale('log')
    ticks = [50, 75, 100, 150, 200, 300, 400, 600]
    ax4.set_yticks(ticks)
    ax4.set_yticklabels([f'${t}' for t in ticks])
    ax4.yaxis.set_minor_formatter(plt.NullFormatter())
    ax4.set_ylim(55, 650)
    
    i_split = dates.get_loc(split_date)
    split_ret = unadjusted[i_split] / unadjusted[i_split - 1] - 1
    ax4.annotate(f'Unadjusted: spurious {split_ret:.0%} "return"\non split date',
                 xy=(split_date, np.sqrt(unadjusted[i_split - 1] * unadjusted[i_split])),
                 xytext=(0.52, 0.80), textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='red'),
                 bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
    ax4.text(0.02, 0.03,
             'Backward adjustment: all series match the traded price today;\n'
             'the gap between blue and black opens at each ex-date going back',
             transform=ax4.transAxes, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax4.set_ylabel('Price ($, log scale)')
    ax4.set_xlabel('Date')
    ax4.legend(loc='center right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    plt.tight_layout()
    plt.savefig('corporate_actions_summary.png', dpi=300, bbox_inches='tight')
    print("✅ Saved corporate actions summary as 'corporate_actions_summary.png'")

def main():
    """Main demonstration function."""
    print("CRSP Corporate Actions Examples for Students")
    print("=" * 50)
    
    # Initialize WRDS connection
    conn = wrds.Connection(wrds_username='cpehlivanian')
    
    # Demonstrate stock splits
    print("\n🔄 STOCK SPLITS SECTION")
    for company, details in EXAMPLES['stock_splits'].items():
        try:
            demonstrate_stock_split(
                conn, company, details['permno'], 
                details['period'][0], details['period'][1], 
                details['split_date']
            )
        except Exception as e:
            print(f"Error with {company}: {e}")
    
    # Demonstrate dividends
    print("\n💰 DIVIDEND PAYMENTS SECTION")
    for company, details in EXAMPLES['high_dividend'].items():
        try:
            demonstrate_dividends(
                conn, company, details['permno'],
                details['period'][0], details['period'][1]
            )
        except Exception as e:
            print(f"Error with {company}: {e}")
    
    # Summary for students
    print("\n" + "="*60)
    print("SUMMARY FOR STUDENTS")
    print("="*60)
    print("""
Key Learning Points:

1. STOCK SPLITS:
   - Pre-split prices are adjusted by the split ratio
   - Post-split prices remain unchanged
   - This maintains comparability across time
   - Example: 4:1 split means pre-split prices ÷ 4

2. DIVIDENDS:
   - Ex-dividend date: when stock trades without dividend
   - Regular cash dividends (distype CD) are common
   - Many companies pay quarterly dividends
   - Dividend yield = Annual Dividends / Stock Price

3. ADJUSTMENT FACTORS:
   - facpr: factor to adjust prices
   - facshr: factor to adjust shares
   - distype/disdetailtype: event type (CD = cash dividend, FRS/STKSPL = split)
   - CRSP's cfacpr adjusts for splits only; cash dividends enter
     through RET, so a total-return series needs both

4. Why Adjust Prices?
   - Makes historical analysis meaningful
   - Enables proper return calculations
   - Accounts for value changes due to corporate actions
""")
    
    # Create educational summary visualization
    print("\n📊 CREATING EDUCATIONAL SUMMARY VISUALIZATIONS")
    print("="*60)
    create_summary_visualization()
    
    conn.close()
    print("\nDemonstration complete!")

if __name__ == "__main__":
    main()
