#!/usr/bin/env python3
"""
Corporate Actions Examples for Students

This script demonstrates different types of corporate actions and their impact
on stock prices using real CRSP data.
"""

import wrds
import pandas as pd
import numpy as np
from crsp_adjusted_prices import get_crsp_adjusted_prices, get_corporate_actions
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

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
        (distributions['distcd'] == 1232) & 
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
    adjusted_prices, distributions = get_crsp_adjusted_prices(
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
        print(split_events[['ex_date', 'distcd', 'facpr', 'facshr']])
    else:
        print("No split event found on specified date")
        # Look for any splits in the period
        any_splits = distributions[distributions['facpr'] > 0]
        if not any_splits.empty:
            print("Splits found in period:")
            print(any_splits[['ex_date', 'distcd', 'facpr', 'facshr']])
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
    
    # Filter for dividends (distcd 1232 = regular cash dividend)
    dividends = distributions[
        (distributions['distcd'] == 1232) & 
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
    adjusted_prices, _ = get_crsp_adjusted_prices(
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
    
    # Simulated data for educational purposes
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Plot 1: Stock Split Example
    ax1.set_title('1. Stock Split Impact', fontsize=14, fontweight='bold')
    
    # Simulate 4:1 split on June 1st (day 152 of year)
    split_date = pd.to_datetime('2023-06-01')
    split_day_index = 151  # May 31st is day 151 (0-indexed)
    
    base_price = 400
    # Pre-split prices (unadjusted)
    prices_before = [base_price + np.random.normal(0, 5) for _ in range(split_day_index + 1)]
    # Post-split prices (unadjusted - should be ~1/4 of pre-split)
    prices_after = [base_price/4 + np.random.normal(0, 1.25) for _ in range(200 - split_day_index - 1)]
    
    # Adjusted prices: back-adjust pre-split prices by dividing by 4
    adjusted_before = [p/4 for p in prices_before]
    # Post-split adjusted prices are same as unadjusted (no further adjustment needed)
    adjusted_after = prices_after.copy()
    
    # Combine dates
    dates_before = dates[:split_day_index + 1]
    dates_after = dates[split_day_index + 1:200]
    
    # Plot unadjusted prices (shows the jump)
    ax1.plot(dates_before, prices_before, 'r-', linewidth=2, alpha=0.8, label='Unadjusted (shows split jump)')
    ax1.plot(dates_after, prices_after, 'r-', linewidth=2, alpha=0.8)
    
    # Plot adjusted prices (smooth series)
    ax1.plot(dates_before, adjusted_before, 'b-', linewidth=2, alpha=0.9, label='Adjusted (continuous)')
    ax1.plot(dates_after, adjusted_after, 'b-', linewidth=2, alpha=0.9)
    
    ax1.axvline(split_date, color='green', linestyle='--', linewidth=2, alpha=0.7, label='4:1 Split Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    # Plot 2: Dividend Pattern
    ax2.set_title('2. Quarterly Dividend Payments', fontsize=14, fontweight='bold')
    
    # Quarterly dividends
    div_dates = pd.date_range('2023-03-15', '2023-12-15', freq='QE')  # Use QE instead of deprecated Q
    div_amounts = [2.50, 2.55, 2.60]  # Match the number of quarters
    
    ax2.bar(div_dates, div_amounts, width=20, color='green', alpha=0.7, label='Dividend Amount')
    ax2.plot(div_dates, div_amounts, 'ro-', markersize=8, linewidth=2, label='Trend')
    ax2.set_ylabel('Dividend ($)')
    ax2.set_xlabel('Quarter')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    # Plot 3: Adjustment Factor Explanation - should align with Graph 1
    ax3.set_title('3. Cumulative Adjustment Factors', fontsize=14, fontweight='bold')
    
    # Use the SAME split date as Graph 1 for consistency
    split_date_graph3 = split_date  # June 1st, same as Graph 1
    
    # Build adjustment factors correctly: 1.0 at end, decreasing backwards through splits
    adj_factors = []
    for date in dates[:200]:
        if date >= split_date_graph3:
            factor = 1.0  # No adjustment needed (current period - post split)
        else:
            factor = 0.25  # Adjust for the 4:1 split (÷ 4)
        adj_factors.append(factor)
    
    ax3.plot(dates[:200], adj_factors, 'purple', linewidth=3, alpha=0.9, label='Cumulative Adjustment Factor')
    ax3.axvline(split_date_graph3, color='green', linestyle='--', linewidth=2, alpha=0.7, label='4:1 Split Date (same as Graph 1)')
    ax3.axhline(y=1.0, color='black', linestyle='-', alpha=0.3, label='Current Period (Factor = 1.0)')
    ax3.set_ylabel('Adjustment Factor')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    # Add explanatory text
    ax3.text(0.02, 0.98, 'Pre-split prices × 0.25\nto match current values\n(4:1 split adjustment)', 
            transform=ax3.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 4: Return Comparison
    ax4.set_title('4. Returns: Adjusted vs Unadjusted', fontsize=14, fontweight='bold')
    
    # Simulate daily returns
    np.random.seed(123)
    normal_returns = np.random.normal(0, 1, 180)
    split_gaps = [0] * 49 + [-25] + [0] * 49 + [-15] + [0] * 80  # Large gaps on split dates - fix length
    
    unadj_returns = [n + g for n, g in zip(normal_returns, split_gaps)]
    adj_returns = normal_returns  # Smooth returns
    
    ax4.plot(dates[:180], unadj_returns, 'r-', alpha=0.7, linewidth=1, label=f'Unadjusted (σ={np.std(unadj_returns):.1f}%)')
    ax4.plot(dates[:180], adj_returns, 'b-', alpha=0.9, linewidth=1, label=f'Adjusted (σ={np.std(adj_returns):.1f}%)')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_ylabel('Daily Return (%)')
    ax4.set_xlabel('Date')
    ax4.legend()
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
   - Regular dividends (distcd 1232) are common
   - Many companies pay quarterly dividends
   - Dividend yield = Annual Dividends / Stock Price

3. ADJUSTMENT FACTORS:
   - facpr: factor to adjust prices
   - facshr: factor to adjust shares
   - distcd: distribution code (1232=dividend, 5523=split)

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
