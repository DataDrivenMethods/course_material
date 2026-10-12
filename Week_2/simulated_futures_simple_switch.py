import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def simulate_monthly_contracts(base_price: float = 100.0,
                             start_date: str = '2020-01-01',
                             end_date: str = '2023-12-31',
                             volatility: float = 0.02,
                             target_trading_days: int = 200) -> pd.DataFrame:
    """
    Simulate monthly futures contracts for demonstration purposes.

    This creates realistic futures contract data with:
    - Multiple monthly expiration cycles
    - Each contract has approximately the target number of trading days
    - Staggered start dates for realistic front-month progression
    - Contango/backwardation patterns
    - Volume and open interest patterns

    Args:
        base_price: Starting price level
        start_date: Start date for simulation
        end_date: End date for simulation
        volatility: Daily volatility for price movements
        target_trading_days: Target number of trading days per contract (default 200)

    Returns:
        DataFrame with simulated contract data
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Generate monthly expiration dates (3rd Friday of each month)
    expiration_dates = []
    current_date = start_dt.replace(day=1)
    while current_date <= end_dt + timedelta(days=365):  # Extra buffer for forward contracts
        # Find 3rd Friday of the month
        first_friday = current_date + timedelta(days=(4 - current_date.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
        expiration_dates.append(third_friday)
        
        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    # Create trading days
    trading_days = pd.bdate_range(start=start_date, end=end_date)
    
    contracts_data = []
    # Use random seed for different data each run
    np.random.seed(None)
    
    for exp_date in expiration_dates:
        if exp_date < start_dt:
            continue

        # Contract stops trading 1 day before expiration (more realistic)
        stop_trading = min(end_dt, exp_date - timedelta(days=1))

        # Calculate start date to get exactly target_trading_days
        # Work backwards from stop_trading date
        available_days = [d for d in trading_days if d <= stop_trading]

        if len(available_days) < target_trading_days:
            continue

        # Take exactly target_trading_days before expiration
        contract_days = available_days[-target_trading_days:]

        # Only include contract if it starts after our simulation start date
        if contract_days[0] < start_dt:
            continue

        # Only include if the contract has enough overlap with our simulation period
        simulation_days = [d for d in contract_days if start_dt <= d <= end_dt]
        if len(simulation_days) < 30:  # Reduced minimum for more contracts
            continue

        # Use the days that overlap with our simulation period
        contract_days = simulation_days

        if not contract_days:
            continue
            
        # Generate contract identifier
        month_code = exp_date.strftime('%Y%m')
        contract_name = f"Contract_{month_code}"
        
        # Generate price series with contango/backwardation that converges
        days_to_expiry = [(exp_date - d).days for d in contract_days]
        
        # Create much more volatile price paths with frequent direction changes
        # Use higher volatility and add mean reversion
        daily_volatility = volatility * 3.0  # Triple the base volatility

        # Generate random walk with occasional large moves and mean reversion
        returns = []
        for i in range(len(contract_days)):
            # Base random return
            base_return = np.random.normal(0, daily_volatility)

            # Add occasional large jumps (5% chance of 2x larger move)
            if np.random.random() < 0.05:
                base_return *= 2.0

            # Add mean reversion tendency (pull back toward starting price)
            if i > 0 and len(returns) > 10:
                recent_avg = np.mean([ret for ret in returns[-10:]])
                mean_reversion = -0.1 * recent_avg  # 10% reversion
                base_return += mean_reversion

            returns.append(base_return)

        # Build price series from returns
        underlying_prices = [base_price]
        for ret in returns[1:]:
            underlying_prices.append(underlying_prices[-1] * (1 + ret))

        # Create stronger and more variable term structure differences
        # Each contract gets a random bias that changes over time
        contract_premium = np.random.normal(0, 0.15)  # Random premium for this contract

        term_structure = []
        for i, (underlying, dte) in enumerate(zip(underlying_prices, days_to_expiry)):
            # Variable term structure that creates larger price differences
            time_premium = np.random.normal(0.03, 0.02) * (dte / 365)  # Random contango/backwardation

            # Contract-specific premium that varies over time
            time_varying_premium = contract_premium * (1 + 0.5 * np.sin(i / 20))  # Oscillating effect
            contract_effect = time_varying_premium * (dte / 200)

            # Add extra randomness to each price point
            random_noise = np.random.normal(0, 0.01)

            # This creates significant and variable price differences between contracts
            final_price = underlying * (1 + time_premium + contract_effect + random_noise)
            term_structure.append(final_price)
        
        # Volume pattern (higher closer to expiration)
        max_volume = 10000
        volumes = [max_volume * (1 - dte / 180) for dte in days_to_expiry]
        volumes = [max(100, v) for v in volumes]  # Minimum volume
        
        # Open interest pattern (builds up, then declines near expiration)
        max_oi = 50000
        open_interests = [max_oi * (1 - abs(dte - 90) / 90) for dte in days_to_expiry]
        open_interests = [max(100, oi) for oi in open_interests]
        
        for i, day in enumerate(contract_days):
            price = term_structure[i]
            # Add some intraday variation
            high = price * (1 + abs(np.random.normal(0, volatility/4)))
            low = price * (1 - abs(np.random.normal(0, volatility/4)))
            open_price = price * (1 + np.random.normal(0, volatility/8))
            
            contracts_data.append({
                'contract_name': contract_name,
                'date': day,
                'expiration_date': exp_date,
                'days_to_expiry': days_to_expiry[i],
                'open': open_price,
                'high': high,
                'low': low,
                'settlement': price,
                'volume': volumes[i],
                'open_interest': open_interests[i]
            })
    
    return pd.DataFrame(contracts_data)

def create_simple_switch_continuous(futures_df: pd.DataFrame,
                                  days_before_expiry: int = 5) -> pd.DataFrame:
    """
    Create continuous futures series using simple switch rolling.

    Args:
        futures_df: DataFrame with futures contract data
        days_before_expiry: Number of days before expiry to switch contracts

    Returns:
        DataFrame with continuous series
    """
    if futures_df.empty:
        return pd.DataFrame()

    # Ensure dates are datetime
    df = futures_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['expiration_date'] = pd.to_datetime(df['expiration_date'])

    # Sort by date and expiration
    df = df.sort_values(['date', 'expiration_date'])

    continuous_data = []
    all_dates = sorted(df['date'].unique())

    for date in all_dates:
        # Get all contracts available on this date
        available_contracts = df[df['date'] == date].copy()

        if available_contracts.empty:
            continue

        # Calculate days to expiry
        available_contracts['days_to_expiry'] = (
            available_contracts['expiration_date'] - date
        ).dt.days

        # Filter out expired contracts and sort by expiration
        active_contracts = available_contracts[available_contracts['days_to_expiry'] > 0]
        active_contracts = active_contracts.sort_values('days_to_expiry')

        if active_contracts.empty:
            continue

        # Select contract based on switching rule
        front_contract = active_contracts.iloc[0]

        # Switch to next contract if front contract is too close to expiry
        if (front_contract['days_to_expiry'] <= days_before_expiry and
            len(active_contracts) > 1):
            selected_contract = active_contracts.iloc[1]
        else:
            selected_contract = front_contract

        # Add to continuous series
        continuous_data.append({
            'date': date,
            'contract_name': selected_contract['contract_name'],
            'expiration_date': selected_contract['expiration_date'],
            'days_to_expiry': selected_contract['days_to_expiry'],
            'open': selected_contract['open'],
            'high': selected_contract['high'],
            'low': selected_contract['low'],
            'settlement': selected_contract['settlement'],
            'volume': selected_contract['volume'],
            'open_interest': selected_contract['open_interest']
        })

    return pd.DataFrame(continuous_data)

def plot_continuous_series(continuous_df: pd.DataFrame,
                          original_df: pd.DataFrame,
                          days_before_expiry: int):
    """
    Plot the continuous futures series along with individual contracts.

    Args:
        continuous_df: DataFrame with continuous series
        original_df: DataFrame with original individual contracts
        days_before_expiry: Days before expiry used for switching
    """
    if continuous_df.empty:
        print("No data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Simple Switch Continuous Futures (Switch {days_before_expiry} days before expiry)',
                 fontsize=16, fontweight='bold')

    # Plot 1: Individual contracts vs continuous series
    ax1.set_title('Individual Contracts vs Continuous Series', fontsize=14)

    # Plot a few individual contracts for context
    unique_contracts = original_df['contract_name'].unique()
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

    for i, contract in enumerate(unique_contracts[:6]):  # Show first 6 contracts
        contract_data = original_df[original_df['contract_name'] == contract]
        if not contract_data.empty:
            expiry = contract_data['expiration_date'].iloc[0]
            ax1.plot(contract_data['date'], contract_data['settlement'],
                    color=colors[i % len(colors)], alpha=0.6, linewidth=1,
                    label=f'{contract} (exp: {expiry.strftime("%b %Y")})')

    # Plot continuous series on top
    continuous_df = continuous_df.sort_values('date')
    ax1.plot(continuous_df['date'], continuous_df['settlement'],
            color='black', linewidth=3, alpha=0.9,
            label='Continuous Series')

    # Mark roll dates (contract switches)
    prev_contract = None
    roll_dates = []
    for _, row in continuous_df.iterrows():
        if row['contract_name'] != prev_contract and prev_contract is not None:
            ax1.axvline(row['date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
            roll_dates.append(row['date'])
        prev_contract = row['contract_name']

    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Plot 2: Contract transitions and roll impact
    ax2.set_title('Price Changes at Roll Dates', fontsize=14)

    # Show price jumps at roll dates
    if roll_dates:
        price_jumps = []
        jump_dates = []

        for roll_date in roll_dates:
            # Get price before and after roll
            before_roll = continuous_df[continuous_df['date'] < roll_date].tail(1)
            at_roll = continuous_df[continuous_df['date'] == roll_date].head(1)

            if not before_roll.empty and not at_roll.empty:
                price_before = before_roll['settlement'].iloc[0]
                price_after = at_roll['settlement'].iloc[0]
                jump = price_after - price_before
                price_jumps.append(jump)
                jump_dates.append(roll_date)

        if price_jumps:
            ax2.bar(jump_dates, price_jumps,
                   color=['green' if x > 0 else 'red' for x in price_jumps],
                   alpha=0.7, width=10)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_ylabel('Price Jump ($)', fontsize=12)
            ax2.set_xlabel('Roll Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No roll dates found',
                    transform=ax2.transAxes, ha='center', va='center')

    plt.tight_layout()
    plt.savefig(f'simple_switch_continuous_{days_before_expiry}days.png',
                dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot as 'simple_switch_continuous_{days_before_expiry}days.png'")

def demonstrate_simple_switch():
    """Demonstrate the simple switch continuous futures functionality."""
    print("🚀 SIMPLE SWITCH CONTINUOUS FUTURES DEMONSTRATION")
    print("=" * 60)

    # Create simulated futures data
    print("Creating simulated futures contracts...")
    futures_data = simulate_monthly_contracts(
        base_price=100.0,
        start_date='2023-01-01',
        end_date='2023-12-31',
        volatility=0.03,
        target_trading_days=120
    )

    print(f"✅ Generated {len(futures_data):,} observations")
    print(f"📊 Contracts: {futures_data['contract_name'].nunique()} individual monthly contracts")

    # Show contract summary
    print(f"\n📋 CONTRACT SUMMARY:")
    unique_contracts = futures_data['contract_name'].unique()
    for i, contract in enumerate(unique_contracts[:8]):  # Show first 8
        contract_data = futures_data[futures_data['contract_name'] == contract]
        start_date = contract_data['date'].min()
        end_date = contract_data['date'].max()
        expiry = contract_data['expiration_date'].iloc[0]
        trading_days = len(contract_data)
        price_range = f"${contract_data['settlement'].min():.2f}-${contract_data['settlement'].max():.2f}"

        print(f"{i+1:2d}. {contract}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"    Expires: {expiry.strftime('%Y-%m-%d')} | {trading_days} days | {price_range}")

    # Test different switching thresholds
    switch_days_list = [1, 5, 10]

    for days_before in switch_days_list:
        print(f"\n🔄 CREATING CONTINUOUS SERIES (Switch {days_before} days before expiry)")

        continuous_series = create_simple_switch_continuous(
            futures_data,
            days_before_expiry=days_before
        )

        print(f"✅ Continuous series: {len(continuous_series):,} daily observations")

        # Count roll events
        roll_changes = (continuous_series['contract_name'] !=
                       continuous_series['contract_name'].shift(1)).sum() - 1
        print(f"🔄 Roll events: {roll_changes}")

        # Show first few roll events
        print(f"\nFirst few contract transitions:")
        prev_contract = None
        count = 0
        for _, row in continuous_series.iterrows():
            if row['contract_name'] != prev_contract and count < 5:
                print(f"  {row['date'].strftime('%Y-%m-%d')}: {row['contract_name']} "
                      f"(days to expiry: {int(row['days_to_expiry'])}, price: ${row['settlement']:.2f})")
                prev_contract = row['contract_name']
                count += 1

        # Create plot for this threshold
        plot_continuous_series(continuous_series, futures_data, days_before)

    print(f"\n✅ DEMONSTRATION COMPLETE!")
    print("Created continuous futures series with different switching thresholds.")
    print("Notice how different switching rules affect roll frequency and timing.")

if __name__ == "__main__":
    demonstrate_simple_switch()
