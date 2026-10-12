#!/usr/bin/env python
# coding: utf-8

# # Big-10 Index Analysis
# Point-in-Time (PIT) Portfolio Construction and Analysis
# 
# This notebook demonstrates PIT analysis using WRDS data to construct portfolios of the N largest U.S. companies by market capitalization.
#
# Usage:
#     python Big_10_Index.py                                   # top 10, 2013-01-01..2024-12-31
#     python Big_10_Index.py --top-n 100 --start 2015-01-01 --end 2020-06-30
#
# The date range [--start, --end] is inclusive. Lists are selected at the last
# trading day of each year, from the year before --start through the last
# trading day on or before --end, so every day in the range is covered by a
# list chosen before it.

# In[1]:


# Import libraries
import argparse
import os
import sys
from pathlib import Path
import wrds
import pandas as pd
import matplotlib.pyplot as plt

# Adjusted-price builder from Week_2 (same course)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'Week_2'))
from crsp_adjusted_prices import get_crsp_adjusted_prices

parser = argparse.ArgumentParser(description='Top-N market-cap index: PIT membership, adjusted prices, portfolio analysis.')
parser.add_argument('--top-n', type=int, default=10, help='Number of companies in each list (default 10)')
parser.add_argument('--start', default='2013-01-01', help='First day of the analysis range, inclusive (YYYY-MM-DD)')
parser.add_argument('--end', default='2024-12-31', help='Last day of the analysis range, inclusive (YYYY-MM-DD)')
args = parser.parse_args()

N = args.top_n
START = pd.Timestamp(args.start).strftime('%Y-%m-%d')
END = pd.Timestamp(args.end).strftime('%Y-%m-%d')
if START > END:
    sys.exit(f'--start {START} is after --end {END}')
FIRST_SELECTION_YEAR = int(START[:4]) - 1      # year-end list that covers START
RANGE = f'{START}..{END}'

# Output stores, named by N and range so different runs do not overwrite each other
DATA_DIR = HERE / 'data'
TAG = f'top{N}_{START}_{END}'
MEMBERSHIP_H5 = DATA_DIR / f'{TAG}_membership.h5'
PRICES_H5 = DATA_DIR / f'{TAG}_adjusted_prices.h5'


# In[2]:


# Establish WRDS connection
# Note: This will prompt for your WRDS credentials.
# Set the WRDS_USERNAME environment variable to use a different account.
conn = wrds.Connection(wrds_username=os.environ.get('WRDS_USERNAME', 'cpehlivanian'))


# ## Task 1: Top N Companies by Year

# In[3]:


# Query to get the top N companies by market cap at end of each year
# This query implements PIT logic by:
# 1. Finding the last trading day of each year
# 2. Restricting to common stock (securitytype = 'EQTY', sharetype = 'NS'),
#    which excludes ETFs such as SPY and ADRs such as BABA
# 3. Summing market cap across share classes by company (permco), so that
#    GOOG/GOOGL count as one company
# 4. Ranking companies by market cap on that specific date and keeping the top N
# Each row is one share class (permno); a company with two classes has two rows.
query_PIT = f"""
WITH last_trading_days AS (
    SELECT
        EXTRACT(YEAR FROM dlycaldt) AS year,
        MAX(dlycaldt) AS last_date
    FROM
        crsp_a_stock.wrds_dsfv2_query
    WHERE
        EXTRACT(YEAR FROM dlycaldt) >= {FIRST_SELECTION_YEAR}
        AND dlycaldt <= '{END}'
    GROUP BY
        EXTRACT(YEAR FROM dlycaldt)
),
common_stock AS (
    SELECT
        a.dlycaldt, a.permno, a.permco, a.ticker, a.dlycap
    FROM
        crsp_a_stock.wrds_dsfv2_query a
    JOIN
        last_trading_days b ON a.dlycaldt = b.last_date
    WHERE
        a.dlycap IS NOT NULL
        AND a.securitytype = 'EQTY'
        AND a.sharetype = 'NS'
),
ranked_companies AS (
    SELECT
        dlycaldt, permco,
        SUM(dlycap) AS company_cap,
        ROW_NUMBER() OVER (PARTITION BY dlycaldt ORDER BY SUM(dlycap) DESC) AS rank
    FROM
        common_stock
    GROUP BY
        dlycaldt, permco
)
SELECT
    c.dlycaldt, c.permno, c.permco, c.ticker, c.dlycap, r.company_cap, r.rank
FROM
    common_stock c
JOIN
    ranked_companies r ON c.dlycaldt = r.dlycaldt AND c.permco = r.permco
WHERE
    r.rank <= {N}
ORDER BY
    c.dlycaldt, r.rank, c.dlycap DESC;
"""

# Execute the query
df_pit = conn.raw_sql(query_PIT)

# Convert date column and display results
df_pit['dlycaldt'] = pd.to_datetime(df_pit['dlycaldt'])
print(f"Top {N} Companies by Market Cap at End of Each Year:")
print(df_pit)


# In[4]:


print(df_pit.dtypes)


# In[5]:


print(df_pit.iloc[0]['dlycaldt'])


# In[7]:


print(df_pit[df_pit['dlycaldt'] == df_pit['dlycaldt'].max()])


# ## Store membership and adjusted prices (HDF5)

# In[7b]:


# Membership table: one row per (selection date, share class).
# A list is held from the day after its selection_date through the next
# selection_date, inclusive: (effective_from, effective_to). The final list,
# selected at the end of the range, has no holding window (NaT); it is kept
# because the Non-PIT portfolio below uses it.
membership = df_pit.copy()
membership['ticker'] = membership['ticker'].fillna('').astype(str)
selection_dates = sorted(membership['dlycaldt'].unique())
next_selection = dict(zip(selection_dates[:-1], selection_dates[1:]))
membership['effective_from'] = membership['dlycaldt'] + pd.Timedelta(days=1)
membership['effective_to'] = pd.to_datetime(membership['dlycaldt'].map(next_selection))
membership.loc[membership['effective_to'].isna(), 'effective_from'] = pd.NaT
membership = membership.rename(columns={'dlycaldt': 'selection_date'})
membership = membership.astype({'permno': 'int64', 'permco': 'int64', 'rank': 'int64',
                                'dlycap': 'float64', 'company_cap': 'float64'})

DATA_DIR.mkdir(exist_ok=True)
membership.to_hdf(MEMBERSHIP_H5, key='membership', mode='w', format='table',
                  data_columns=['selection_date', 'permno', 'permco'])
print(f"Saved {len(membership)} membership rows to {MEMBERSHIP_H5}")


# In[7c]:


# Adjusted daily prices for every security that was ever a member, from the
# first selection date through END, inclusive. Adjustment factors come
# from CRSP distributions (splits, stock dividends) via crsp_adjusted_prices.py;
# prc_adj is expressed in share units as of END.
member_permnos = sorted(membership['permno'].unique().tolist())
adj_prices, _ = get_crsp_adjusted_prices(
    conn=conn,
    permnos=member_permnos,
    start_date=membership['selection_date'].min().strftime('%Y-%m-%d'),
    end_date=END,
    include_distributions=False,
)
# HDF5 cannot store pandas nullable dtypes (Int64, Float64) that WRDS returns:
# permno -> int64, every other column except date -> float64 (missing -> NaN)
adj_prices = adj_prices.astype({c: 'float64' for c in adj_prices.columns if c not in ('permno', 'date')})
adj_prices['permno'] = adj_prices['permno'].astype('int64')

adj_prices.to_hdf(PRICES_H5, key='prices', mode='w', format='table',
                  data_columns=['permno', 'date'])
print(f"Saved {len(adj_prices)} price rows for {adj_prices['permno'].nunique()} "
      f"securities to {PRICES_H5}")


# ## Task 2: Turnover Analysis

# In[8]:


# Query for monthly data to calculate turnover
# Knowledge Date consideration: We use end-of-month data that would have been
# available to investors at that time
# Companies are identified by permco rather than ticker, so a ticker change
# (e.g. FB -> META in 2022) is not counted as turnover
first_selection = df_pit['dlycaldt'].min().strftime('%Y-%m-%d')
query_EOM = f"""
WITH last_trading_days AS (
    SELECT
        EXTRACT(YEAR FROM dlycaldt) AS year,
        EXTRACT(MONTH FROM dlycaldt) AS month,
        MAX(dlycaldt) AS last_date
    FROM
        crsp_a_stock.wrds_dsfv2_query
    WHERE
        dlycaldt BETWEEN '{first_selection}' AND '{END}'
    GROUP BY
        EXTRACT(YEAR FROM dlycaldt), EXTRACT(MONTH FROM dlycaldt)
),
ranked_companies AS (
    SELECT
        a.dlycaldt, a.permco,
        SUM(a.dlycap) AS company_cap,
        ROW_NUMBER() OVER (PARTITION BY a.dlycaldt ORDER BY SUM(a.dlycap) DESC) AS rank
    FROM
        crsp_a_stock.wrds_dsfv2_query a
    JOIN
        last_trading_days b ON a.dlycaldt = b.last_date
    WHERE
        a.dlycap IS NOT NULL
        AND a.securitytype = 'EQTY'
        AND a.sharetype = 'NS'
    GROUP BY
        a.dlycaldt, a.permco
)
SELECT
    dlycaldt, permco, company_cap
FROM
    ranked_companies
WHERE
    rank <= {N}
ORDER BY
    dlycaldt, company_cap DESC;
"""

# Execute query and calculate turnover
df_eom = conn.raw_sql(query_EOM)
df_eom['dlycaldt'] = pd.to_datetime(df_eom['dlycaldt'])

# Calculate turnover between consecutive months
turnover_data = []
eom_dates = sorted(df_eom['dlycaldt'].unique())

for i in range(1, len(eom_dates)):
    current_companies = set(df_eom[df_eom['dlycaldt'] == eom_dates[i]]['permco'])
    previous_companies = set(df_eom[df_eom['dlycaldt'] == eom_dates[i-1]]['permco'])

    # Count companies that changed (entered or exited)
    changes = len(current_companies.symmetric_difference(previous_companies))
    turnover = changes / N  # Normalize by total positions

    turnover_data.append({'date': eom_dates[i], 'turnover': turnover})

df_turnover = pd.DataFrame(turnover_data)


# In[9]:


# Plot turnover over time
plt.figure(figsize=(12, 6))
plt.plot(df_turnover['date'], df_turnover['turnover'], marker='o', linestyle='-', color='g')
plt.title(f'Turnover Among Top {N} Companies (End of Month Analysis)')
plt.xlabel('Date')
plt.ylabel('Turnover Rate')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"Average monthly turnover: {df_turnover['turnover'].mean():.2%}")
print(f"Maximum monthly turnover: {df_turnover['turnover'].max():.2%}")


# ## Task 3A: Non-PIT Portfolio Analysis

# In[10]:


# Get the top N companies as of the last selection date (end of the range)
last_selection = df_pit['dlycaldt'].max()
topN_last = df_pit[df_pit['dlycaldt'] == last_selection]
print(f"Top {N} companies as of {last_selection:%Y-%m-%d}:")
print(topN_last[['ticker', 'permno', 'permco', 'dlycap']])

# Store the permnos for portfolio construction
# We select by permno, not ticker: tickers change over time (META traded as FB
# until 2022) and get reused (GOOG was the class A ticker before April 2014)
nonpit_permnos = topN_last['permno'].tolist()
print(f"\nNon-PIT portfolio will hold: {topN_last['ticker'].tolist()}")


# In[11]:


# Get daily return and market cap data for these companies over the full period
# dlyret is the total return (adjusted for splits and including dividends);
# computing dlyprc/dlyprevprc - 1 is wrong on split days because dlyprevprc
# is not split-adjusted
permnos_str = ", ".join(str(p) for p in nonpit_permnos)
query_nonpit_data = f"""
SELECT dlycaldt, permno, permco, ticker, dlyret, dlyprevcap
FROM crsp_a_stock.wrds_dsfv2_query
WHERE permno IN ({permnos_str})
AND dlycaldt BETWEEN '{START}' AND '{END}'
ORDER BY dlycaldt, permno;
"""

df_nonpit = conn.raw_sql(query_nonpit_data)
df_nonpit['dlycaldt'] = pd.to_datetime(df_nonpit['dlycaldt'])
df_nonpit['daily_return'] = df_nonpit['dlyret'].astype(float)
df_nonpit = df_nonpit.dropna(subset=['daily_return', 'dlyprevcap'])

print(f"Loaded {len(df_nonpit)} daily observations for Non-PIT portfolio")


# In[12]:


# Calculate market-cap weighted portfolio returns
# Note: We weight each day's return by the PREVIOUS day's market cap (dlyprevcap);
# using the same day's cap would put more weight on stocks that went up that day
df_nonpit['total_market_cap'] = df_nonpit.groupby('dlycaldt')['dlyprevcap'].transform('sum')
df_nonpit['market_cap_weight'] = df_nonpit['dlyprevcap'] / df_nonpit['total_market_cap']
df_nonpit['weighted_return'] = df_nonpit['daily_return'] * df_nonpit['market_cap_weight']

# Aggregate to portfolio level
nonpit_portfolio = df_nonpit.groupby('dlycaldt')['weighted_return'].sum().reset_index()
nonpit_portfolio.columns = ['date', 'portfolio_return']

# Calculate cumulative returns
nonpit_portfolio['cumulative_return'] = (1 + nonpit_portfolio['portfolio_return']).cumprod() - 1

# Final return
final_nonpit_return = nonpit_portfolio['cumulative_return'].iloc[-1]
print(f"Non-PIT Market-Cap Weighted Portfolio Total Return ({RANGE}): {final_nonpit_return:.2%}")


# In[13]:


# Equal-weighted version for comparison (rebalanced daily, equal weight per company)
# A company with two share classes (GOOG/GOOGL) gets one weight, split across
# its classes by market cap
df_nonpit['cap_x_return'] = df_nonpit['dlyprevcap'] * df_nonpit['daily_return']
company_daily = df_nonpit.groupby(['dlycaldt', 'permco'])[['cap_x_return', 'dlyprevcap']].sum()
company_daily['company_return'] = company_daily['cap_x_return'] / company_daily['dlyprevcap']

nonpit_equal = company_daily.groupby('dlycaldt')['company_return'].mean().reset_index()
nonpit_equal.columns = ['date', 'portfolio_return']
nonpit_equal['cumulative_return'] = (1 + nonpit_equal['portfolio_return']).cumprod() - 1

final_nonpit_equal_return = nonpit_equal['cumulative_return'].iloc[-1]
print(f"Non-PIT Equal-Weighted Portfolio Total Return ({RANGE}): {final_nonpit_equal_return:.2%}")


# ## Task 3B: PIT Portfolio Analysis

# In[14]:


# The PIT portfolio rebalances annually to the top N companies
# We'll calculate returns year by year, using each year's top N companies
# selected at the previous year-end

# Get daily returns for every security that was ever in the PIT portfolio
pit_permnos_str = ", ".join(str(p) for p in df_pit['permno'].unique())
query_pit_data = f"""
SELECT dlycaldt, permno, dlyret
FROM crsp_a_stock.wrds_dsfv2_query
WHERE permno IN ({pit_permnos_str})
AND dlycaldt BETWEEN '{START}' AND '{END}';
"""
df_pit_daily = conn.raw_sql(query_pit_data)
df_pit_daily['dlycaldt'] = pd.to_datetime(df_pit_daily['dlycaldt'])
df_pit_daily['year'] = df_pit_daily['dlycaldt'].dt.year
df_pit_daily['dlyret'] = df_pit_daily['dlyret'].astype(float)

# Buy-and-hold return of each security over each calendar year:
# compound the daily total returns
annual_returns = (
    df_pit_daily.groupby(['permno', 'year'])['dlyret']
    .apply(lambda r: (1 + r).prod() - 1)
    .rename('annual_return')
    .reset_index()
)

pit_annual_returns = []

# Process each year in the range; each year holds the list selected at the
# previous year-end. A partial first or last year only compounds the days in range.
for year in range(int(START[:4]), int(END[:4]) + 1):
    # Get the top N companies for the previous year-end (this is our PIT selection)
    year_top10 = df_pit[df_pit['dlycaldt'].dt.year == (year-1)].copy()

    # Weight each security by its market cap at the previous year-end
    total_cap = year_top10['dlycap'].sum()
    year_top10['weight'] = year_top10['dlycap'] / total_cap

    # Return from holding these securities over the whole of 'year'
    # This represents what an investor would have earned following PIT methodology
    year_top10 = year_top10.merge(
        annual_returns[annual_returns['year'] == year], on='permno', how='left'
    )
    year_top10['weighted_return'] = year_top10['annual_return'] * year_top10['weight']

    portfolio_return = year_top10['weighted_return'].sum()
    pit_annual_returns.append(portfolio_return)

    print(f"Year {year}: Portfolio Return = {portfolio_return:.2%}")
    print(f"  Companies: {year_top10['ticker'].tolist()}")

# Calculate total cumulative return
pit_cumulative = (pd.Series([1+x for x in pit_annual_returns]).cumprod() - 1.).iloc[-1]
print(f"\nPIT Portfolio Total Return ({RANGE}): {pit_cumulative:.2%}")


# ## Task 4: Performance Analysis

# In[16]:


# Create a summary comparison
print("=" * 60)
print(f"PORTFOLIO PERFORMANCE COMPARISON (top {N}, {RANGE})")
print("=" * 60)
print(f"Non-PIT Market-Cap Weighted:  {final_nonpit_return:>8.2%}")
print(f"Non-PIT Equal-Weighted:       {final_nonpit_equal_return:>8.2%}")
print(f"PIT Market-Cap Weighted:      {pit_cumulative:>8.2%}")
print("=" * 60)

print("\nKey Insights:")
print("• Non-PIT portfolios benefit from SURVIVORSHIP BIAS")
print(f"• They select companies known to be successful as of {last_selection:%Y-%m-%d}")
print("• PIT portfolio reflects REAL-TIME investment decisions")
print("• PIT rebalancing may capture companies at their peak")
print("• Transaction costs and timing issues affect PIT performance")


# In[ ]:


# Close connection
conn.close()


