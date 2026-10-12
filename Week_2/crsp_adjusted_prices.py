import wrds
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import datetime

def get_unadjusted_prices(conn: wrds.Connection, 
                         permnos: Optional[List[int]] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch unadjusted daily stock prices from CRSP.
    
    Args:
        conn: WRDS database connection
        permnos: List of PERMNOs to fetch. If None, fetches all.
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with columns: permno, dlycaldt, dlyprc, dlyvol, dlyret, dlybid, dlyask, shrout, dlycumfacpr, dlycumfacshr
    """
    query = """
    SELECT permno, dlycaldt as date, dlyprc as prc, dlyvol as vol, dlyret as ret, 
           dlybid as bid, dlyask as ask, shrout, dlycumfacpr as cfacpr, dlycumfacshr as cfacshr
    FROM crsp.dsf_v2
    WHERE 1=1
    """
    
    conditions = []
    if permnos is not None:
        permno_list = ','.join(map(str, permnos))
        conditions.append(f"permno IN ({permno_list})")
    
    if start_date is not None:
        conditions.append(f"dlycaldt >= '{start_date}'")
        
    if end_date is not None:
        conditions.append(f"dlycaldt <= '{end_date}'")
    
    if conditions:
        query += " AND " + " AND ".join(conditions)
    
    query += " ORDER BY permno, dlycaldt"
    
    return conn.raw_sql(query, date_cols=['date'])

def get_corporate_actions(conn: wrds.Connection,
                         permnos: Optional[List[int]] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch corporate action data from the CRSP distributions table.

    Uses crsp.stkdistributions, the CIZ-format event table that pairs with
    crsp.dsf_v2.  The legacy SIZ table (crsp.dsedist) carries the same events
    under different column names and is no longer the maintained one.

    The single SIZ DISTCD is replaced by a family of narrower codes.  The two
    that matter for classification are DISTYPE and DISDETAILTYPE:

        CD  / CDIV    cash dividend (FACPR = 0)
        SD  / SDIV    special cash dividend (FACPR = 0)
        CG  / CAPG    capital gains distribution
        ROC / ROC     return of capital
        FRS / STKSPL  stock split, forward or reverse (FACPR != 0)
        FRS / STKDIV  stock dividend (FACPR != 0)
        SP  / SEC*    distribution of another security, e.g. a spinoff
        CP  / *       merger, liquidation or other terminal cash payment;
                      FACPR is ~-1.0 because the share line goes to zero

    DISORDINARYFLG ('Y'/'N') separates ordinary distributions from the rest,
    and DISTAXTYPE carries the tax character.  See the CRSP manual for the
    full code lists - the ones above cover the overwhelming majority of rows.

    Args:
        conn: WRDS database connection
        permnos: List of PERMNOs to fetch. If None, fetches all.
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format

    Returns:
        DataFrame with distribution events including dividends and splits.
        Columns are aliased to the SIZ names (ex_date, divamt, facpr, facshr,
        acperm, accomp) so the rest of this module is unchanged.
    """
    query = """
    SELECT permno, disexdt as ex_date, disrecorddt as record_date,
           dispaydt as pay_date, disdeclaredt as declare_date, disseqnbr as seq_nbr,
           distype, disdetailtype, disfreqtype, dispaymenttype, distaxtype,
           disordinaryflg, disorigcurtype,
           disdivamt as divamt, disfacpr as facpr, disfacshr as facshr,
           dispermno as acperm, dispermco as accomp
    FROM crsp.stkdistributions
    WHERE 1=1
    """

    conditions = []
    if permnos is not None:
        permno_list = ','.join(map(str, permnos))
        conditions.append(f"permno IN ({permno_list})")

    if start_date is not None:
        conditions.append(f"disexdt >= '{start_date}'")

    if end_date is not None:
        conditions.append(f"disexdt <= '{end_date}'")

    if conditions:
        query += " AND " + " AND ".join(conditions)

    # DISSEQNBR orders multiple distributions sharing one ex-date.
    query += " ORDER BY permno, disexdt, disseqnbr"

    dist = conn.raw_sql(query, date_cols=['ex_date', 'record_date', 'pay_date',
                                          'declare_date'])

    # DISDIVAMT is NULL on pure share-count events (SIZ wrote 0.0 there), and
    # FACPR/FACSHR are NULL on pure cash events.  Left as NaN on purpose:
    # calculate_adjustment_factors fills the factors, and a NaN dividend is a
    # more honest input to a yield calculation than a fabricated zero.
    return dist

def calculate_adjustment_factors(distributions: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cumulative adjustment factors for prices from corporate actions.
    
    Args:
        distributions: DataFrame from get_corporate_actions()
        
    Returns:
        DataFrame with cumulative adjustment factors by permno and date
    """
    if distributions.empty:
        return pd.DataFrame(columns=['permno', 'ex_date', 'price_adj_factor', 'shares_adj_factor'])
    
    # Sort by permno and ex_date
    dist_sorted = distributions.sort_values(['permno', 'ex_date']).copy()
    
    # Initialize adjustment factors
    dist_sorted['price_adj_factor'] = 1.0
    dist_sorted['shares_adj_factor'] = 1.0
    
    # CRSP states a distribution as a PROPORTIONAL change in share count:
    # FACPR = (s(t) - s(t')) / s(t').  A 4-for-1 split is FACPR = 3.0, a 2-for-1
    # is 1.0, a 1-for-2 reverse split is -0.5, and a cash dividend is 0.  Prices
    # quoted before the ex-date are therefore divided by (1 + FACPR), never by
    # FACPR.  Cash dividends drop out of the formula on their own: 1 + 0 = 1.
    facpr = pd.to_numeric(dist_sorted['facpr'], errors='coerce').fillna(0.0)
    facshr = pd.to_numeric(dist_sorted['facshr'], errors='coerce').fillna(0.0)
    price_mult = 1.0 + facpr
    shares_mult = 1.0 + facshr
    dist_sorted['price_adj_factor'] = np.where(price_mult > 0, 1.0 / price_mult, 1.0)
    dist_sorted['shares_adj_factor'] = np.where(shares_mult > 0, shares_mult, 1.0)
    
    # Calculate cumulative factors for each permno
    result_list = []
    
    for permno in dist_sorted['permno'].unique():
        permno_data = dist_sorted[dist_sorted['permno'] == permno].copy()
        
        # Calculate cumulative factors (going backward in time)
        permno_data = permno_data.sort_values('ex_date', ascending=False)
        permno_data['cum_price_adj'] = permno_data['price_adj_factor'].cumprod()
        permno_data['cum_shares_adj'] = permno_data['shares_adj_factor'].cumprod()
        
        result_list.append(permno_data[['permno', 'ex_date', 'cum_price_adj', 'cum_shares_adj']])
    
    if result_list:
        return pd.concat(result_list, ignore_index=True)
    else:
        return pd.DataFrame(columns=['permno', 'ex_date', 'cum_price_adj', 'cum_shares_adj'])

def apply_adjustments(prices: pd.DataFrame, 
                     adjustment_factors: pd.DataFrame) -> pd.DataFrame:
    """
    Apply adjustment factors to unadjusted prices.
    
    Args:
        prices: DataFrame from get_unadjusted_prices()
        adjustment_factors: DataFrame from calculate_adjustment_factors()
        
    Returns:
        DataFrame with adjusted prices
    """
    if prices.empty:
        return prices.copy()
        
    result = prices.copy()
    
    # Convert price to positive (CRSP stores negative prices for certain conditions)
    result['prc_unadj'] = result['prc']
    result['prc'] = abs(result['prc'])
    
    # Adjusted columns are plain float64: WRDS may return shrout as a nullable
    # Int64, which cannot hold the fractional values a split factor produces.
    result['prc_adj'] = result['prc'].astype('float64')
    result['shrout_adj'] = result['shrout'].astype('float64')

    if adjustment_factors.empty:
        # No adjustments needed
        return result
    
    for permno in result['permno'].unique():
        permno_prices = result['permno'] == permno
        permno_adj = adjustment_factors[adjustment_factors['permno'] == permno]
        
        if permno_adj.empty:
            continue
            
        # Sort adjustment factors by ex_date
        permno_adj = permno_adj.sort_values('ex_date')
        
        # cum_price_adj already compounds every event from the end of the sample
        # back to its own ex-date, so it is ASSIGNED to the dates it covers -
        # never multiplied in event by event.  Walking the events from the
        # latest backwards lets each one overwrite its own narrower band.
        price_mult = pd.Series(1.0, index=result.index)
        shares_mult = pd.Series(1.0, index=result.index)

        for _, adj_row in permno_adj.sort_values('ex_date', ascending=False).iterrows():
            date_mask = permno_prices & (result['date'] < adj_row['ex_date'])
            price_mult[date_mask] = adj_row['cum_price_adj']
            if pd.notna(adj_row['cum_shares_adj']):
                shares_mult[date_mask] = adj_row['cum_shares_adj']

        result.loc[permno_prices, 'prc_adj'] = (
            result.loc[permno_prices, 'prc'] * price_mult[permno_prices])
        result.loc[permno_prices, 'shrout_adj'] = (
            result.loc[permno_prices, 'shrout'] * shares_mult[permno_prices])
    
    return result

def get_crsp_adjusted_prices(conn: wrds.Connection,
                           permnos: Optional[List[int]] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           include_distributions: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to get CRSP adjusted stock prices.
    
    Args:
        conn: WRDS database connection
        permnos: List of PERMNOs to fetch. If None, fetches all.
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        include_distributions: Whether to include distribution data in output
        
    Returns:
        Tuple of (adjusted_prices_df, distributions_df)
    """
    # Extend date range for distributions to capture relevant corporate actions
    extended_start = None
    if start_date:
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        extended_start = (start_dt.replace(year=start_dt.year-2)).strftime('%Y-%m-%d')
    
    # Fetch data
    print("Fetching unadjusted prices...")
    prices = get_unadjusted_prices(conn, permnos, start_date, end_date)
    
    print("Fetching corporate actions...")
    distributions = get_corporate_actions(conn, permnos, extended_start, end_date)
    
    print("Calculating adjustment factors...")
    adjustment_factors = calculate_adjustment_factors(distributions)
    
    print("Applying adjustments...")
    adjusted_prices = apply_adjustments(prices, adjustment_factors)
    
    print(f"Retrieved {len(adjusted_prices)} price observations for {adjusted_prices['permno'].nunique()} securities")
    print(f"Found {len(distributions)} corporate actions")
    
    if include_distributions:
        return adjusted_prices, distributions
    else:
        return adjusted_prices, pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Initialize connection
    conn = wrds.Connection(wrds_username='cpehlivanian')
    
    # Example: Get adjusted prices for specific stocks
    sample_permnos = [10107, 10113, 10145]  # Sample PERMNOs

    adjusted_prices, distributions = get_crsp_adjusted_prices(
        conn=conn,
        permnos=sample_permnos,
        start_date='2002-01-01',
        end_date='2023-12-31'
    )
    
    print("\nSample adjusted prices:")
    print(adjusted_prices.head(10))
    
    print("\nSample distributions:")
    print(distributions.head(10))
    
    conn.close()

    breakpoint()
