"""Plain-Python duplicate of price_adjustment_example.ipynb.

Cells are marked with `# %%` so this still runs cell-by-cell in
Jupytext, VS Code or Spyder.
"""

# %%
import wrds
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import datetime

# %%
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

# %%
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

# %%
# Distribution types that pay actual CASH.  A spinoff pays a security, not
# cash, so it does not belong in a cash dividend amount.
CASH_DISTYPES = ('CD', 'SD', 'CG', 'ROC')


def calculate_adjustment_factors(distributions: pd.DataFrame) -> pd.DataFrame:
    """
    Per-event price adjustment factors from CRSP distributions.

    One row per (permno, ex_date).  Every column depends only on that event,
    so nothing here is ever rewritten by a later corporate action.

    Cumulative factors are deliberately NOT returned.  A cumulative factor is
    compounded backward from whatever the last event happens to be, so every
    historical value changes the next time a split lands - it is a rendering
    against an anchor, not something to store.  Compound at the point of use.

    Args:
        distributions: DataFrame from get_corporate_actions()

    Returns:
        permno, ex_date, and

            divamt            cash per share, summed over same-day events
            price_adj_factor  1 / (1 + FACPR)
    """
    cols = ['permno', 'ex_date', 'divamt', 'price_adj_factor']

    if distributions.empty:
        return pd.DataFrame(columns=cols)

    dist = distributions.copy()
    dist['ex_date'] = pd.to_datetime(dist['ex_date']).dt.normalize()

    # CRSP states a distribution as a PROPORTIONAL change in share count:
    # FACPR = (s(t) - s(t')) / s(t').  A 2-for-1 split is 1.0, a 1-for-2
    # reverse split is -0.5, a cash dividend is 0.  So a price quoted before
    # the ex-date is divided by (1 + FACPR), never by FACPR, and cash
    # dividends drop out of that formula on their own: 1 + 0 = 1.
    dist['_mult'] = 1.0 + pd.to_numeric(dist['facpr'], errors='coerce').fillna(0.0)
    is_cash = dist['distype'].astype(str).str.strip().str.upper().isin(CASH_DISTYPES)
    dist['_div'] = (pd.to_numeric(dist['divamt'], errors='coerce')
                      .fillna(0.0).where(is_cash, 0.0))

    # Several distributions can share one ex-date (DISSEQNBR orders them).
    # Cash amounts ADD, split factors COMPOUND.
    ev = (dist.groupby(['permno', 'ex_date'], as_index=False)
              .agg(divamt=('_div', 'sum'), mult=('_mult', 'prod')))

    # mult <= 0 means the share line went to zero - a liquidation, not a price
    # adjustment.  Leave the price alone and handle it as a terminal cash flow.
    ev['price_adj_factor'] = np.where(ev['mult'] > 0, 1.0 / ev['mult'], 1.0)

    return ev.sort_values(['permno', 'ex_date']).reset_index(drop=True)[cols]

# %%
conn = wrds.Connection(wrds_username='cpehlivanian')

# %%
start_date = '2002-01-01'
end_date = '2023-12-31'
permnos = [10107]

prices = get_unadjusted_prices(conn, permnos, start_date, end_date)
distributions = get_corporate_actions(conn, permnos, start_date, end_date)
adjustments = calculate_adjustment_factors(distributions)

# %%
print(prices.head(10))

# %%
print(distributions.head(10))

# %%
print(adjustments.head(10))
