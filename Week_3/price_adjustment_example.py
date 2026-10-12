"""Plain-Python duplicate of price_adjustment_example.ipynb.

Cells are marked with `# %%` so the function definitions still run
cell-by-cell in Jupytext, VS Code or Spyder.  The demo at the bottom sits
under `if __name__ == "__main__":` so other scripts can import the CRSP
functions from here without opening a WRDS connection.
"""

# %%
import wrds
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import datetime

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

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
def build_adjustment_table(adjustments: pd.DataFrame,
                           prices: pd.DataFrame) -> pd.DataFrame:
    """
    Combined dividend + split multiplicative adjustment table.

    A split is already multiplicative (price_adj_factor), but a dividend is an
    additive cash amount.  Dividing it by the CUM-DIVIDEND close - the last
    close strictly BEFORE the ex-date - turns it into a multiplicative factor
    too:

        split_factor = 1 / (1 + FACPR)          (= price_adj_factor)
        div_factor   = 1 - D / P(t-1)
        total_factor = split_factor * div_factor

    D / P(t-1) is a ratio of two unadjusted prices on the same share basis, so
    it is scale free and the two factors simply multiply.

    Check the arithmetic on a $1 dividend on a $100 stock that opens ex at $99.
    div_factor = 1 - 1/100 = 0.99, and 99 / (100 * 0.99) - 1 = 0, a zero total
    return across the ex-date, which is right.

    The cum_ columns compound the per-event factors BACKWARD from the last
    event, so they are anchored to the end of the sample (1.0 after the last
    ex-date) and every value changes when a later event arrives.  They are a
    rendering against that anchor - rebuild them, don't store them.

    Args:
        adjustments: DataFrame from calculate_adjustment_factors()
        prices: DataFrame from get_unadjusted_prices().  Needed for P(t-1), so
            it must cover the trading day before each ex-date.

    Returns:
        One row per (permno, ex_date):

            divamt            cash per share
            prev_close        the P(t-1) above
            split_factor      this event, splits only
            div_factor        this event, dividends only (NaN if unusable)
            total_factor      split_factor * div_factor
            cum_split_factor  product of split_factor over this and all later
                              events -> CRSP CFACPR, re-anchored
            cum_total_factor  same for total_factor

        Both cum_ columns apply to every date STRICTLY BEFORE that row's
        ex_date.
    """
    cols = ['permno', 'ex_date', 'divamt', 'prev_close',
            'split_factor', 'div_factor', 'total_factor',
            'cum_split_factor', 'cum_total_factor']

    if adjustments.empty:
        return pd.DataFrame(columns=cols)

    ev = adjustments.copy()
    ev['ex_date'] = pd.to_datetime(ev['ex_date']).dt.normalize()

    # prev_close: the last close STRICTLY before the ex-date.  The ex-date's
    # own close is already ex-dividend, hence allow_exact_matches=False.
    # CRSP stores a NEGATIVE price when the close is a bid/ask average.
    px = prices[['permno', 'date', 'prc']].copy()
    px['date'] = pd.to_datetime(px['date']).dt.normalize()
    px['prc'] = pd.to_numeric(px['prc'], errors='coerce').abs()
    ev = pd.merge_asof(
        ev.sort_values('ex_date'),
        (px.dropna(subset=['prc']).sort_values('date')
           .rename(columns={'date': 'ex_date', 'prc': 'prev_close'})),
        on='ex_date', by='permno',
        direction='backward', allow_exact_matches=False)

    ev['split_factor'] = ev['price_adj_factor'].astype(float)
    div = pd.to_numeric(ev['divamt'], errors='coerce').fillna(0.0)
    # A pure split pays no cash, so pin it at 1.0 rather than computing
    # 1 - 0/prev_close, which would be NaN whenever prev_close is missing.
    div_f = (1.0 - div / ev['prev_close']).where(div != 0.0, 1.0)
    # Non-positive or NaN means either no prior close in the sample, or a
    # distribution worth the whole security (a liquidation).  Neither is a
    # usable price factor: left NaN here, treated as 1.0 when compounding.
    ev['div_factor'] = div_f.where(div_f > 0)
    ev['total_factor'] = ev['split_factor'] * ev['div_factor'].fillna(1.0)

    # Compound backward from the end of the sample.
    ev = ev.sort_values(['permno', 'ex_date'], ascending=[True, False])
    grp = ev.groupby('permno', sort=False)
    ev['cum_split_factor'] = grp['split_factor'].cumprod()
    ev['cum_total_factor'] = grp['total_factor'].cumprod()

    return ev.sort_values(['permno', 'ex_date']).reset_index(drop=True)[cols]

# %%
def adjust_prices(prices: pd.DataFrame,
                  adjustment_table: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-adjusted daily prices from build_adjustment_table().

    The factor on date t is the cum_ value of the FIRST event with
    ex_date > t.  Dates on or after the last ex-date get 1.0, so the latest
    prices are the traded ones and earlier prices are restated.

    Args:
        prices: DataFrame from get_unadjusted_prices()
        adjustment_table: DataFrame from build_adjustment_table()

    Returns:
        prices plus

            cum_split_factor, cum_total_factor
            split_adj_prc   |prc| * cum_split_factor: real price levels,
                            dividends still visible as drops on the ex-date
            total_adj_prc   |prc| * cum_total_factor: total-return series.
                            Its daily returns match CRSP ret exactly off
                            ex-dates; on an ex-date they differ by
                            D / (P(t-1) - D) * ret, second order in yield
    """
    out = prices.copy()
    out['date'] = pd.to_datetime(out['date']).dt.normalize()
    fac_cols = ['cum_split_factor', 'cum_total_factor']

    if adjustment_table.empty:
        out[fac_cols] = 1.0
    else:
        f = adjustment_table[['permno', 'ex_date'] + fac_cols].copy()
        f['ex_date'] = pd.to_datetime(f['ex_date']).dt.normalize()
        f['permno'] = f['permno'].astype(out['permno'].dtype)
        out = pd.merge_asof(
            out.sort_values('date'), f.sort_values('ex_date'),
            left_on='date', right_on='ex_date', by='permno',
            direction='forward', allow_exact_matches=False)
        out[fac_cols] = out[fac_cols].astype(float).fillna(1.0)
        out = out.drop(columns='ex_date')

    # Adjust |prc|, but leave prc itself alone: its sign is CRSP's flag for a
    # bid/ask-average close.
    abs_prc = pd.to_numeric(out['prc'], errors='coerce').abs()
    out['split_adj_prc'] = abs_prc * out['cum_split_factor']
    out['total_adj_prc'] = abs_prc * out['cum_total_factor']

    return out.sort_values(['permno', 'date']).reset_index(drop=True)

# %%
# Demo.  Guarded so that `import price_adjustment_example` only defines the
# functions above (corporate_actions_examples.py imports them).
if __name__ == "__main__":
    conn = wrds.Connection(wrds_username='cpehlivanian')

    start_date = '2002-01-01'
    end_date = '2023-12-31'
    permnos = [10107]

    prices = get_unadjusted_prices(conn, permnos, start_date, end_date)
    distributions = get_corporate_actions(conn, permnos, start_date, end_date)
    adjustments = calculate_adjustment_factors(distributions)
    adjustment_table = build_adjustment_table(adjustments, prices)
    adjusted_prices = adjust_prices(prices, adjustment_table)

    print(prices.head(10))

    print(distributions.head(10))

    print(adjustments.head(10))

    print(adjustment_table.head(10))

    # Across the 2003-02-18 2-for-1 split and the 2004-11-15 $3 special dividend.
    show = ['permno', 'date', 'prc', 'ret', 'cum_split_factor', 'cum_total_factor',
            'split_adj_prc', 'total_adj_prc']
    print(adjusted_prices.loc[adjusted_prices['date'].between('2003-02-12', '2003-02-21'), show])
    print(adjusted_prices.loc[adjusted_prices['date'].between('2004-11-09', '2004-11-17'), show])

    # Checks.  Split-adjusted prices must match CRSP's own cumulative factor,
    # re-anchored to the end of the sample.  Total-return prices must reproduce
    # CRSP's daily total return - exactly off ex-dates, and to second order on
    # them (CRSP adds D to the ex-day price, the factor subtracts it from P(t-1)).
    chk = adjusted_prices.copy()
    cfacpr = pd.to_numeric(chk['cfacpr'], errors='coerce')
    chk['crsp_split_adj'] = (pd.to_numeric(chk['prc'], errors='coerce').abs() / cfacpr
                             * cfacpr.groupby(chk['permno']).transform('last'))
    chk['tr_ret'] = chk.groupby('permno')['total_adj_prc'].pct_change()
    ret_err = (chk['tr_ret'] - pd.to_numeric(chk['ret'], errors='coerce')).abs()
    on_ex = chk.set_index(['permno', 'date']).index.isin(
        adjustment_table.set_index(['permno', 'ex_date']).index)
    print('max |split_adj_prc - CRSP|      :',
          (chk['split_adj_prc'] - chk['crsp_split_adj']).abs().max())
    print('max |total return - CRSP ret|, off ex-dates:', ret_err[~on_ex].max())
    print('max |total return - CRSP ret|, on ex-dates :', ret_err[on_ex].max())

    conn.close()
