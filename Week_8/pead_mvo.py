"""PEAD signal with a two-stage covariance estimate and walk-forward MVO.

The signal is the IBES-based SUE (sue3) of ../Week_6/pead.py: build_signal_data() below
is its Steps 0-7, unchanged apart from being wrapped in a function. As in its Step 9, an
announcement is held from the close of event day PF_FIRST - 1 through event day PF_LAST,
or until the next announcement.

Each day t the active names are those with a held announcement. Two dollar-neutral
portfolios are formed at the close of t and earn the returns of t + 1:

  Stratified  sue3 quintiles across the active names on day t; long the top quintile and
              short the bottom one, equal-weighted, $1 per leg.
  MVO         max w'mu - lambda w'Sigma w subject to 1'w = 0, in closed form
                  w proportional to Sigma^-1 mu - (b/a) Sigma^-1 1
              with a = 1'Sigma^-1 1 and b = 1'Sigma^-1 mu. The alpha is
              mu_i = sigma_i z_i, with z_i the normal score of the rank of sue3 across
              the active names and sigma_i the model volatility. lambda is set each day
              so that the predicted volatility is TARGET_VOL.

Sigma is the two-stage PCA estimate B diag(F) B' + D from the trailing COV_WINDOW days
of returns of all names in the sample (n x T, rows = assets), re-estimated every
COV_EVERY days with data up to t only. Sigma^-1 is applied with the Woodbury identity.

Costs are COST_BPS per dollar traded, as in pead.py. Since both portfolios are dollar
neutral, raw and market-excess returns give the same P&L.

The WRDS pull is cached in data/pead_signal.pkl; run with --refresh to rebuild it.
"""

import argparse
import datetime
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
WEEK6 = os.path.join(HERE, '..', 'Week_6')
CACHE = os.path.join(HERE, 'data', 'pead_signal.pkl')
MCAP_CACHE = os.path.join(HERE, 'data', 'pead_mcap.pkl')
PLOT = os.path.join(HERE, 'plot', 'pead_mvo_equity.png')

PF_FIRST, PF_LAST = 2, 50       # event days held, as in pead.py Step 9
MIN_NAMES = 50                  # minimum active names to trade on a day
N_UNIVERSE = None               # if set, keep the N largest names (market cap at each re-estimation)
N_FACTORS = 10                  # k in the two-stage estimate
COV_WINDOW = 252                # trailing days for the covariance estimate
COV_EVERY = 1                   # re-estimate the covariance every day
MIN_OBS = 0.9                   # fraction of the window a name needs to be in the model
TARGET_VOL = 0.10               # annualized predicted volatility of the MVO portfolio
COST_BPS = 5


def build_signal_data():
    """Steps 0-7 of ../Week_6/pead.py. Returns the event-day panel and daily returns."""
    import wrds

    ###################
    # Connect to WRDS #
    ###################
    conn = wrds.Connection(wrds_username='cpehlivanian')

    # set sample date range (reduced for memory efficiency)
    begdate = '01/01/2018'
    enddate = '12/31/2022'

    # set CRSP date range a bit wider to guarantee collecting all information
    crsp_begdate = '01/01/2017'
    crsp_enddate = '12/31/2023'

    #################################
    # Step 0: Read in ICLINK output #
    #################################

    # iclink.pkl is the output from the python program iclink
    # it contains the linking between crsp and ibes
    with open('iclink.pkl', 'rb') as f:
        iclink = pkl.load(f)

    ##################################
    # Step 1. S&P 500 Index Universe #
    ##################################

    # All companies that were ever included in S&P 500 index as an example
    # Old version of the code uses comp.idxcst_his

    # New code uses crsp.msp500list
    # Linking Compustat GVKEY and IBES Tickers using ICLINK
    # For unmatched GVKEYs, use header IBTIC link in Compustat Security file

    sp500 = conn.raw_sql(f"""
                            select a.*
                            from crsp.msp500list as a;
                            """, date_cols=['start', 'ending'])

    # CCM data
    _ccm = conn.raw_sql(""" select gvkey, lpermco as permco, lpermno as permno,
                            linkdt, linkenddt
                            from crsp.ccmxpf_linktable
                            where usedflag=1
                            and linkprim in ('P', 'C')""", date_cols=['linkdt', 'linkenddt'])

    _ccm[['permco', 'permno']] = _ccm[['permco', 'permno']].astype(int)

    # Fill linkenddt missing value (.E in SAS dataset) with today's date
    _ccm['linkenddt'] = _ccm.linkenddt.fillna(datetime.date.today())

    _sec = conn.raw_sql(""" select ibtic, gvkey from comp.security """)

    # Start the sequence of left join
    gvkey = pd.merge(sp500, _ccm, how='left', on=['permno'])
    gvkey = pd.merge(gvkey, _sec.loc[_sec.ibtic.notna()], how='left', on=['gvkey'])

    # high quality links from iclink
    # score = 0 or 1
    iclink_hq = iclink.loc[(iclink.score <= 1)]

    gvkey = pd.merge(gvkey, iclink_hq, how='left', on=['permno'])

    # fill missing ticker with ibtic
    gvkey.ticker = np.where(gvkey.ticker.notnull(), gvkey.ticker, gvkey.ibtic)

    # Keep relevant columns and drop duplicates if there is any
    gvkey = gvkey[['gvkey', 'permco', 'permno', 'linkdt', 'linkenddt', 'ticker']]

    gvkey = gvkey.drop_duplicates()

    # date ranges from gvkey
    # min linkdt for ticker and permno combination
    gvkey_mindt = gvkey.groupby(['ticker', 'permno']).linkdt.min().reset_index()

    # max linkenddt for ticker and permno combination
    gvkey['linkenddt'] = gvkey['linkenddt'].apply(lambda r: pd.Timestamp(r))
    gvkey_maxdt = gvkey.groupby(['ticker', 'permno']).linkenddt.max().reset_index()

    # link date range
    gvkey_dt = pd.merge(gvkey_mindt, gvkey_maxdt, how='inner', on=['ticker', 'permno'])

    #######################################
    # Step 2. Extract Estimates from IBES #
    #######################################

    # Extract estimates from IBES Unadjusted file and select
    # the latest estimate for a firm within broker-analyst group
    # "fpi in (6,7)" selects quarterly forecast for the current
    # and the next fiscal quarter

    ibes_temp = conn.raw_sql(f"""
                            select ticker, estimator, analys, pdf, fpi, value,
                            fpedats, revdats, revtims, anndats, anntims
                            from ibes.detu_epsus
                            where fpedats between '{begdate}' and '{enddate}'
                            and (fpi='6' or fpi='7')
                            """, date_cols=['revdats', 'anndats', 'fpedats'])

    # merge to get date range linkdt and linkenddt to fulfill date requirement
    ibes_temp = pd.merge(ibes_temp, gvkey_dt, how='left', on=['ticker'])
    ibes_temp = ibes_temp.loc[(ibes_temp.linkdt <= ibes_temp.anndats) & (ibes_temp.anndats <= ibes_temp.linkenddt)]

    # Count number of estimates reported on primary/diluted basis

    p_sub = ibes_temp[['ticker', 'fpedats', 'pdf']].loc[ibes_temp.pdf == 'P']
    d_sub = ibes_temp[['ticker', 'fpedats', 'pdf']].loc[ibes_temp.pdf == 'D']

    p_count = p_sub.groupby(['ticker', 'fpedats']).pdf.count().reset_index().rename(columns={'pdf': 'p_count'})
    d_count = d_sub.groupby(['ticker', 'fpedats']).pdf.count().reset_index().rename(columns={'pdf': 'd_count'})

    ibes = pd.merge(ibes_temp, d_count, how='left', on=['ticker', 'fpedats'])
    ibes = pd.merge(ibes, p_count, how='left', on=['ticker', 'fpedats'])
    ibes['d_count'] = ibes.d_count.fillna(0)
    ibes['p_count'] = ibes.p_count.fillna(0)

    # Determine whether most analysts report estimates on primary/diluted basis
    # following Livnat and Mendenhall (2006)

    ibes['basis'] = np.where(ibes.p_count > ibes.d_count, 'P', 'D')

    ibes = ibes.sort_values(by=['ticker', 'fpedats', 'estimator', 'analys', 'anndats', 'anntims', 'revdats', 'revtims'])\
        .drop(['linkdt', 'linkenddt', 'p_count', 'd_count', 'pdf', 'fpi'], axis=1)

    # Keep the latest observation for a given analyst
    # Group by company fpedats estimator analys then pick the last record in the group

    ibes_1 = ibes.groupby(['ticker', 'fpedats', 'estimator', 'analys']).apply(lambda x: x.index[-1], include_groups=False).to_frame().reset_index()

    # reset index to the old dataframe index for join in the next step
    ibes_1 = ibes_1.set_index(0)

    # Inner join with the last analyst record per group
    ibes = pd.merge(ibes, ibes_1[['analys']], left_index=True, right_index=True)

    # drop duplicate column
    ibes = ibes.drop(['analys_y'], axis=1).rename(columns={'analys_x': 'analys'})

    #######################################
    # Step 3. Link Estimates with Actuals #
    #######################################

    # Link Unadjusted estimates with Unadjusted actuals and CRSP permnos
    # Keep only the estimates issued within 90 days before the report date

    # Getting actual piece of data
    ibes_act = conn.raw_sql(f"""
                            select ticker, anndats as repdats, value as act, pends as fpedats, pdicity
                            from ibes.actu_epsus
                            where pends between '{begdate}' and '{enddate}'
                            and pdicity='QTR'
                            """, date_cols=['repdats', 'fpedats'])

    # Join with the estimate piece of the data

    ibes1 = pd.merge(ibes, ibes_act, how='left', on=['ticker', 'fpedats'])
    ibes1['dgap'] = ibes1.repdats - ibes1.anndats

    ibes1['flag'] = np.where((ibes1.dgap >= datetime.timedelta(days=0)) & (ibes1.dgap <= datetime.timedelta(days=90)) & (ibes1.repdats.notna()) & (ibes1.anndats.notna()), 1, 0)

    ibes1 = ibes1.loc[ibes1.flag == 1].drop(['flag', 'dgap', 'pdicity'], axis=1)

    # Select all relevant combinations of Permnos and Date

    ibes1_dt1 = ibes1[['permno', 'anndats']].drop_duplicates()

    ibes1_dt2 = ibes1[['permno', 'repdats']].drop_duplicates().rename(columns={'repdats': 'anndats'})

    ibes_anndats = pd.concat([ibes1_dt1, ibes1_dt2]).drop_duplicates()

    # Adjust all estimate and earnings announcement dates to the closest
    # preceding trading date in CRSP to ensure that adjustment factors won't
    # be missing after the merge

    # unique anndats from ibes
    uniq_anndats = ibes_anndats[['anndats']].drop_duplicates()

    # unique trade dates from crsp.dsi
    crsp_dats = conn.raw_sql("""
                                select date
                                from crsp.dsi
                             """, date_cols=['date'])

    # Create up to 5 days prior dates relative to anndats

    for i in range(0, 5):
        uniq_anndats[i] = uniq_anndats.anndats - datetime.timedelta(days=i)

    # reshape (transpose) the df for later join with crsp trading dates

    expand_anndats = uniq_anndats.set_index('anndats').stack().reset_index().\
        rename(columns={'level_1': 'prior', 0: 'prior_date'})

    # merge with crsp trading dates
    tradedates = pd.merge(expand_anndats, crsp_dats, how='left', left_on=['prior_date'], right_on=['date'])

    # create the dgap (days gap) variable for min selection
    tradedates['dgap'] = tradedates.anndats - tradedates.date

    # choosing the row with the smallest dgap for a given anndats
    tradedates = tradedates.loc[tradedates.groupby('anndats')['dgap'].idxmin()]

    tradedates = tradedates[['anndats', 'date']]

    # merge the CRSP adjustment factors for all estimate and report dates

    # extract CRSP adjustment factors
    cfacshr = conn.raw_sql(f"""
                            select permno, date, cfacshr
                            from crsp.dsf
                            where date between '{crsp_begdate}' and '{crsp_enddate}'
                            """, date_cols=['date'])

    ibes_anndats = pd.merge(ibes_anndats, tradedates, how='left', on=['anndats'])

    ibes_anndats = pd.merge(ibes_anndats, cfacshr, how='left', on=['permno', 'date'])

    #########################################
    # Step 4. Adjust Estimates with CFACSHR #
    #########################################

    # Put the estimate on the same per share basis as
    # company reported EPS using CRSP Adjustment factors.
    # New_value is the estimate adjusted to be on the
    # same basis with reported earnings.

    ibes1 = pd.merge(ibes1, ibes_anndats, how='inner', on=['permno', 'anndats'])
    ibes1 = ibes1.drop(['anndats', 'date'], axis=1).rename(columns={'cfacshr': 'cfacshr_ann'})

    ibes1 = pd.merge(ibes1, ibes_anndats, how='inner', left_on=['permno', 'repdats'], right_on=['permno', 'anndats'])
    ibes1 = ibes1.drop(['anndats', 'date'], axis=1).rename(columns={'cfacshr': 'cfacshr_rep'})

    ibes1['new_value'] = (ibes1.cfacshr_rep / ibes1.cfacshr_ann) * ibes1.value

    # Sanity check: there should be one most recent estimate for
    # a given firm-fiscal period end combination
    ibes1 = ibes1.sort_values(by=['ticker', 'fpedats', 'estimator', 'analys']).drop_duplicates()

    # Compute the median forecast based on estimates in the 90 days prior to the EAD

    grp_permno = ibes1.groupby(['ticker', 'fpedats', 'basis', 'repdats', 'act']).permno.max().reset_index()

    medest = ibes1.groupby(['ticker', 'fpedats', 'basis', 'repdats', 'act']).new_value.agg(['median', 'count']).reset_index()
    medest = pd.merge(medest, grp_permno, how='inner', on=['ticker', 'fpedats', 'basis', 'repdats', 'act'])
    medest = medest.rename(columns={'median': 'medest', 'count': 'numest'})

    ######################################
    # Step 5. Merge with Compustat Data  #
    ######################################

    # get items from fundq
    fundq = conn.raw_sql(f"""
                            select gvkey, fyearq, fqtr, conm, datadate, rdq, epsfxq, epspxq, cshoq, prccq,
                            ajexq, spiq, cshoq, cshprq, cshfdq, saleq, atq, fyr, datafqtr, cshoq*prccq as mcap
                            from comp.fundq
                            where consol='C' and popsrc='D' and indfmt='INDL' and datafmt='STD'
                            and datadate between '{crsp_begdate}' and '{crsp_enddate}'
                            """, date_cols=['datadate', 'datafqtr', 'rdq'])

    fundq = fundq.loc[((fundq.atq > 0) | (fundq.saleq.notna())) & (fundq.datafqtr.notna())]

    # Calculate link date ranges for givken gvkey and ticker combination

    gvkey_mindt1 = gvkey.groupby(['gvkey', 'ticker']).linkdt.min().reset_index().rename(columns={'linkdt': 'mindate'})
    gvkey_maxdt1 = gvkey.groupby(['gvkey', 'ticker']).linkenddt.max().reset_index().rename(columns={'linkenddt': 'maxdate'})
    gvkey_dt1 = pd.merge(gvkey_mindt1, gvkey_maxdt1, how='inner', on=['gvkey', 'ticker'])

    # Use the date range to merge
    comp = pd.merge(fundq, gvkey_dt1, how='left', on=['gvkey'])
    comp = comp.loc[(comp.ticker.notna()) & (comp.datadate <= comp.maxdate) & (comp.datadate >= comp.mindate)]

    # Merge with the median esitmates
    comp = pd.merge(comp, medest, how='left', left_on=['ticker', 'datadate'], right_on=['ticker', 'fpedats'])

    # Sort data and drop duplicates
    comp = comp.sort_values(by=['gvkey', 'fqtr', 'fyearq']).drop_duplicates()

    ###########################
    # Step 6. Calculate SUEs  #
    ###########################

    # block handling lag eps

    sue = comp.sort_values(by=['gvkey', 'fqtr', 'fyearq'])

    sue['dif_fyearq'] = sue.groupby(['gvkey', 'fqtr']).fyearq.diff()
    sue['laggvkey'] = sue.gvkey.shift(1)

    # handling same qtr previous year

    cond_year = sue.dif_fyearq == 1  # year increment is 1
    cond_year = cond_year.fillna(False)

    sue['lagadj'] = np.where(cond_year, sue.ajexq.shift(1), None)
    sue['lageps_p'] = np.where(cond_year, sue.epspxq.shift(1), None)
    sue['lageps_d'] = np.where(cond_year, sue.epsfxq.shift(1), None)
    sue['lagshr_p'] = np.where(cond_year, sue.cshprq.shift(1), None)
    sue['lagshr_d'] = np.where(cond_year, sue.cshfdq.shift(1), None)
    sue['lagspiq'] = np.where(cond_year, sue.spiq.shift(1), None)

    # handling first gvkey

    cond_gvkey = sue.gvkey != sue.laggvkey  # first.gvkey
    cond_gvkey = cond_gvkey.fillna(False)

    sue['lagadj'] = np.where(cond_gvkey, None, sue.lagadj)
    sue['lageps_p'] = np.where(cond_gvkey, None, sue.lageps_p)
    sue['lageps_d'] = np.where(cond_gvkey, None, sue.lageps_d)
    sue['lagshr_p'] = np.where(cond_gvkey, None, sue.lagshr_p)
    sue['lagshr_d'] = np.where(cond_gvkey, None, sue.lagshr_d)
    sue['lagspiq'] = np.where(cond_gvkey, None, sue.lagspiq)

    # handling reporting basis

    # Basis = P and missing are treated the same

    sue['actual1'] = np.where(sue.basis == 'D', sue.epsfxq / sue.ajexq, sue.epspxq / sue.ajexq)

    sue['actual2'] = np.where(sue.basis == 'D',
                              ((sue.epsfxq.fillna(0) - (0.65 * sue.spiq / sue.cshfdq).fillna(0)) / sue.ajexq).infer_objects(copy=False),
                              ((sue.epspxq.fillna(0) - (0.65 * sue.spiq / sue.cshprq).fillna(0)) / sue.ajexq).infer_objects(copy=False)
                              )

    sue['expected1'] = np.where(sue.basis == 'D', sue.lageps_d / sue.lagadj, sue.lageps_p / sue.lagadj)
    sue['expected2'] = np.where(sue.basis == 'D',
                                ((sue.lageps_d.fillna(0) - (0.65 * sue.lagspiq / sue.lagshr_d).fillna(0)) / sue.lagadj).infer_objects(copy=False),
                                ((sue.lageps_p.fillna(0) - (0.65 * sue.lagspiq / sue.lagshr_p).fillna(0)) / sue.lagadj).infer_objects(copy=False)
                                )

    # SUE calculations
    sue['sue1'] = (sue.actual1 - sue.expected1) / (sue.prccq / sue.ajexq)
    sue['sue2'] = (sue.actual2 - sue.expected2) / (sue.prccq / sue.ajexq)
    sue['sue3'] = (sue.act - sue.medest) / sue.prccq

    sue = sue[['ticker', 'permno', 'gvkey', 'conm', 'fyearq', 'fqtr', 'fyr', 'datadate', 'repdats', 'rdq',
               'sue1', 'sue2', 'sue3', 'basis', 'act', 'medest', 'numest', 'prccq', 'mcap']]

    # Shifting the announcement date to be the next trading day
    # Defining the day after the following quarterly EA as leadrdq1

    # unique rdq
    uniq_rdq = comp[['rdq']].drop_duplicates()

    # Create up to 5 days post rdq relative to rdq
    for i in range(0, 5):
        uniq_rdq[i] = uniq_rdq.rdq + datetime.timedelta(days=i)

    # reshape (transpose) for later join with crsp trading dates
    expand_rdq = uniq_rdq.set_index('rdq').stack().reset_index().\
        rename(columns={'level_1': 'post', 0: 'post_date'})

    # merge with crsp trading dates
    eads1 = pd.merge(expand_rdq, crsp_dats, how='left', left_on=['post_date'], right_on=['date'])

    # create the dgap (days gap) variable for min selection
    eads1['dgap'] = eads1.date - eads1.rdq

    # LOC deprecated, use reindex instead
    eads1 = eads1.reindex(eads1.groupby('rdq')['dgap'].idxmin()).rename(columns={'date': 'rdq1'})

    # create sue_final
    sue_final = pd.merge(sue, eads1[['rdq', 'rdq1']], how='left', on=['rdq'])
    sue_final = sue_final.sort_values(by=['gvkey', 'fyearq', 'fqtr'], ascending=[True, False, False]).drop_duplicates()

    #  Filter from Livnat & Mendenhall (2006):
    #- earnings announcement date is reported in Compustat
    #- the price per share is available from Compustat at fiscal quarter end
    #- price is greater than $1
    #- the market (book) equity at fiscal quarter end is available and is
    # EADs in Compustat and in IBES (if available)should not differ by more
    # than one calendar day larger than $5 mil.

    sue_final['leadrdq1'] = sue_final.rdq1.shift(1)  # next consecutive EAD
    sue_final['leadgvkey'] = sue_final.gvkey.shift(1)

    # If first gvkey then leadrdq1 = rdq1+3 months
    # Else leadrdq1 = previous rdq1

    sue_final['leadrdq1'] = np.where(sue_final.gvkey.eq(sue_final.leadgvkey).fillna(False),
                                      sue_final.rdq1.shift(1),
                                      sue_final.rdq1 + pd.DateOffset(months=3))

    sue_final['dgap'] = (sue_final.repdats - sue_final.rdq).fillna(pd.Timedelta(days=0))
    sue_final = sue_final.loc[(sue_final.rdq1 != sue_final.leadrdq1)]

    # Various conditioning for filtering
    cond1 = (sue_final.sue1.notna()) & (sue_final.sue2.notna()) & (sue_final.repdats.isna())
    cond2 = (sue_final.repdats.notna()) & (sue_final.dgap <= datetime.timedelta(days=1)) & (sue_final.dgap >= datetime.timedelta(days=-1))
    sue_final = sue_final.loc[cond1 | cond2]

    # Impose restriction on price and marketcap
    sue_final = sue_final.loc[(sue_final.rdq.notna()) & (sue_final.prccq > 1) & (sue_final.mcap > 5)]

    # Keep relevant columns
    sue_final = sue_final[['gvkey', 'ticker', 'permno', 'conm', 'dgap',
                           'fyearq', 'fqtr', 'datadate', 'fyr', 'rdq', 'rdq1', 'leadrdq1', 'repdats',
                           'mcap', 'medest', 'act', 'numest', 'basis', 'sue1', 'sue2', 'sue3']]

    #########################################
    # Step 7. Form Portfolios Based on SUE  #
    #########################################

    # Extract file of raw daily returns around and between EADs and link them
    # to Standardized Earnings Surprises for forming SUE-based portfolios

    # Records from dsf and dsi to calculate exret
    dsf = conn.raw_sql(f"""
                            select permno, date, prc, abs(prc*shrout) as mcap, ret from crsp.dsf
                            where date between '{crsp_begdate}' and '{crsp_enddate}'
                            """, date_cols=['date'])

    dsi = conn.raw_sql(f"""
                        select date, vwretd from crsp.dsi where date between '{crsp_begdate}' and '{crsp_enddate}'
                        """, date_cols=['date'])

    ds = pd.merge(dsf, dsi, how='left', on=['date'])
    ds['exret'] = ds.ret - ds.vwretd
    ds = ds.rename(columns={'vwretd': 'mkt'})

    # Records from sue_final that meet the condition
    sue_final_join = sue_final.loc[(sue_final.rdq.notna()) & (sue_final.leadrdq1.notna()) & (sue_final.permno.notna())
                                   & (sue_final.leadrdq1 - sue_final.rdq1 > datetime.timedelta(days=30))]
    sue_final_join1 = sue_final_join.copy()

    sue_final_join1['lb_date'] = sue_final_join.rdq1 - datetime.timedelta(days=5)
    sue_final_join1['ub_date'] = sue_final_join.leadrdq1 + datetime.timedelta(days=5)

    sfj_indexed = sue_final_join1[['permno', 'rdq1', 'leadrdq1', 'sue1', 'sue2', 'sue3', 'lb_date', 'ub_date']].set_index('permno')
    crsprets = ds.join(sfj_indexed, how='left', on='permno')
    crsprets = crsprets.loc[(crsprets.date <= crsprets.ub_date) & (crsprets.date >= crsprets.lb_date)]
    crsprets = crsprets.drop(['lb_date', 'ub_date'], axis=1)

    # To estimate the drift, sum daily returns over the period from
    # 1 day after the earnings announcement through the day of
    # the following quarterly earnings announcement

    temp = crsprets.sort_values(by=['permno', 'rdq1', 'date'])
    temp['lpermno'] = temp.permno.shift(1)

    # If first permno then lagmcap = missing
    # Else lagmcap = lag(mcap)
    temp['lagmcap'] = np.where(temp.permno.eq(temp.lpermno).fillna(False),
                               temp.mcap.shift(1),
                               None)

    temp = temp.loc[(temp.rdq1 <= temp.date) & (temp.date <= temp.leadrdq1)]

    # create count variable within the group
    temp['ncount'] = temp.groupby(['permno', 'rdq1']).cumcount()

    # One row per announcement and trading day; day 0 is rdq1 (as in pead.py Step 8)
    events = temp.drop_duplicates(subset=['permno', 'rdq1', 'date']).sort_values(by=['permno', 'rdq1', 'date'])
    events['day'] = events.groupby(['permno', 'rdq1']).cumcount()
    events = events[['permno', 'rdq1', 'date', 'day', 'sue3']].reset_index(drop=True)

    rets = ds.loc[ds.permno.isin(events.permno.unique()), ['permno', 'date', 'ret']]
    rets = rets.drop_duplicates(subset=['date', 'permno']).reset_index(drop=True)
    conn.close()
    return events, rets


def load_data(refresh=False):
    if refresh or not os.path.exists(CACHE):
        cwd = os.getcwd()
        os.chdir(WEEK6)                       # pead.py reads iclink.pkl from its own directory
        try:
            events, rets = build_signal_data()
        finally:
            os.chdir(cwd)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, 'wb') as f:
            pkl.dump({'events': events, 'rets': rets}, f)
    with open(CACHE, 'rb') as f:
        d = pkl.load(f)
    return d['events'], d['rets']



def load_mcap(permnos, refresh=False):
    """Daily market cap (|prc| x shrout) of the sample names, from CRSP."""
    if refresh or not os.path.exists(MCAP_CACHE):
        import wrds
        conn = wrds.Connection(wrds_username='cpehlivanian')
        mcap = conn.raw_sql(f"""select permno, date, abs(prc*shrout) as mcap from crsp.dsf
                                where permno in {tuple(int(p) for p in permnos)}
                                and date between '01/01/2017' and '12/31/2023'""", date_cols=['date'])
        conn.close()
        mcap.to_pickle(MCAP_CACHE)
    return pd.read_pickle(MCAP_CACHE)


def two_stage_cov(X, k):
    """Two-stage PCA estimate from X (n x T, rows = assets, de-meaned).

    Sigma = B diag(F) B' + diag(D), with B the top k left singular vectors, F the top k
    eigenvalues of X X' / (T - 1), and D the residual variances.
    """
    T = X.shape[1]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    B, F = U[:, :k], s[:k] ** 2 / (T - 1)
    E = X - B @ (s[:k, None] * Vt[:k])
    D = (E ** 2).sum(1) / (T - 1)
    return B, F, D


def factor_solve(B, F, D, V):
    """Sigma^-1 V for Sigma = B diag(F) B' + diag(D), by the Woodbury identity."""
    DV, DB = V / D[:, None], B / D[:, None]
    M = np.diag(1 / F) + B.T @ DB
    return DV - DB @ np.linalg.solve(M, B.T @ DV)


def factor_var(B, F, D, w):
    """w' Sigma w for Sigma = B diag(F) B' + diag(D)."""
    return ((B.T @ w) ** 2 * F).sum() + (w ** 2 * D).sum()


def stratified_weights(sue):
    """Long the top sue quintile, short the bottom one, equal-weighted, $1 per leg."""
    q = pd.qcut(sue, 5, labels=False, duplicates='drop')
    w = np.zeros(len(sue))
    top, bot = q == q.max(), q == q.min()
    w[top], w[bot] = 1 / top.sum(), -1 / bot.sum()
    return w


def mvo_weights(sue, B, F, D):
    """Dollar-neutral risk-reward MVO in closed form, scaled to TARGET_VOL."""
    sigma = np.sqrt((B ** 2 * F).sum(1) + D)
    z = norm.ppf((sue.rank().to_numpy() - 0.5) / len(sue))
    mu = sigma * z
    X = factor_solve(B, F, D, np.column_stack([mu, np.ones(len(mu))]))
    a, b = X[:, 1].sum(), X[:, 0].sum()
    w = X[:, 0] - b / a * X[:, 1]
    return w * TARGET_VOL / np.sqrt(252 * factor_var(B, F, D, w))


def backtest(W, R):
    """Daily returns and dollars traded of weights W set at the close of each day.

    W and R are dates x names; W[t] earns R[t + 1]. Costs follow pead.py: the trade at
    the close of t is from the drifted to the new weights, charged on t + 1.
    """
    Rz = np.nan_to_num(R)
    Wp = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])                  # held during t
    r = (Wp * Rz).sum(1)
    drift = Wp * (1 + Rz) / (1 + r)[:, None]
    trade = np.abs(W - drift).sum(1)
    traded = np.concatenate([[0.0], trade[:-1]])
    return r, traded


def stats(r, traded, w_gross):
    net = r - COST_BPS * 1e-4 * traded
    out = {}
    for name, x in [('gross', r), (f'net {COST_BPS} bps', net)]:
        out[(name, 'mean_ann')] = x.mean() * 252
        out[(name, 'vol_ann')] = x.std() * np.sqrt(252)
        out[(name, 'sharpe')] = x.mean() / x.std() * np.sqrt(252)
    on = w_gross > 0
    out[('', 'gross_lev')] = w_gross[on].mean()
    out[('', 'turnover')] = 0.5 * traded.sum() / w_gross[on].sum()     # one-way, per $ of gross, per day
    return out, net


def build_weights(events, rets):
    """Daily stratified and MVO weights (dates x names), set at the close of each day."""
    held = events.loc[events.day.between(PF_FIRST - 1, PF_LAST - 1) & events.sue3.notna()]
    held = held.sort_values('rdq1').drop_duplicates(subset=['date', 'permno'], keep='last')

    R = rets.pivot(index='date', columns='permno', values='ret').sort_index()
    S = held.pivot(index='date', columns='permno', values='sue3').reindex(index=R.index, columns=R.columns)
    Rv, Sv = R.to_numpy(dtype=float), S.to_numpy(dtype=float)
    n_dates, n_names = Rv.shape
    if N_UNIVERSE is not None:
        cap = load_mcap(R.columns).pivot(index='date', columns='permno', values='mcap')
        capv = cap.reindex(index=R.index, columns=R.columns).ffill().to_numpy(dtype=float)

    W_strat = np.zeros((n_dates, n_names))
    W_mvo = np.zeros((n_dates, n_names))
    pred_vol = {'Stratified': np.full(n_dates, np.nan), 'MVO': np.full(n_dates, np.nan)}
    model = None
    for t in range(COV_WINDOW - 1, n_dates):
        if model is None or (t - model['t']) % COV_EVERY == 0:
            win = Rv[t - COV_WINDOW + 1:t + 1].T                          # n x T, data up to t
            ok = np.isfinite(win).mean(1) >= MIN_OBS
            if N_UNIVERSE is not None:                                    # N largest at t
                c = np.where(ok, np.nan_to_num(capv[t]), 0.0)
                ok = np.zeros(n_names, dtype=bool)
                ok[np.argsort(-c)[:N_UNIVERSE]] = True
            X = win[ok] - np.nanmean(win[ok], axis=1, keepdims=True)
            B, F, D = two_stage_cov(np.nan_to_num(X), N_FACTORS)
            model = {'t': t, 'idx': np.flatnonzero(ok), 'B': B, 'F': F, 'D': D}

        active = np.isfinite(Sv[t, model['idx']])
        if active.sum() < MIN_NAMES:
            continue
        idx = model['idx'][active]
        B, F, D = model['B'][active], model['F'], model['D'][active]
        sue = pd.Series(Sv[t, idx])

        w = stratified_weights(sue)
        W_strat[t, idx] = w
        pred_vol['Stratified'][t] = np.sqrt(252 * factor_var(B, F, D, w))
        w = mvo_weights(sue, B, F, D)
        W_mvo[t, idx] = w
        pred_vol['MVO'][t] = np.sqrt(252 * factor_var(B, F, D, w))

    W = {'Stratified': pd.DataFrame(W_strat, index=R.index, columns=R.columns),
         'MVO': pd.DataFrame(W_mvo, index=R.index, columns=R.columns)}
    return W, R, pred_vol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true', help='rebuild the WRDS cache')
    args = parser.parse_args()

    events, rets = load_data(args.refresh)
    W, R, pred_vol = build_weights(events, rets)
    W_strat, W_mvo = W['Stratified'].to_numpy(), W['MVO'].to_numpy()
    Rv = R.to_numpy(dtype=float)

    # Evaluate from the first day with positions
    first = np.flatnonzero(np.abs(W_strat).sum(1) > 0)[0]
    dates = R.index[first + 1:]
    rows, nets = {}, {}
    for name, W in [('Stratified', W_strat), ('MVO', W_mvo)]:
        r, traded = backtest(W, Rv)
        w_gross = np.abs(W).sum(1)
        w_gross = np.concatenate([[np.nan], w_gross[:-1]])[first + 1:]
        rows[name], nets[name] = stats(r[first + 1:], traded[first + 1:], w_gross)
        rows[name][('', 'pred_vol')] = np.nanmean(pred_vol[name])

    table = pd.DataFrame(rows).T
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    print(f'PEAD (sue3), event days {PF_FIRST}..{PF_LAST}, {dates[0].date()} to {dates[-1].date()}')
    print(f'Universe: {N_UNIVERSE or "all"} names. Two-stage covariance: k = {N_FACTORS}, {COV_WINDOW}-day window, re-estimated every {COV_EVERY} days')
    print(table.round(3).to_string())

    # Equity curves of net returns, each scaled to 10% realized volatility (display only)
    os.makedirs(os.path.dirname(PLOT), exist_ok=True)
    plt.figure(figsize=(16, 10))
    for (name, x), c in zip(nets.items(), ['skyblue', 'orchid']):
        x = pd.Series(x, index=dates)
        x = x * 0.10 / (x.std() * np.sqrt(252))
        plt.plot(x.cumsum(), color=c, linewidth=3, label=f'{name} (Sharpe {rows[name][(f"net {COST_BPS} bps", "sharpe")]:.2f})')
    plt.axhline(0, color='grey', linewidth=1)
    plt.title(f'PEAD long-short portfolios, net of {COST_BPS} bps\n'
              'Cumulative return, each scaled to 10% annualized volatility', fontsize=20)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Cumulative return', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.savefig(PLOT)
    print(f'Plot saved to {PLOT}')
    return pd.DataFrame(nets, index=dates)


if __name__ == '__main__':
    main()
