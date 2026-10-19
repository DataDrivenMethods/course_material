#####################################################
# Post Earnings Announcement Drift, no look-ahead   #
# Based on pead.py (Qingyi (Freda) Song Drechsler)  #
#####################################################

# Same sample, SUE definitions and outputs as pead.py, with every use of
# information before it was public removed:
#
# 1. Announcement timing. pead.py starts the event at the first trading day on or
#    after Compustat's rdq and forms portfolios at its close, which is before the
#    announcement when the firm reports after the close. Here the information day
#    d0 is the first trading day whose close (ann_cutoff) is after the IBES
#    announcement time (anndats + anntims). Positions are set at the close of d0
#    and earn returns from d0 + 1. A missing time counts as after the close.
# 2. Forecasts. Only estimates issued before the announcement date are used, and
#    each analyst's latest estimate is chosen among those (pead.py chose the latest
#    first, then dropped it if it came after the announcement).
# 3. Universe. pead.py keeps every firm ever in the S&P 500, including firms that
#    joined later. Here an announcement counts only if the firm is a member on d0.
# 4. Next announcement. pead.py drops events with no next EAD or one within 30
#    days, which requires knowing the future. Here every event is kept and a
#    position is held until event day 50 or the next announcement's d0.
# 5. Quintiles. pead.py ranks SUE across the whole sample on each event day. Here
#    each announcement is ranked against announcements of the previous 91 days.
#
# Not fixed: Compustat fundq holds restated values, so sue1 and sue2 can use
# numbers that differ from those first reported. sue3 (IBES) does not.

import math
import os

import pandas as pd
import numpy as np
import wrds
import matplotlib.pyplot as plt
import pickle as pkl
import statsmodels.api as sm
from dateutil.relativedelta import *
import datetime

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

# Announcements at or after this New York time count toward the next trading day
ann_cutoff = '15:30:00'

os.makedirs('./plot', exist_ok=True)

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

# All companies that were ever included in S&P 500 index; membership on the
# information day is imposed in Step 7 (sp500 keeps the start/ending dates)

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

# Extract estimates from IBES Unadjusted file
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

# Actuals with their announcement date (repdats) and time (reptims)
ibes_act = conn.raw_sql(f"""
                        select ticker, anndats as repdats, anntims as reptims, value as act,
                        pends as fpedats, pdicity
                        from ibes.actu_epsus
                        where pends between '{begdate}' and '{enddate}'
                        and pdicity='QTR'
                        """, date_cols=['repdats', 'fpedats'])
ibes_act = ibes_act.drop_duplicates(subset=['ticker', 'fpedats'])

# Keep estimates issued 1 to 90 days before the announcement date, BEFORE
# choosing each analyst's latest one (same-day estimates may follow the release)
ibes_temp = pd.merge(ibes_temp, ibes_act.drop(['pdicity'], axis=1), how='inner', on=['ticker', 'fpedats'])
ibes_temp['dgap'] = ibes_temp.repdats - ibes_temp.anndats
ibes_temp = ibes_temp.loc[(ibes_temp.dgap >= datetime.timedelta(days=1)) & (ibes_temp.dgap <= datetime.timedelta(days=90))]
ibes_temp = ibes_temp.drop(['dgap'], axis=1)

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

# Keep the latest pre-announcement estimate of each analyst
ibes1 = ibes.groupby(['ticker', 'fpedats', 'estimator', 'analys']).tail(1)

#######################################
# Step 3. CRSP Adjustment Factors     #
#######################################

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

# Compute the median forecast based on estimates in the 90 days prior to the EAD

grp_permno = ibes1.groupby(['ticker', 'fpedats', 'basis', 'repdats', 'act']).permno.max().reset_index()

medest = ibes1.groupby(['ticker', 'fpedats', 'basis', 'repdats', 'act']).new_value.agg(['median', 'count']).reset_index()
medest = pd.merge(medest, grp_permno, how='inner', on=['ticker', 'fpedats', 'basis', 'repdats', 'act'])
medest = medest.rename(columns={'median': 'medest', 'count': 'numest'})

# announcement time of the actual
medest = pd.merge(medest, ibes_act[['ticker', 'fpedats', 'reptims']], how='left', on=['ticker', 'fpedats'])

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

sue = sue[['ticker', 'permno', 'gvkey', 'conm', 'fyearq', 'fqtr', 'fyr', 'datadate', 'repdats', 'reptims', 'rdq',
           'sue1', 'sue2', 'sue3', 'basis', 'act', 'medest', 'numest', 'prccq', 'mcap']]

###############################################
# Step 7. Information Day and Sample Filters  #
###############################################

# Information day d0: the first trading day whose ann_cutoff is after the
# announcement. Announced before the cutoff on a trading day -> that day;
# otherwise (after the cutoff, a non-trading day, or time unknown) -> the next
# trading day. Without IBES the Compustat rdq is used, time unknown.

trading_days = np.sort(crsp_dats.date.dropna().unique())


def to_seconds(t):
    """IBES time ('HH:MM:SS', datetime.time or seconds) -> seconds after midnight."""
    s = t.astype(str).str.strip()
    secs = pd.to_timedelta(s.where(s.str.contains(':')), errors='coerce').dt.total_seconds()
    return secs.fillna(pd.to_numeric(t, errors='coerce'))


ann_date = sue.repdats.fillna(sue.rdq)
ann_secs = to_seconds(sue.reptims).where(sue.repdats.notna())
# 00:00:00 usually means "time not recorded", so it counts as unknown
same_day = (ann_secs > 0) & (ann_secs < pd.Timedelta(ann_cutoff).total_seconds())
pos = np.where(same_day,
               np.searchsorted(trading_days, ann_date.to_numpy(), side='left'),    # on or after
               np.searchsorted(trading_days, ann_date.to_numpy(), side='right'))   # strictly after
pos = np.where(ann_date.notna(), pos, len(trading_days))
sue['d0'] = pd.Series(np.append(trading_days, np.datetime64('NaT'))[pos], index=sue.index)
sue['after_close'] = ~same_day

sue_final = sue.loc[sue.d0.notna()].copy()

#  Filter from Livnat & Mendenhall (2006):
#- earnings announcement date is reported in Compustat
#- the price per share is available from Compustat at fiscal quarter end
#- price is greater than $1
#- the market (book) equity at fiscal quarter end is available and is larger than $5 mil.
#- EADs in Compustat and in IBES (if available) should not differ by more than one calendar day

sue_final['dgap'] = (sue_final.repdats - sue_final.rdq).fillna(pd.Timedelta(days=0))

# Various conditioning for filtering
cond1 = (sue_final.sue1.notna()) & (sue_final.sue2.notna()) & (sue_final.repdats.isna())
cond2 = (sue_final.repdats.notna()) & (sue_final.dgap <= datetime.timedelta(days=1)) & (sue_final.dgap >= datetime.timedelta(days=-1))
sue_final = sue_final.loc[cond1 | cond2]

# Impose restriction on price and marketcap
sue_final = sue_final.loc[(sue_final.rdq.notna()) & (sue_final.prccq > 1) & (sue_final.mcap > 5) & (sue_final.permno.notna())]
sue_final['permno'] = sue_final.permno.astype(int)

# S&P 500 member on the information day (not just at some point in its history)
spells = sue_final[['permno', 'd0']].drop_duplicates().merge(sp500[['permno', 'start', 'ending']], on='permno')
members = spells.loc[(spells.start <= spells.d0) & (spells.d0 <= spells.ending), ['permno', 'd0']].drop_duplicates()
sue_final = sue_final.merge(members, on=['permno', 'd0'])

# One announcement per stock and information day (the latest fiscal quarter)
sue_final = sue_final.sort_values(by=['permno', 'd0', 'fyearq', 'fqtr']).drop_duplicates(subset=['permno', 'd0'], keep='last')

# The next announcement ends the drift window; it is known when it happens, so no
# event is dropped for lacking one. Without one, the window ends 120 days later.
sue_final['lead_d0'] = sue_final.groupby('permno').d0.shift(-1)
sue_final['end_date'] = sue_final.lead_d0.fillna(sue_final.d0 + pd.Timedelta(days=120))

# Keep relevant columns
sue_final = sue_final[['gvkey', 'ticker', 'permno', 'conm', 'dgap',
                       'fyearq', 'fqtr', 'datadate', 'fyr', 'rdq', 'repdats', 'reptims', 'after_close',
                       'd0', 'lead_d0', 'end_date', 'mcap', 'medest', 'act', 'numest', 'basis', 'sue1', 'sue2', 'sue3']]

print(f'{len(sue_final)} announcements; {sue_final.after_close.mean():.0%} after the cutoff or with unknown time')

# sue3 quintile of each announcement from breakpoints of the previous 91 days'
# announcements only (pead.py ranked on the whole sample)
q_min_past = 50                 # minimum past announcements for breakpoints

ann = sue_final[['permno', 'd0', 'sue3']].dropna().reset_index(drop=True)
ann['q'] = np.nan
for d, idx in ann.groupby('d0').groups.items():
    past = ann.sue3[(ann.d0 < d) & (ann.d0 >= d - pd.Timedelta(days=91))]
    if len(past) >= q_min_past:
        bp = np.quantile(past, [0.2, 0.4, 0.6, 0.8])
        ann.loc[idx, 'q'] = np.searchsorted(bp, ann.sue3[idx], side='right') + 1
sue_final = sue_final.merge(ann[['permno', 'd0', 'q']], how='left', on=['permno', 'd0'])

#########################################
# Step 8. Event-Time SUE Portfolios     #
#########################################

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

# Daily returns from 5 days before d0 (for the lagged market cap) to the end of the window
sue_final_join1 = sue_final.copy()
sue_final_join1['lb_date'] = sue_final_join1.d0 - datetime.timedelta(days=5)

sfj_indexed = sue_final_join1[['permno', 'd0', 'end_date', 'q', 'sue1', 'sue2', 'sue3', 'lb_date']].set_index('permno')
crsprets = ds.join(sfj_indexed, how='inner', on='permno')
crsprets = crsprets.loc[(crsprets.date <= crsprets.end_date) & (crsprets.date >= crsprets.lb_date)]
crsprets = crsprets.drop(['lb_date'], axis=1)

# lagged market cap within each announcement's rows
temp = crsprets.drop_duplicates(subset=['permno', 'd0', 'date']).sort_values(by=['permno', 'd0', 'date'])
temp['lagmcap'] = temp.groupby(['permno', 'd0']).mcap.shift(1)

# Event day 0 = d0 (positions set at its close); returns from day 1 on are tradable
temp = temp.loc[temp.d0 <= temp.date]
temp['ncount'] = temp.groupby(['permno', 'd0']).cumcount()

# Value-weighted excess return of each quintile on each event day
peadrets3 = temp.loc[temp.q.notna() & temp.lagmcap.notna()].copy()
peadrets3['sue3r'] = peadrets3.q.astype(int) - 1
peadrets3['total_lagmcap'] = peadrets3.groupby(['ncount', 'sue3r']).lagmcap.transform('sum')
peadrets3['wt_exret'] = peadrets3.exret * peadrets3.lagmcap / peadrets3.total_lagmcap
peadsue3port = peadrets3.groupby(['ncount', 'sue3r']).wt_exret.sum().reset_index()

# set ncount=0 all five portfolio weighted returns to be 0
peadsue3port['wt_exret'] = np.where(peadsue3port.ncount == 0, 0, peadsue3port.wt_exret)

peadsue3port = peadsue3port.pivot_table(index=['ncount'], columns='sue3r', values='wt_exret').reset_index()

# keep only first 50 days after EADs
peadsue3port = peadsue3port.loc[peadsue3port.ncount <= 50]

# Cumulating Excess Returns
for j in range(5):
    peadsue3port[f'sueport{j + 1}'] = peadsue3port[j].cumsum()

plt.figure(figsize=(16, 10))
plt.title('CARs following the information day for Analyst-based SUE portfolios, no look-ahead \n Sample: S&P 500 members', fontsize=20)
plt.xlabel('Event time, t=0 is the information day (positions set at its close)', fontsize=16)
plt.ylabel('Cumulative Value-Weighted Excess Returns', fontsize=16)

plt.plot('ncount', 'sueport1', data=peadsue3port, color='skyblue', linewidth=3, label="sue1: Most Negative SUE Port")
plt.plot('ncount', 'sueport2', data=peadsue3port, color='olive', linewidth=3, label="sue2")
plt.plot('ncount', 'sueport3', data=peadsue3port, color='gold', linewidth=3, label="sue3")
plt.plot('ncount', 'sueport4', data=peadsue3port, color='coral', linewidth=3, label="sue4")
plt.plot('ncount', 'sueport5', data=peadsue3port, color='orchid', linewidth=3, label="sue5: Most Positive SUE Port")

plt.legend(loc="best", fontsize=16)

plt.savefig('./plot/pead_corrected_car.png')
print("Plot saved to ./plot/pead_corrected_car.png")

#############################################
# Step 9. IC, t-stats and IC decay of SUE   #
#############################################

# Event-time IC: in each calendar month, the Spearman correlation across the
# announcements with d0 in that month between SUE and the excess return after d0.
#   cum k: sum of exret over event days 1..k (the CAR in the plot above)
#   lag k: exret on event day k alone
# The IC is averaged over months; t-stats use Newey-West standard errors. A
# cumulative window of k trading days overlaps the next ~k/21 months, so it uses
# ceil(k/21) lags; single-day ICs use none. Day 1 is tradable here.

ic_min_names = 20                       # minimum announcements in a month for an IC
ic_days = range(1, 51)                  # event days, as in the plot
ic_horizons = [1, 2, 5, 10, 20, 50]     # rows of the summary table

# one row per announcement (permno, d0), one column per event day (0 = d0)
events = temp.rename(columns={'ncount': 'day'})
exret_ev = events.pivot(index=['permno', 'd0'], columns='day', values='exret')
sue_ev = events.groupby(['permno', 'd0'])[['sue1', 'sue2', 'sue3']].first().reindex(exret_ev.index)
ev_month = exret_ev.index.get_level_values('d0').to_period('M')

# CAR over days 1..k; NaN once a day is missing (e.g. after the next announcement)
car_ev = exret_ev.loc[:, 1:].cumsum(axis=1, skipna=False)


def monthly_ic(sue, ret):
    """Spearman IC of sue and ret across each month's announcements."""
    df = pd.DataFrame({'sue': sue.to_numpy(), 'ret': ret.to_numpy(), 'month': ev_month}).dropna()
    df[['sue', 'ret']] = df.groupby('month')[['sue', 'ret']].rank()
    g = df.groupby('month')
    ic = g.apply(lambda d: d.sue.corr(d.ret), include_groups=False)
    return ic[g.size() >= ic_min_names]


def nw_stats(ic, lags):
    """Mean, Newey-West t-stat, hit rate and number of months of an IC series."""
    ic = ic.dropna()
    res = sm.OLS(ic.to_numpy(), np.ones(len(ic))).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return {'ic': res.params[0], 't': res.tvalues[0], 'hit': (ic > 0).mean(), 'months': len(ic)}


rows = []
for s in ['sue1', 'sue2', 'sue3']:
    for k in ic_days:
        cum = nw_stats(monthly_ic(sue_ev[s], car_ev[k]), math.ceil(k / 21))
        lag = nw_stats(monthly_ic(sue_ev[s], exret_ev[k]), 0)
        rows.append({'sue': s, 'k': k,
                     'ic_cum': cum['ic'], 't_cum': cum['t'], 'hit_cum': cum['hit'], 'months': cum['months'],
                     'ic_lag': lag['ic'], 't_lag': lag['t']})
ic_decay = pd.DataFrame(rows).set_index(['sue', 'k'])

# Summary: IC of SUE with the CAR over days 1..k
ic_summary = ic_decay.loc[(slice(None), ic_horizons), ['ic_cum', 't_cum', 'hit_cum', 'months']]
print('IC of SUE with CAR over event days 1..k (monthly Spearman, Newey-West t)')
print(ic_summary.unstack('sue').round(3).to_string())

print('\nIC decay for sue3: cumulative (1..k) and single-day (day k)')
print(ic_decay.loc['sue3', ['ic_cum', 't_cum', 'ic_lag', 't_lag']].loc[ic_horizons].round(3).to_string())

# Plot: cumulative IC by SUE measure, and single-day IC of sue3 with its 95% band
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
for s, c in zip(['sue1', 'sue2', 'sue3'], ['skyblue', 'gold', 'orchid']):
    ax1.plot(ic_days, ic_decay.loc[s, 'ic_cum'], color=c, linewidth=3, label=s)
ax1.axhline(0, color='grey', linewidth=1)
ax1.set_title('IC of SUE with CAR over days 1..k', fontsize=16)
ax1.set_xlabel('k (trading days after the information day)', fontsize=14)
ax1.set_ylabel('Mean monthly Spearman IC', fontsize=14)
ax1.legend(fontsize=12)

d3 = ic_decay.loc['sue3']
ax2.bar(ic_days, d3['ic_lag'], color='orchid')
ax2.plot(ic_days, 1.96 * d3['ic_lag'] / d3['t_lag'], color='grey', linestyle='--', label='±1.96 s.e.')
ax2.plot(ic_days, -1.96 * d3['ic_lag'] / d3['t_lag'], color='grey', linestyle='--')
ax2.axhline(0, color='grey', linewidth=1)
ax2.set_title('IC of sue3 with the excess return on day k', fontsize=16)
ax2.set_xlabel('k (trading days after the information day)', fontsize=14)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.savefig('./plot/pead_corrected_ic_decay.png')
print("Plot saved to ./plot/pead_corrected_ic_decay.png")

############################################
# Step 10. Calendar-time SUE portfolios    #
############################################

# Each announcement's sue3 quintile (q, from past breakpoints) is bought at the
# close of event day pf_first - 1 and held through event day pf_last (or the next
# announcement, where the window ends). Each day the quintile portfolios are
# value-weighted (market cap at the previous close) and earn excess returns over
# the CRSP value-weighted index. Costs are pf_cost_bps per dollar traded, on the
# trade from the drifted to the new weights, charged the next day; the long-short
# (Q5 - Q1) pays both legs.

pf_first, pf_last = 1, 50       # event days held (day 1 is tradable here)
pf_cost_bps = 5

# Positions held at the close of each day: event days pf_first - 1 .. pf_last - 1
held = events.loc[events.day.between(pf_first - 1, pf_last - 1) & events.q.notna(), ['permno', 'date', 'q']]
held = held.drop_duplicates(subset=['date', 'permno'])

# date x permno matrices on the CRSP calendar
pf_permnos = held.permno.unique()
ds_pf = ds.loc[ds.permno.isin(pf_permnos)].drop_duplicates(subset=['date', 'permno'])
R = ds_pf.pivot(index='date', columns='permno', values='exret')
cap = ds_pf.pivot(index='date', columns='permno', values='mcap').reindex_like(R)
Q = held.pivot(index='date', columns='permno', values='q').reindex_like(R)

Rv = R.to_numpy()
valid = ~np.isnan(Rv)
Rz = np.where(valid, Rv, 0.0)
gross, traded = {}, {}
for j in range(1, 6):
    base = np.where(Q.to_numpy() == j, np.nan_to_num(cap.to_numpy()), 0.0)
    tot = base.sum(1, keepdims=True)
    W = np.divide(base, tot, out=np.zeros_like(base), where=tot > 0)      # weights at the close of t
    Wp = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])                   # held during t
    Wv = np.where(valid, Wp, 0.0)
    wsum = Wv.sum(1)
    r = np.divide((Wv * Rz).sum(1), wsum, out=np.full(len(wsum), np.nan), where=wsum > 0)
    drift = Wp * (1 + Rz) / (1 + np.nan_to_num(r))[:, None]               # weights after t's returns
    trade = np.abs(W - drift).sum(1)                                      # sum |dW| at the close of t
    gross[f'Q{j}'] = r
    traded[f'Q{j}'] = np.concatenate([[np.nan], trade[:-1]])              # charged on t + 1

pf_gross = pd.DataFrame(gross, index=R.index).dropna(how='all')
pf_trade = pd.DataFrame(traded, index=R.index).reindex(pf_gross.index)
pf_gross['Q5-Q1'] = pf_gross.Q5 - pf_gross.Q1
pf_trade['Q5-Q1'] = pf_trade.Q5 + pf_trade.Q1
pf_net = pf_gross - pf_cost_bps * 1e-4 * pf_trade


def pf_stats(rets):
    """Annualized mean, volatility and Sharpe ratio of daily returns."""
    return pd.DataFrame({'mean_ann': rets.mean() * 252, 'vol_ann': rets.std() * np.sqrt(252),
                         'sharpe': rets.mean() / rets.std() * np.sqrt(252)})


pf_table = pd.concat({'gross': pf_stats(pf_gross), f'net {pf_cost_bps} bps': pf_stats(pf_net)}, axis=1)
pf_table['turnover'] = 0.5 * pf_trade.mean()      # one-way, per day
print(f'\nSUE quintile portfolios, event days {pf_first}..{pf_last}, value-weighted excess returns')
print(pf_table.round(3).to_string())

# Equity curves (growth of $1): gross, then net of costs
pf_colors = {'Q1': 'skyblue', 'Q2': 'olive', 'Q3': 'gold', 'Q4': 'coral', 'Q5': 'orchid', 'Q5-Q1': 'black'}
for rets, label, fname in [(pf_gross, '0 bps cost', 'pead_corrected_equity_0bps.png'),
                           (pf_net, f'{pf_cost_bps} bps cost', f'pead_corrected_equity_{pf_cost_bps}bps.png')]:
    plt.figure(figsize=(16, 10))
    for c in rets.columns:
        plt.plot((1 + rets[c].fillna(0)).cumprod(), color=pf_colors[c], linewidth=3,
                 linestyle='--' if c == 'Q5-Q1' else '-', label=c)
    plt.axhline(1, color='grey', linewidth=1)
    plt.title(f'SUE quintile portfolios, {label}, no look-ahead\n Value-weighted excess returns, event days '
              f'{pf_first}..{pf_last}, S&P 500 members', fontsize=20)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Growth of $1', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.savefig(f'./plot/{fname}')
    print(f'Plot saved to ./plot/{fname}')
