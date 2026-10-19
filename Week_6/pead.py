#####################################
# Post Earnings Announcement Drift  #
# Qingyi (Freda) Song Drechsler     #
# Date: June 2019                   #
# Update: Jan 2021                  #
#####################################

# Update Note: this version uses CRSP data to extract S&P500 Index constituents,
# as comp.idxcst_hist data is no longer available on WRDS
# Also updated various .loc usage that has been deprecated by Python.
#
# Additional fixes (2025):
# - Fixed NA comparison issues in np.where() by using .eq().fillna(False)
# - Fixed FutureWarning for fillna downcasting by using .infer_objects(copy=False)
# - Reduced sample period to 2017-2018 for memory efficiency

import pandas as pd
import numpy as np
import wrds
import matplotlib.pyplot as plt
import pickle as pkl
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

# Form quintiles based on SUE
peadrets = temp.sort_values(by=['ncount', 'permno', 'rdq1']).drop_duplicates()

peadrets['sue1r'] = peadrets.groupby('ncount')['sue1'].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))
peadrets['sue2r'] = peadrets.groupby('ncount')['sue2'].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))
peadrets['sue3r'] = peadrets.groupby('ncount')['sue3'].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))

# Form portfolios on Compustat-based SUEs (=sue1 or =sue2) or IBES-based SUE (=sue3)
# Code uses sue3

peadrets3 = peadrets.loc[peadrets.sue3r.notna()].sort_values(by=['ncount', 'sue3'])
peadrets3['sue3r'] = peadrets3['sue3r'].astype(int)

# Form value-weighted exret
# Calculate group weight sum
grp_lagmcap = peadrets3.groupby(['ncount', 'sue3r']).lagmcap.sum().reset_index().rename(columns={'lagmcap': 'total_lagmcap'})

# join group weight sum back to the df
peadrets3 = pd.merge(peadrets3, grp_lagmcap, how='left', on=['ncount', 'sue3r'])

# vw exret
peadrets3 = peadrets3.loc[peadrets3.total_lagmcap > 0]

peadrets3['wt_exret'] = peadrets3.exret * peadrets3.lagmcap / peadrets3.total_lagmcap
peadsue3port = peadrets3.groupby(['ncount', 'sue3r']).wt_exret.sum().reset_index()

# set ncount=0 all five portfolio weighted returns to be 0
peadsue3port['wt_exret'] = np.where(peadsue3port.ncount == 0, 0, peadsue3port.wt_exret)

# transpose table for cumulative return calculation
peadsue3port = peadsue3port.pivot_table(index=['ncount'], columns='sue3r')

# reset column index level
peadsue3port.columns = [col[1] for col in peadsue3port.columns]
peadsue3port = peadsue3port.reset_index()

# keep only first 50 days after EADs
peadsue3port = peadsue3port.loc[peadsue3port.ncount <= 50]

# Cumulating Excess Returns

peadsue3port['sueport1'] = peadsue3port[0].cumsum()
peadsue3port['sueport2'] = peadsue3port[1].cumsum()
peadsue3port['sueport3'] = peadsue3port[2].cumsum()
peadsue3port['sueport4'] = peadsue3port[3].cumsum()
peadsue3port['sueport5'] = peadsue3port[4].cumsum()

# Plotting the output
import os

# Create plot directory if it doesn't exist
os.makedirs('./plot', exist_ok=True)

plt.figure(figsize=(16, 10))
plt.title('CARs following EAD for Analyst-based SUE portfolios \n Sample: S&P 500 members', fontsize=20)
plt.xlabel('Event time, t=0 is Earnings Annoucement Date', fontsize=16)
plt.ylabel('Cumulative Value-Weighted Excess Returns', fontsize=16)

plt.plot('ncount', 'sueport1', data=peadsue3port, color='skyblue', linewidth=3, label="sue1: Most Negative SUE Port")
plt.plot('ncount', 'sueport2', data=peadsue3port, color='olive', linewidth=3, label="sue2")
plt.plot('ncount', 'sueport3', data=peadsue3port, color='gold', linewidth=3, label="sue3")
plt.plot('ncount', 'sueport4', data=peadsue3port, color='coral', linewidth=3, label="sue4")
plt.plot('ncount', 'sueport5', data=peadsue3port, color='orchid', linewidth=3, label="sue5: Most Positive SUE Port")

plt.legend(loc="best", fontsize=16)

plt.savefig('./plot/pead_08_18.png')
print("Plot saved to ./plot/pead_08_18.png")

#############################################
# Step 8. IC, t-stats and IC decay of SUE   #
#############################################

# Event-time IC: in each calendar month, the Spearman correlation across the
# announcements made that month (rdq1 in the month) between SUE and the excess
# return after the EAD. Two return windows for event day k:
#   cum k: sum of exret over event days 1..k (the CAR in the plot above)
#   lag k: exret on event day k alone
# The IC is averaged over months; t-stats use Newey-West standard errors. A
# cumulative window of k trading days overlaps the next ~k/21 months, so it uses
# ceil(k/21) lags; single-day ICs use none.
# Note: day 1 contains the announcement reaction when the EA comes after the close of rdq1.

import math
import statsmodels.api as sm

ic_min_names = 20                       # minimum announcements in a month for an IC
ic_days = range(1, 51)                  # event days, as in the plot
ic_horizons = [1, 2, 5, 10, 20, 50]     # rows of the summary table

# one row per announcement (permno, rdq1), one column per event day (0 = rdq1)
events = temp.drop_duplicates(subset=['permno', 'rdq1', 'date']).sort_values(by=['permno', 'rdq1', 'date'])
events['day'] = events.groupby(['permno', 'rdq1']).cumcount()
exret_ev = events.pivot(index=['permno', 'rdq1'], columns='day', values='exret')
sue_ev = events.groupby(['permno', 'rdq1'])[['sue1', 'sue2', 'sue3']].first().reindex(exret_ev.index)
ev_month = exret_ev.index.get_level_values('rdq1').to_period('M')

# CAR over days 1..k; NaN once a day is missing (e.g. after the next EAD)
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
ax1.set_xlabel('k (trading days after EAD)', fontsize=14)
ax1.set_ylabel('Mean monthly Spearman IC', fontsize=14)
ax1.legend(fontsize=12)

d3 = ic_decay.loc['sue3']
ax2.bar(ic_days, d3['ic_lag'], color='orchid')
ax2.plot(ic_days, 1.96 * d3['ic_lag'] / d3['t_lag'], color='grey', linestyle='--', label='±1.96 s.e.')
ax2.plot(ic_days, -1.96 * d3['ic_lag'] / d3['t_lag'], color='grey', linestyle='--')
ax2.axhline(0, color='grey', linewidth=1)
ax2.set_title('IC of sue3 with the excess return on day k', fontsize=16)
ax2.set_xlabel('k (trading days after EAD)', fontsize=14)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.savefig('./plot/pead_ic_decay.png')
print("Plot saved to ./plot/pead_ic_decay.png")

###########################################
# Step 9. Calendar-time SUE portfolios    #
###########################################

# Tradable version of the drift: each announcement is put in a sue3 quintile using
# breakpoints from the announcements of the previous 91 days only (no look-ahead),
# bought at the close of event day pf_first - 1 and held through event day pf_last
# (or the next EAD, where the drift window ends). Day 1 is skipped because it
# contains the announcement reaction for after-close announcers.
# Each day the quintile portfolios are value-weighted (market cap at the previous
# close) and earn excess returns over the CRSP value-weighted index. Costs are
# pf_cost_bps per dollar traded, on the trade from the drifted to the new weights,
# charged the next day; the long-short (Q5 - Q1) pays both legs.

pf_first, pf_last = 2, 50       # event days held
pf_min_past = 50                # minimum past announcements for breakpoints
pf_cost_bps = 5

# sue3 quintile of each announcement from the previous 91 days' breakpoints
ann = sue_ev[['sue3']].dropna().reset_index()
ann['q'] = np.nan
for d, idx in ann.groupby('rdq1').groups.items():
    past = ann.sue3[(ann.rdq1 < d) & (ann.rdq1 >= d - pd.Timedelta(days=91))]
    if len(past) >= pf_min_past:
        bp = np.quantile(past, [0.2, 0.4, 0.6, 0.8])
        ann.loc[idx, 'q'] = np.searchsorted(bp, ann.sue3[idx], side='right') + 1

# Positions held at the close of each day: event days pf_first - 1 .. pf_last - 1
held = events.loc[events.day.between(pf_first - 1, pf_last - 1), ['permno', 'rdq1', 'date']]
held = held.merge(ann[['permno', 'rdq1', 'q']].dropna(), on=['permno', 'rdq1'])
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
print(f'SUE quintile portfolios, event days {pf_first}..{pf_last}, value-weighted excess returns')
print(pf_table.round(3).to_string())

# Equity curves (growth of $1): gross, then net of costs
pf_colors = {'Q1': 'skyblue', 'Q2': 'olive', 'Q3': 'gold', 'Q4': 'coral', 'Q5': 'orchid', 'Q5-Q1': 'black'}
for rets, label, fname in [(pf_gross, '0 bps cost', 'pead_equity_0bps.png'),
                           (pf_net, f'{pf_cost_bps} bps cost', f'pead_equity_{pf_cost_bps}bps.png')]:
    plt.figure(figsize=(16, 10))
    for c in rets.columns:
        plt.plot((1 + rets[c].fillna(0)).cumprod(), color=pf_colors[c], linewidth=3,
                 linestyle='--' if c == 'Q5-Q1' else '-', label=c)
    plt.axhline(1, color='grey', linewidth=1)
    plt.title(f'SUE quintile portfolios, {label}\n Value-weighted excess returns, event days '
              f'{pf_first}..{pf_last}, S&P 500 members', fontsize=20)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Growth of $1', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.savefig(f'./plot/{fname}')
    print(f'Plot saved to ./plot/{fname}')
