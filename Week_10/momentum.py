import pandas as pd
import numpy as np
import wrds
import matplotlib.pyplot as plt
from pandas.tseries.offsets import *
from scipy import stats

###################
# Connect to WRDS #
###################
conn=wrds.Connection(wrds_username="cpehlivanian")

###################
# CRSP Block      #
###################
# sql similar to crspmerge macro
# added exchcd=-2,-1,0 to address the issue that stocks temp stopped trading
# without exchcd=-2,-1, 0 the non-trading months will be tossed out in the output
# leading to wrong cumret calculation in momentum step
# Code	Definition
# -2	Halted by the NYSE or AMEX
# -1	Suspended by the NYSE, AMEX, or NASDAQ
# 0	Not Trading on NYSE, AMEX, or NASDAQ
# 1	New York Stock Exchange
# 2	American Stock Exchange

# Trading period. momentum_backtest.py sets TRADE_START / TRADE_END from its
# --start / --end options; these defaults apply when momentum.py runs alone.
TRADE_START = pd.Timestamp(globals().get('TRADE_START', '1995-01-01'))
TRADE_END = pd.Timestamp(globals().get('TRADE_END', '1999-12-31'))

# Monthly data starts 12 months earlier: the first formation date is the month
# end before TRADE_START and its signal uses the 11 months before that.
START_DATE = (TRADE_START.to_period('M') - 12).to_timestamp().strftime('%Y-%m-%d')
END_DATE = TRADE_END.strftime('%Y-%m-%d')

crsp_m = conn.raw_sql(f"""
                      select a.permno, a.permco, b.ncusip, a.date, 
                      b.shrcd, b.exchcd, b.siccd,
                      a.ret, a.vol, a.shrout, a.prc, a.cfacpr, a.cfacshr
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date between '{START_DATE}' and '{END_DATE}'
                      and b.exchcd between -2 and 2
                      and b.shrcd between 10 and 11
                      """) 

# Change variable format to int
crsp_m[['permco','permno','shrcd','exchcd']]=\
    crsp_m[['permco','permno','shrcd','exchcd']].astype(int)

# Line up date to be end of month
crsp_m['date']=pd.to_datetime(crsp_m['date'])

#######################################################
# 12-1 Momentum Signal (Jegadeesh-Titman)             #
# Formation at end of month t: cumulative return over #
# months t-11 .. t-1. Month t itself is skipped       #
# (short-term reversal). Hold for 1 month.            #
#######################################################

FORM_MONTHS = 11   # months t-11 .. t-1
SKIP_MONTHS = 1    # skip month t
MIN_PRICE = 5.0    # exclude stocks priced below $5 at formation

_tmp_crsp = crsp_m[['permno','date','ret','prc']].sort_values(['permno','date']).copy()

# Replace missing return with 0
_tmp_crsp['ret'] = _tmp_crsp['ret'].fillna(0)
_tmp_crsp['logret'] = np.log(1 + _tmp_crsp['ret'])

# Sum of log returns over months t-11 .. t-1 (shift by SKIP_MONTHS, then roll)
_tmp_crsp['formret'] = (_tmp_crsp.groupby('permno')['logret']
                        .transform(lambda x: x.shift(SKIP_MONTHS)
                                   .rolling(FORM_MONTHS, min_periods=FORM_MONTHS).sum()))
_tmp_crsp['cumret'] = np.exp(_tmp_crsp['formret']) - 1

# Price filter at formation (CRSP prc is negative for bid-ask midpoints)
umd = _tmp_crsp[_tmp_crsp['prc'].abs() >= MIN_PRICE].dropna(subset=['cumret']).copy()

########################################
# Formation of 10 Momentum Portfolios  #
########################################

# For each date: assign ranking 1-10 based on cumret
# 1 = losers (lowest cumret), 10 = winners (highest cumret)
umd['momr'] = umd.groupby('date')['cumret'].transform(lambda x: pd.qcut(x, 10, labels=False))
umd['momr'] = umd['momr'].astype(int) + 1

# Holding period: the month after formation
umd['form_date'] = umd['date']
umd['medate'] = umd['date'] + MonthEnd(0)
umd['hdate1'] = umd['medate'] + MonthBegin(1)
umd['hdate2'] = umd['medate'] + MonthEnd(1)
umd = umd[['permno', 'form_date','momr','hdate1','hdate2']]

# Keep formation dates from the month before TRADE_START through the month
# before TRADE_END, so every holding month lies inside the trading period
_first_form = (TRADE_START.to_period('M') - 1).to_timestamp()
_last_form = TRADE_END.to_period('M').to_timestamp()
umd = umd[(umd['form_date'] >= _first_form) & (umd['form_date'] < _last_form)]


##
## AT THIS POINT THE MOMENTUM DECILES CALCULATION IS COMPLETE
## form_date: date when momentum signal is known
## medate: first month-end after form_date
## hdate1: first month-begin after form_date
## hdate2: month-end of the holding month
##


if False:
    # join rank and return data together
    # note: this step consumes a lot of memory so takes a while
    _tmp_ret = crsp_m[['permno','date','ret']]
    port = pd.merge(_tmp_ret, umd, on=['permno'], how='inner')
    port = port[(port['hdate1']<=port['date']) & (port['date']<=port['hdate2'])]
    
    umd2 = port.sort_values(by=['date','momr','form_date','permno']).drop_duplicates()
    umd3 = umd2.groupby(['date','momr','form_date'])['ret'].mean().reset_index()
        
    # Skip first two years of the sample 
    start_yr = umd3['date'].dt.year.min()+2
    umd3 = umd3[umd3['date'].dt.year>=start_yr]
    umd3 = umd3.sort_values(by=['date','momr'])
    
    # Create one return series per MOM group every month
    ewret = umd3.groupby(['date','momr'])['ret'].mean().reset_index()
    ewstd = umd3.groupby(['date','momr'])['ret'].std().reset_index()
    ewret = ewret.rename(columns={'ret':'ewret'})
    ewstd = ewstd.rename(columns={'ret':'ewretstd'})
    ewretdat = pd.merge(ewret, ewstd, on=['date','momr'], how='inner')
    ewretdat = ewretdat.sort_values(by=['momr'])
    
    # portfolio summary
    ewretdat.groupby(['momr'])['ewret'].describe()[['count','mean', 'std']]
    
    #################################
    # Long-Short Portfolio Returns  #
    #################################
    
    # Transpose portfolio layout to have columns as portfolio returns
    ewretdat2 = ewretdat.pivot(index='date', columns='momr', values='ewret')

    # Add prefix port in front of each column
    ewretdat2 = ewretdat2.add_prefix('port')
    ewretdat2 = ewretdat2.rename(columns={'port1':'losers', 'port10':'winners'})
    ewretdat2['long_short'] = ewretdat2['winners'] - ewretdat2['losers']
    
    # Compute Long-Short Portfolio Cumulative Returns
    ewretdat3 = ewretdat2
    ewretdat3['1+losers']=1+ewretdat3['losers']
    ewretdat3['1+winners']=1+ewretdat3['winners']
    ewretdat3['1+ls'] = 1+ewretdat3['long_short']
    
    ewretdat3['cumret_winners']=ewretdat3['1+winners'].cumprod()-1
    ewretdat3['cumret_losers']=ewretdat3['1+losers'].cumprod()-1
    ewretdat3['cumret_long_short']=ewretdat3['1+ls'].cumprod()-1
    
    #################################
    # Portfolio Summary Statistics  #
    ################################# 

    # Mean 
    mom_mean = ewretdat3[['winners', 'losers', 'long_short']].mean().to_frame()
    mom_mean = mom_mean.rename(columns={0:'mean'}).reset_index()
    
    # T-Value and P-Value
    t_losers = pd.Series(stats.ttest_1samp(ewretdat3['losers'],0.0)).to_frame().T
    t_winners = pd.Series(stats.ttest_1samp(ewretdat3['winners'],0.0)).to_frame().T
    t_long_short = pd.Series(stats.ttest_1samp(ewretdat3['long_short'],0.0)).to_frame().T
    
    t_losers['momr']='losers'
    t_winners['momr']='winners'
    t_long_short['momr']='long_short'
    
    t_output =pd.concat([t_winners, t_losers, t_long_short])\
               .rename(columns={0:'t-stat', 1:'p-value'})
    
    # Combine mean, t and p
    mom_output = pd.merge(mom_mean, t_output, on=['momr'], how='inner')
