import wrds
import numpy as np
import pandas as pd

wrds_username = 'cpehlivanian'

conn = wrds.Connection(wrds_username=wrds_username)

# SP500 monthly constituents:

q1 = """
select a.*, b.date
from crsp.msp500list as a,
crsp.msf as b
where a.permno=b.permno
and b.date >= a.start and b.date <= a.ending
order by date asc;
"""

sp500 = conn.raw_sql(q1, date_cols=['start', 'ending', 'date'])

# Enrich with reference data

q2 ="""
select comnam, ncusip, namedt, nameendt,
permno, shrcd, exchcd, hsiccd, ticker
from crsp.msenames
"""

mse = conn.raw_sql(q2, date_cols=['namedt', 'nameendt'])
mse['nameendt'] = mse['nameendt'].fillna(pd.Timestamp.today().normalize())
sp500_full = pd.merge(sp500, mse, how='left', on='permno')
sp500_full = sp500_full.loc[(sp500_full['date'] >= sp500_full['namedt']) \
                            & (sp500_full['date'] <= sp500_full['nameendt'])]

# Merge with Compustat identifiers
q3 = """
select gvkey, liid as iid, lpermno as permno, linktype, linkprim,
linkdt, linkenddt
from crsp.ccmxpf_linktable
where substr(linktype, 1, 1) ='L'
and (linkprim = 'C' or linkprim = 'P')
"""

ccm = conn.raw_sql(q3, date_cols=['linkdt', 'linkenddt'])

# To merge:

# 1. make sure linkenddt has valid dates
ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.to_datetime('today'))
# 2. naive merge
sp500ccm = pd.merge(sp500_full, ccm, how='left', on=['permno'])
# 3. filter on valid linkdts
sp500ccm = sp500ccm.loc[(sp500ccm['date'] >= sp500ccm['linkdt']) \
                        & (sp500ccm['date'] <= sp500ccm['linkenddt'])]
