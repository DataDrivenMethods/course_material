import os
import h5py
import wrds

WRDS_USERNAME = 'cpehlivanian'      # password is read from ~/.pgpass
START, END = '2024-01-02', '2024-12-31'
source = 'CRSP dsf_v2 (WRDS)'
filename = 'price_volume.h5'

def data(sym_root):
    """Daily unadjusted close price and volume for one ticker from crsp.dsf_v2."""
    conn = wrds.Connection(wrds_username=WRDS_USERNAME)
    try:
        df = conn.raw_sql(
            """
            SELECT permno, yyyymmdd, dlyprc, dlyvol
            FROM crsp.dsf_v2
            WHERE ticker = %(ticker)s
              AND dlycaldt BETWEEN %(start)s AND %(end)s
            ORDER BY dlycaldt
            """,
            params={'ticker': sym_root, 'start': START, 'end': END},
        )
    finally:
        conn.close()
    if df.empty:
        raise ValueError(f"No CRSP data for {sym_root} between {START} and {END}")
    if df['permno'].nunique() > 1:     # tickers can be reused by different securities
        raise ValueError(f"{sym_root} maps to several PERMNOs: {df['permno'].unique()}")
    date = df['yyyymmdd'].to_numpy(dtype='int64')
    price = df['dlyprc'].to_numpy(dtype='float64')
    volume = df['dlyvol'].to_numpy(dtype='float64')
    return date, price, volume

#####################################################################
# We will create and inspect an hdf5 store.                         #
# Note that pd.HDFStore is not appropriate to use for file access   #
# once the store has been written using h5py methods.               #
#####################################################################

# Write
def w(sym_root):

    mode = 'a' if os.path.exists(filename) else 'w'

    date, price, volume = data(sym_root)

    with h5py.File(filename, mode) as fh:

        fh.attrs['start'] = START
        fh.attrs['end'] = END

        path = f"equities/{sym_root}"
        if path in fh:                # re-running replaces the symbol's data
            del fh[path]
        grp = fh.create_group(path)

        ds_date = grp.create_dataset('date', data=date)
        ds_date.attrs['format'] = 'YYYYMMDD'

        ds_price = grp.create_dataset('price',
                                      data=price,
                                      chunks=True,
                                      compression='gzip',
                                      compression_opts=4)
        ds_price.attrs['units'] = 'USD (unadjusted)'
        ds_price.attrs['source'] = source

        ds_volume = grp.create_dataset('volume',
                                       data=volume,
                                       chunks=True,
                                       compression='gzip',
                                       compression_opts=4)
        ds_volume.attrs['units'] = 'shares (unadjusted)'
        ds_volume.attrs['source'] = source


# Inspect
def i(sym_root):
    with h5py.File(filename, 'r') as fh:
        fh.visit(print)
        print(fh[f"equities/{sym_root}/price"].attrs['units'])

# Peek (only reads in necessary chunk)
def p(sym_root, ind):
    with h5py.File(filename, 'r') as fh:
        first_few = fh[f"equities/{sym_root}/price"][0:ind]
    return first_few

# Full read
# Returns numpy ndarrays
def r(sym_root):
    with h5py.File(filename, 'r') as fh:
        p = fh[f"equities/{sym_root}/price"][:]
        v = fh[f"equities/{sym_root}/volume"][:]
    return p,v

if __name__ == "__main__":
    # Examples`

    w("WMT")
    w("MCD")

    p_WMT,v_WMT = r("WMT")
    p_MCD,v_MCD = r("MCD")
