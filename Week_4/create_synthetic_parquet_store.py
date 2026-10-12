import os
import pyarrow as pa
import pyarrow.parquet as pq
import wrds

WRDS_USERNAME = 'cpehlivanian'      # password is read from ~/.pgpass
START, END = '2024-01-02', '2024-12-31'
source = 'CRSP dsf_v2 (WRDS)'
dirname = 'price_volume'          # a Parquet "store" is a directory of files

def data(sym_root):
    """Daily unadjusted close price and volume for one ticker from crsp.dsf_v2."""
    conn = wrds.Connection(wrds_username=WRDS_USERNAME)
    try:
        df = conn.raw_sql(
            """
            SELECT permno, dlycaldt, dlyprc, dlyvol
            FROM crsp.dsf_v2
            WHERE ticker = %(ticker)s
              AND dlycaldt BETWEEN %(start)s AND %(end)s
            ORDER BY dlycaldt
            """,
            params={'ticker': sym_root, 'start': START, 'end': END},
            date_cols=['dlycaldt'],
        )
    finally:
        conn.close()
    if df.empty:
        raise ValueError(f"No CRSP data for {sym_root} between {START} and {END}")
    if df['permno'].nunique() > 1:     # tickers can be reused by different securities
        raise ValueError(f"{sym_root} maps to several PERMNOs: {df['permno'].unique()}")
    date = df['dlycaldt'].dt.date.to_numpy()
    price = df['dlyprc'].to_numpy(dtype='float64')
    volume = df['dlyvol'].to_numpy(dtype='float64')
    return date, price, volume

#####################################################################
# We will create and inspect a Parquet store.                       #
# HDF5 groups become directories (hive-style: symbol=WMT), each     #
# group's datasets become columns of one Parquet file, and HDF5     #
# attributes become Arrow field / schema metadata.                  #
# Parquet files are immutable: re-running w() overwrites the file   #
# rather than appending to it.                                      #
#####################################################################

# Write
def w(sym_root):

    path = os.path.join(dirname, 'equities', f"symbol={sym_root}")
    os.makedirs(path, exist_ok=True)

    date, price, volume = data(sym_root)

    schema = pa.schema([
        pa.field('date',   pa.date32()),
        pa.field('price',  pa.float64(), metadata={'units': 'USD (unadjusted)',    'source': source}),
        pa.field('volume', pa.float64(), metadata={'units': 'shares (unadjusted)', 'source': source}),
    ], metadata={'start': START, 'end': END})

    table = pa.table({'date': date, 'price': price, 'volume': volume}, schema=schema)

    pq.write_table(table,
                   os.path.join(path, 'data.parquet'),
                   row_group_size=100_000,   # Parquet analogue of HDF5 chunks
                   compression='gzip',
                   compression_level=4)

# Inspect
def i(sym_root):
    for root, _, files in sorted(os.walk(dirname)):
        for f in files:
            print(os.path.relpath(os.path.join(root, f), dirname))
    pf = pq.ParquetFile(os.path.join(dirname, 'equities', f"symbol={sym_root}", 'data.parquet'))
    print(pf.schema_arrow.field('price').metadata[b'units'].decode())
    print(pf.metadata)                 # rows, row groups, sizes -- no data read

# Peek (only reads in the first row group, price column only)
def p(sym_root, ind):
    pf = pq.ParquetFile(os.path.join(dirname, 'equities', f"symbol={sym_root}", 'data.parquet'))
    batch = next(pf.iter_batches(batch_size=ind, columns=['price']))
    first_few = batch.column('price').to_numpy()
    return first_few

# Full read
# Returns numpy ndarrays
def r(sym_root):
    t = pq.read_table(os.path.join(dirname, 'equities'),
                      columns=['price', 'volume'],
                      filters=[('symbol', '==', sym_root)])   # partition pruning
    p = t.column('price').to_numpy()
    v = t.column('volume').to_numpy()
    return p,v

if __name__ == "__main__":
    # Examples

    w("WMT")
    w("MCD")

    i("WMT")
    print(p("WMT", 5))

    p,v = r("WMT")
    p,v = r("MCD")
