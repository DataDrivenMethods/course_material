"""
create_ITCH_parquet_store.py
============================
Parse a NASDAQ TotalView-ITCH 5.0 file and store every message type as a
Parquet dataset -- the Parquet analogue of create_ITCH_HDF5_store.py.

    HDF5 store (itch.h5)                 Parquet store (itch_parquet/)
    --------------------                 -----------------------------
    one key per message type: /A, /E     one directory per message type:
                                           itch_parquet/msg_type=A/
                                           itch_parquet/msg_type=E/  ...
    store.append() adds rows to the      Parquet files cannot be appended to, so
    existing table                       each batch is written as a NEW file:
                                           part-001.parquet, part-002.parquet, ...

The parsing code below is identical to the HDF5 script. Only store_messages()
changes, and it has two steps for you to write, marked TODO (a) and TODO (b).

Usage (from a directory containing 10302019.NASDAQ_ITCH50 and message_types.xlsx):
    python create_ITCH_parquet_store.py
"""

from pathlib import Path
from time import time
from struct import unpack
from collections import namedtuple, Counter, defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


data_path = Path('.')
file_name = '10302019.NASDAQ_ITCH50'
parquet_root = data_path / 'itch_parquet'
FLUSH_EVERY = 25_000_000            # write a batch of files every this many messages


def format_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:0>2.0f}:{m:0>2.0f}:{s:0>5.2f}'

def clean_message_types(df):
    df.columns = [c.lower().strip() for c in df.columns]
    df.value = df.value.str.strip()
    df.name = (df.name
               .str.strip()
               .str.lower()
               .str.replace(' ', '_')
               .str.replace('-', '_')
               .str.replace('/', '_'))
    df.notes = df.notes.str.strip()
    df['message_type'] = df.loc[df.name == 'message_type', 'value']
    return df

def format_alpha(mtype, data):
    """Process byte strings of type alpha"""
    for col in alpha_formats.get(mtype).keys():
        if mtype != 'R' and col == 'stock':
            data = data.drop(col, axis=1)
            continue
        data[col] = data[col].str.decode("utf-8").str.strip()
        if encoding.get(col):
            data[col] = data[col].map(encoding.get(col)).astype(int)
    return data


batch_number = 0

def store_messages(m):
    """Write the messages collected so far: one new Parquet file per message type."""
    global batch_number
    batch_number += 1
    for mtype, data in m.items():
        # Convert to DataFrame (same as the HDF5 script)
        data = pd.DataFrame(data)

        # parse timestamp: nanoseconds since midnight -> timedelta64[ns]
        data.timestamp = data.timestamp.apply(int.from_bytes, byteorder='big')
        data.timestamp = pd.to_timedelta(data.timestamp)

        # apply alpha formatting
        if mtype in alpha_formats.keys():
            data = format_alpha(mtype, data)

        # Unlike HDFStore, no min_itemsize or data_columns are needed:
        # Parquet strings are variable-length and every column can be filtered on.

        # ------------------------------------------------------------------
        # TODO (a): convert the DataFrame `data` to a pyarrow Table `table`.
        #           Do not store the pandas index.
        # ------------------------------------------------------------------
        table = None
        table = pa.Table.from_pandas(data, preservce_index=False)

        # ------------------------------------------------------------------
        # TODO (b): write `table` as a new Parquet file
        #               itch_parquet/msg_type=<mtype>/part-<NNN>.parquet
        #           where <NNN> is batch_number zero-padded to 3 digits
        #           (part-001.parquet, part-002.parquet, ...).
        #           Create the directory if needed; use zstd compression.
        # ------------------------------------------------------------------
        out_dir = parquet_root / f'msg_type={mtype}'
        out_dir.mkdir(paretns=True, exist_ok=True)
        out_file = out_dif / f'part-{batch_number:03d}.parquet'
        pq.write_table(table, out_file, compression='zstd')

    return 0


event_codes = {'O': 'Start of Messages',
               'S': 'Start of System Hours',
               'Q': 'Start of Market Hours',
               'M': 'End of Market Hours',
               'E': 'End of System Hours',
               'C': 'End of Messages'
               }

encoding = {'primary_market_maker': {'Y': 1, 'N': 0},
            'printable'           : {'Y': 1, 'N': 0},
            'buy_sell_indicator'  : {'B': 1, 'S': -1},
            'cross_type'          : {'O': 0, 'C': 1, 'H': 2},
            'imbalance_direction' : {'B': 0, 'S': 1, 'N': 0, 'O': -1}
            }

formats = {
    ('integer', 2): 'H',  # int of length 2 => format string 'H'
    ('integer', 4): 'I',
    ('integer', 6): '6s', # int of length 6 => parse as string, convert later
    ('integer', 8): 'Q',
    ('alpha', 1)  : 's',
    ('alpha', 2)  : '2s',
    ('alpha', 4)  : '4s',
    ('alpha', 8)  : '8s',
    ('price_4', 4): 'I',
    ('price_8', 8): 'Q',
}

message_data = (pd.read_excel('message_types.xlsx',
                sheet_name='messages')
        .sort_values('id')
        .drop('id', axis=1))

message_types = clean_message_types(message_data)
message_labels  = (message_types.loc[:, ['message_type', 'notes']]
                   .dropna()
                   .rename(columns={'notes': 'name'}))
message_labels.name = (message_labels.name
                       .str.lower()
                       .str.replace('message', '')
                       .str.replace('.', '')
                       .str.strip().str.replace(' ', '_'))

message_types.message_type = message_types.message_type.ffill()
message_types = message_types[message_types.name != 'message_type']
message_types.value = (message_types.value
                       .str.lower()
                       .str.replace(' ', '_')
                       .str.replace('(', '')
                       .str.replace(')', ''))

message_types.to_csv('message_types.csv', index=False)
message_types = pd.read_csv('message_types.csv')

message_types.loc[:, 'formats'] = (message_types[['value', 'length']]
                                   .apply(tuple, axis=1).map(formats))

alpha_fields = message_types[message_types.value == 'alpha'].set_index('name')
alpha_msgs = alpha_fields.groupby('message_type')
alpha_formats = {k: v.to_dict() for k,v in alpha_msgs.formats}

message_fields, fstring = {}, {}
for t, message in message_types.groupby('message_type'):
    message_fields[t] = namedtuple(typename=t, field_names=message.name.tolist())
    fstring[t] = '>' + ''.join(message.formats.tolist())


if __name__ == '__main__':
    messages = defaultdict(list)
    message_count = 0
    message_type_counter = Counter()

    start = time()
    with (data_path / file_name).open('rb') as data:
        while True:

            message_size = int.from_bytes(data.read(2), byteorder='big', signed=False)

            message_type = data.read(1).decode('ascii')
            message_type_counter.update([message_type])

            record = data.read(message_size - 1)
            message = message_fields[message_type]._make(unpack(fstring[message_type], record))
            messages[message_type].append(message)

            if message_type == 'S':
                seconds = int.from_bytes(message.timestamp, byteorder='big') * 1e-9
                print('\n', event_codes.get(message.event_code.decode('ascii'), 'Error'))
                print(f'\t{format_time(seconds)}\t{message_count:12,.0f}')
                if message.event_code.decode('ascii') == 'C':
                    store_messages(messages)
                    break
            message_count += 1

            if message_count % FLUSH_EVERY == 0:
                seconds = int.from_bytes(message.timestamp, byteorder='big') * 1e-9
                d = format_time(time() - start)
                print(f'\t{format_time(seconds)}\t{message_count:12,.0f}\t{d}')
                store_messages(messages)
                messages.clear()

    print('Duration:', format_time(time() - start))
