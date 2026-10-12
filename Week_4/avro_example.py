import io
from datetime import datetime, timezone

import fastavro
import pandas as pd

# --- 1. Define a schema (JSON) ----------------------------------------------
schema_v1 = fastavro.parse_schema({
    "type": "record",
    "name": "Trade",
    "fields": [
        {"name": "time",   "type": {"type": "long", "logicalType": "timestamp-micros"}},
        {"name": "symbol", "type": "string"},
        {"name": "price",  "type": "double"},
        {"name": "size",   "type": "int"},
    ],
})

trades = [
    {"time": datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc), "symbol": "AAPL", "price": 185.64, "size": 100},
    {"time": datetime(2024, 1, 2, 14, 30, 1, tzinfo=timezone.utc), "symbol": "MSFT", "price": 370.87, "size": 300},
    {"time": datetime(2024, 1, 2, 14, 30, 2, tzinfo=timezone.utc), "symbol": "AAPL", "price": 185.66, "size": 200},
]

# --- 2. Write a file: schema stored in the header, records appended ---------
with open("trades.avro", "wb") as f:
    fastavro.writer(f, schema_v1, trades)

# --- 3. Read it back: no schema needed, it is in the file --------------------
with open("trades.avro", "rb") as f:
    reader = fastavro.reader(f)
    print("schema in file:", [fld["name"] for fld in reader.writer_schema["fields"]])
    for rec in reader:
        print(rec)

# --- 4. Schema evolution: read old data with a newer schema ------------------
schema_v2 = fastavro.parse_schema({
    "type": "record",
    "name": "Trade",
    "fields": schema_v1["fields"] + [
        {"name": "venue", "type": "string", "default": "UNKNOWN"},   # new field
    ],
})
with open("trades.avro", "rb") as f:
    for rec in fastavro.reader(f, reader_schema=schema_v2):
        print(rec["symbol"], rec["venue"])          # old records get the default

# --- 5. One message at a time (what a Kafka producer/consumer does) ----------
buf = io.BytesIO()
fastavro.schemaless_writer(buf, schema_v1, trades[0])   # no header: just the record
payload = buf.getvalue()
print("message size:", len(payload), "bytes")
decoded = fastavro.schemaless_reader(io.BytesIO(payload), schema_v1)
print("decoded:", decoded)

# --- 6. Hand off to analytics: Avro -> pandas -> Parquet ---------------------
with open("trades.avro", "rb") as f:
    df = pd.DataFrame(fastavro.reader(f))
df.to_parquet("trades.parquet", index=False)
print(df.dtypes)
