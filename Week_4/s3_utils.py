import boto3
import pandas as pd
import json
import sys
import threading
import time
from pathlib import Path
from io import StringIO

def read_df_from_s3(filename, delimiter='\t'):
    """
    Read a CSV file from S3 using credentials from s3_credentials.json

    Args:
        filename (str): Name of the file in S3 (without prefix)
        delimiter (str): CSV delimiter, defaults to tab

    Returns:
        pandas.DataFrame: The loaded data
    """
    # Load credentials
    creds_file = Path(__file__).parent / "s3_credentials.json"

    try:
        with open(creds_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        raise Exception(f"Error loading credentials: {e}")

    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=config['access_key'],
        aws_secret_access_key=config['secret_key'],
        region_name=config.get('region', 'us-east-1')
    )

    # Construct S3 key
    prefix = config.get('prefix', '')
    s3_key = f"{prefix}/{filename}" if prefix else filename
    bucket = config['bucket']

    try:
        # Get the object from S3
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)

        # Read the content and convert to pandas DataFrame
        content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(content), delimiter=delimiter)

        print(f"✅ Successfully loaded {filename} from S3 ({len(df)} rows)")
        return df

    except Exception as e:
        raise Exception(f"Error reading {filename} from S3: {e}")


def read_binary_from_s3(filename):
    """
    Read a binary file from S3 using credentials from s3_credentials.json

    Args:
        filename (str): Name of the file in S3 (without prefix)

    Returns:
        bytes: The raw file content
    """
    creds_file = Path(__file__).parent / "s3_credentials.json"

    try:
        with open(creds_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        raise Exception(f"Error loading credentials: {e}")

    s3_client = boto3.client(
        's3',
        aws_access_key_id=config['access_key'],
        aws_secret_access_key=config['secret_key'],
        region_name=config.get('region', 'us-east-1')
    )

    prefix = config.get('prefix', '')
    s3_key = f"{prefix}/{filename}" if prefix else filename
    bucket = config['bucket']

    try:
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        data = response['Body'].read()
        print(f"Successfully loaded {filename} from S3 ({len(data)} bytes)")
        return data
    except Exception as e:
        raise Exception(f"Error reading {filename} from S3: {e}")


def download_file_from_s3(filename, local_path=None):
    """
    Download a file from S3 to disk using credentials from s3_credentials.json.
    The file is streamed to disk in parallel parts, so it works for files
    far larger than memory (e.g. the 16 GB itch.h5).

    Args:
        filename (str): Name of the file in S3 (without prefix)
        local_path (str or Path): Where to save it; defaults to ./<filename>

    Returns:
        Path: The local path of the downloaded file
    """
    creds_file = Path(__file__).parent / "s3_credentials.json"

    try:
        with open(creds_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        raise Exception(f"Error loading credentials: {e}")

    s3_client = boto3.client(
        's3',
        aws_access_key_id=config['access_key'],
        aws_secret_access_key=config['secret_key'],
        region_name=config.get('region', 'us-east-1')
    )

    prefix = config.get('prefix', '')
    s3_key = f"{prefix}/{filename}" if prefix else filename
    bucket = config['bucket']
    local_path = Path(local_path) if local_path else Path(filename)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        size = s3_client.head_object(Bucket=bucket, Key=s3_key)['ContentLength']
        seen = 0
        lock = threading.Lock()          # boto3 reports progress from several threads
        start = time.monotonic()
        last = [0.0]

        def progress(nbytes):
            nonlocal seen
            with lock:
                seen += nbytes
                now = time.monotonic()
                if now - last[0] < 0.5 and seen < size:
                    return
                last[0] = now
                rate = seen / (now - start) if now > start else 0.0
                eta = (size - seen) / rate if rate > 0 else 0.0
                sys.stdout.write(f"\r  {seen / 1024**2:,.0f} / {size / 1024**2:,.0f} MB"
                                 f"  ({100 * seen / size:5.1f}%)  {rate / 1024**2:6.2f} MB/s"
                                 f"  ETA {eta / 60:5.1f} min   ")
                sys.stdout.flush()

        s3_client.download_file(bucket, s3_key, str(local_path), Callback=progress)
        print()
        print(f"Downloaded s3://{bucket}/{s3_key} -> {local_path}")
        return local_path

    except Exception as e:
        raise Exception(f"Error downloading {filename} from S3: {e}")
