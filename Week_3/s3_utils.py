import boto3
import pandas as pd
import json
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
