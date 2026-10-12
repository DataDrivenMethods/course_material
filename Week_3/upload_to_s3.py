#!/usr/bin/env python3
"""
Script to upload data files to S3.
Usage: python upload_to_s3.py <absolute_path_to_file>
"""

import boto3
import json
import os
import sys
from pathlib import Path

def upload_file(file_path):
    """Upload a single file to S3."""

    # Load credentials
    creds_file = Path(__file__).parent / "s3_credentials.json"
    if not creds_file.exists():
        print(f"Error: Credentials file not found: {creds_file}")
        print("Please create s3_credentials.json with your AWS credentials")
        return False

    try:
        with open(creds_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading credentials: {e}")
        return False

    # Initialize S3 client
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=config['access_key'],
            aws_secret_access_key=config['secret_key'],
            region_name=config.get('region', 'us-east-1')
        )
    except Exception as e:
        print(f"Error initializing S3 client: {e}")
        return False

    # Validate file path
    local_path = Path(file_path)
    if not local_path.exists():
        print(f"Error: File not found: {local_path}")
        return False

    if not local_path.is_file():
        print(f"Error: Path is not a file: {local_path}")
        return False

    bucket = config['bucket']
    prefix = config.get('prefix', '')
    filename = local_path.name

    # Construct S3 key
    s3_key = f"{prefix}/{filename}" if prefix else filename

    print(f"Uploading to S3 bucket: {bucket}")
    if prefix:
        print(f"Using prefix: {prefix}")

    try:
        # Upload file
        s3_client.upload_file(str(local_path), bucket, s3_key)
        print(f"✅ Uploaded: {filename} -> s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        print(f"❌ Failed to upload {filename}: {e}")
        return False

def verify_upload():
    """Verify that uploaded files can be accessed."""

    creds_file = Path(__file__).parent / "s3_credentials.json"
    if not creds_file.exists():
        print("Cannot verify - credentials file not found")
        return False

    try:
        with open(creds_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading credentials: {e}")
        return False

    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=config['access_key'],
            aws_secret_access_key=config['secret_key'],
            region_name=config.get('region', 'us-east-1')
        )
    except Exception as e:
        print(f"Error initializing S3 client: {e}")
        return False

    bucket = config['bucket']
    prefix = config.get('prefix', '')

    print(f"\nListing files in S3 bucket: {bucket}")
    if prefix:
        print(f"With prefix: {prefix}")

    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix if prefix else ''
        )

        if 'Contents' in response:
            print("\nFiles found:")
            for obj in response['Contents']:
                print(f"✅ {obj['Key']} ({obj['Size']} bytes)")
        else:
            print("No files found in bucket")

    except Exception as e:
        print(f"❌ Error listing bucket contents: {e}")
        return False

    return True

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python upload_to_s3.py <absolute_path_to_file>")
        print("       python upload_to_s3.py verify")
        sys.exit(1)

    if sys.argv[1] == 'verify':
        verify_upload()
    else:
        file_path = sys.argv[1]
        print("S3 File Upload Script")
        print("=" * 20)

        if upload_file(file_path):
            print(f"\n✅ File uploaded successfully!")
        else:
            print(f"\n❌ Upload failed. Please check the errors above.")
            sys.exit(1)

if __name__ == '__main__':
    main()