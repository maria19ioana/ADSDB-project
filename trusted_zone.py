
import os
from pathlib import PurePosixPath

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

SOURCE_BUCKET = "formattedzone"
TARGET_BUCKET = "trustedzone"

s3 = boto3.client(
    "s3",
    endpoint_url="http://127.0.0.1:9000",
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    region_name="us-east-1"
)


def ensure_bucket():
    """Create the Trusted Zone bucket if it does not exist."""
    try:
        s3.head_bucket(Bucket=TARGET_BUCKET)
    except ClientError as error:
        code = error.response["Error"]["Code"]

        if code in ("404", "NoSuchBucket", "NotFound"):
            s3.create_bucket(Bucket=TARGET_BUCKET)
            print(f"Created bucket: {TARGET_BUCKET}")
        else:
            raise


def list_files(prefix):
    """List all files in a Formatted Zone prefix."""
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(
        Bucket=SOURCE_BUCKET,
        Prefix=prefix
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if not key.endswith("/"):
                yield key


def exists(key):
    """Check whether a file already exists in Trusted Zone."""
    try:
        s3.head_object(Bucket=TARGET_BUCKET, Key=key)
        return True
    except ClientError as error:
        code = error.response["Error"]["Code"]

        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def copy_to_trusted(key):
    """Copy a validated file without changing its contents."""
    s3.copy_object(
        Bucket=TARGET_BUCKET,
        CopySource={
            "Bucket": SOURCE_BUCKET,
            "Key": key
        },
        Key=key
    )


def species_from_key(key):
    """Extract the species folder from an object key."""
    parts = PurePosixPath(key).parts

    if len(parts) >= 4:
        return parts[2]

    return "unknown"
