import os

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

SOURCE_BUCKET = "landingzone"
TARGET_BUCKET = "formattedzone"
PERSISTENT_PREFIX = "persistent_landing/"

s3 = boto3.client(
    "s3",
    endpoint_url="http://127.0.0.1:9000",
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"}
    )
)


def ensure_bucket():
    """Create the Formatted Zone bucket if needed."""
    try:
        s3.head_bucket(Bucket=TARGET_BUCKET)
    except ClientError as error:
        status = error.response["ResponseMetadata"]["HTTPStatusCode"]
        if status == 404:
            s3.create_bucket(Bucket=TARGET_BUCKET)
        else:
            raise


def list_files(prefix):
    """Discover files automatically in Persistent Landing."""
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(
        Bucket=SOURCE_BUCKET,
        Prefix=prefix
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):
                yield key


def target_key(source_key, extension):
    """Preserve folders and filenames, changing only extension."""
    relative = source_key[len(PERSISTENT_PREFIX):]
    stem = relative.rsplit(".", 1)[0]
    return stem + extension


def exists(key):
    try:
        s3.head_object(Bucket=TARGET_BUCKET, Key=key)
        return True
    except ClientError as error:
        status = error.response["ResponseMetadata"]["HTTPStatusCode"]
        if status == 404:
            return False
        raise
