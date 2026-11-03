
import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

BUCKET = "landingzone"
FOLDERS = [
    "temporal_landing/",
    "persistent_landing/"
]

# Connect to MinIO using the environment configuration
s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin")
)

# Create bucket if it does not exist
try:
    s3.head_bucket(Bucket=BUCKET)
    print(f"Bucket {BUCKET} already exists")
except ClientError as error:
    if error.response["Error"]["Code"] in ("404", "NoSuchBucket"):
        s3.create_bucket(Bucket=BUCKET)
        print(f"Bucket {BUCKET} created")
    else:
        raise

# Create Landing Zone folders
for folder in FOLDERS:
    s3.put_object(Bucket=BUCKET, Key=folder)
    print(f"Folder ready: {folder}")

print("Landing Zone setup completed!")
