
import os
import re
from datetime import datetime, timezone
import boto3
from dotenv import load_dotenv

load_dotenv()

# Configuration
BUCKET = "landingzone"
TEMPORAL = "temporal_landing/"
PERSISTENT = "persistent_landing/"

# Connect to MinIO
s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin")
)

# Identify the source and format based on the Temporal Landing path
def get_source_and_format(key):

    if key.startswith(TEMPORAL + "images/"):
        return "huggingface_cub200", "images"

    if key.startswith(TEMPORAL + "text/"):
        return "huggingface_cub200", "text"

    if key.startswith(TEMPORAL + "audio/"):
        return "xeno_canto", "audio"

    return None, None

# Create the destination path
def make_destination(key, timestamp):

    source, data_format = get_source_and_format(key)

    if source is None:
        return None

    relative = key[len(TEMPORAL + data_format + "/"):]

    if not relative or relative.endswith("/"):
        return None

    parts = relative.split("/")

    filename = parts[-1]
    subfolders = parts[:-1]

    # Add the source and timestamp to the original filename
    new_filename = f"{source}_{timestamp}_{filename}"

    destination_parts = [
        PERSISTENT.rstrip("/"),
        source,
        data_format,
        *subfolders,
        new_filename
    ]

    return "/".join(destination_parts)

# List all files from Temporal Landing, ignoring folder markers
def list_temporal_files():

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(
        Bucket=BUCKET,
        Prefix=TEMPORAL
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if key.endswith("/"):
                continue

            yield obj

# Get the files already stored in Persistent Landing
def get_existing_destinations():

    existing = set()

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(
        Bucket=BUCKET,
        Prefix=PERSISTENT
    ):
        for obj in page.get("Contents", []):
            existing.add(obj["Key"])

    return existing


# Find the original Temporal Landing path
def original_key_from_destination(key):

    relative = key[len(PERSISTENT):]
    parts = relative.split("/")

    if len(parts) < 4:
        return None

    source = parts[0]
    data_format = parts[1]
    remaining = parts[2:]

    filename = remaining[-1]

    # Remove the source and timestamp to recover the original filename
    pattern = rf"^{re.escape(source)}_\d{{8}}T\d{{6}}Z_"

    original_filename = re.sub(pattern, "", filename)

    if original_filename == filename:
        return None

    remaining[-1] = original_filename

    return TEMPORAL + data_format + "/" + "/".join(remaining)


def main():
    print("Starting Persistent Landing transfer...\n")

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    existing = get_existing_destinations()

    # Recover original paths to avoid copying files already transferred
    transferred = {
        original_key_from_destination(key)
        for key in existing
    }

    copied = 0
    skipped = 0
    failed = 0

    # Copy files that were not transferred before
    for obj in list_temporal_files():
        source_key = obj["Key"]

        destination_key = make_destination(
            source_key, timestamp
        )

        if destination_key is None:
            continue

        if source_key in transferred:
            print("SKIP:", source_key)
            skipped += 1
            continue

        try:
            s3.copy_object(
                Bucket=BUCKET,
                CopySource={
                    "Bucket": BUCKET,
                    "Key": source_key
                },
                Key=destination_key,
                MetadataDirective="COPY"
            )

            result = s3.head_object(
                Bucket=BUCKET,
                Key=destination_key
            )

            if result["ContentLength"] != obj["Size"]:
                raise RuntimeError("Copied file size mismatch")

            print("COPIED:", destination_key)
            copied += 1

        except Exception as error:
            print("ERROR:", source_key, error)
            failed += 1

    print("Copied:", copied)
    print("Already transferred:", skipped)
    print("Failed:", failed)

    if failed == 0:
        print("\nPersistent Landing transfer finished!")
    else:
        raise RuntimeError(
            f"Persistent Landing transfer failed for {failed} files"
        )


if __name__ == "__main__":
    main()
