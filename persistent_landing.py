
import os
import re
from datetime import datetime, timezone

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

# MinIO configuration
BUCKET = "landingzone"
TEMPORAL = "temporal_landing/"
PERSISTENT = "persistent_landing/"

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


def get_source_and_format(key):
    """
    Determine the original source and data format
    based on the Temporal Landing folder.
    """

    if key.startswith(TEMPORAL + "images/"):
        return "huggingface_cub200", "images"

    if key.startswith(TEMPORAL + "text/"):
        return "huggingface_cub200", "text"

    if key.startswith(TEMPORAL + "audio/"):
        return "xeno_canto", "audio"

    return None, None


def make_destination(key, timestamp):
    """
    Create an organized Persistent Landing path.
    """

    source, data_format = get_source_and_format(key)

    if source is None:
        return None

    # Remove the Temporal Landing prefix and format.
    relative = key[len(TEMPORAL + data_format + "/"):]

    if not relative or relative.endswith("/"):
        return None

    parts = relative.split("/")

    # Keep the original species folder.
    filename = parts[-1]
    subfolders = parts[:-1]

    # Add source and timestamp to the filename.
    new_filename = f"{source}_{timestamp}_{filename}"

    destination_parts = [
        PERSISTENT.rstrip("/"),
        source,
        data_format,
        *subfolders,
        new_filename
    ]

    return "/".join(destination_parts)


def list_temporal_files():
    """Discover all files in Temporal Landing."""

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


def get_existing_destinations():
    """
    Find files already copied to Persistent Landing.
    Used to prevent duplicates on later runs.
    """

    existing = set()

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(
        Bucket=BUCKET,
        Prefix=PERSISTENT
    ):
        for obj in page.get("Contents", []):
            existing.add(obj["Key"])

    return existing


def original_key_from_destination(key):
    """
    Recover the original Temporal Landing key
    from the Persistent Landing filename.
    """

    relative = key[len(PERSISTENT):]
    parts = relative.split("/")

    if len(parts) < 4:
        return None

    source = parts[0]
    data_format = parts[1]
    remaining = parts[2:]

    filename = remaining[-1]

    # Remove source and timestamp prefix.
    pattern = rf"^{re.escape(source)}_\d{{8}}T\d{{6}}Z_"

    original_filename = re.sub(pattern, "", filename)

    if original_filename == filename:
        return None

    remaining[-1] = original_filename

    return TEMPORAL + data_format + "/" + "/".join(remaining)


def main():
    print("Starting Persistent Landing transfer...\n")

    # One timestamp for this ingestion batch.
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    existing = get_existing_destinations()

    # Identify files already transferred.
    transferred = {
        original_key_from_destination(key)
        for key in existing
    }

    copied = 0
    skipped = 0
    failed = 0

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

            # Verify that the copied object exists
            # and has the expected size.
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

    print("\n========== SUMMARY ==========")
    print("Copied:", copied)
    print("Already transferred:", skipped)
    print("Failed:", failed)
    print("=============================")

    if failed == 0:
        print("\nPersistent Landing transfer finished!")
    else:
        print("\nSome files failed. Review errors above.")


if __name__ == "__main__":
    main()
