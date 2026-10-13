
import os
import re
import zipfile
from pathlib import Path

import boto3
from botocore.config import Config
from huggingface_hub import hf_hub_download

# =====================================
# 1. CONFIGURATION
# =====================================

HF_REPO = "KAKIZHOU/CUB-200"

MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")

BUCKET = "landingzone"
DESTINATION = "temporal_landing"

START_BIRD = 31
END_BIRD = 46

DOWNLOAD_DIR = Path("hf_downloads")

# =====================================
# 2. CONNECT TO MINIO
# =====================================

s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    region_name="us-east-1",
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"}
    )
)

try:
    s3.head_bucket(Bucket=BUCKET)
    print("Connected to MinIO successfully!")
except Exception as error:
    raise RuntimeError(
        "Cannot access the landingzone bucket. "
        "Check Docker, MinIO, and your credentials."
    ) from error


# =====================================
# 3. DOWNLOAD AND UPLOAD SUBSET
# =====================================

def get_species_folder(zip_path):
    """
    Find the species folder in a ZIP member path.
    Example: images/031.Black_billed_Cuckoo/file.jpg
    """
    parts = Path(zip_path).parts

    for index, part in enumerate(parts):
        match = re.match(r"^(\d{3})\.(.+)$", part)

        if match:
            bird_number = int(match.group(1))

            if START_BIRD <= bird_number <= END_BIRD:
                return index

    return None


def process_archive(archive_name, destination_folder):

    print(f"\nDownloading {archive_name}...")

    archive_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=archive_name,
        repo_type="dataset",
        local_dir=str(DOWNLOAD_DIR)
    )

    print(f"Downloaded: {archive_path}")

    uploaded = 0
    species_found = set()

    with zipfile.ZipFile(archive_path, "r") as archive:

        for member in archive.infolist():

            if member.is_dir():
                continue

            species_index = get_species_folder(
                member.filename
            )

            if species_index is None:
                continue

            parts = Path(member.filename).parts

            # Keep the original species folder and file names
            relative_path = "/".join(parts[species_index:])

            species_found.add(parts[species_index])

            object_key = (
                f"{DESTINATION}/"
                f"{destination_folder}/"
                f"{relative_path}"
            )

            print(f"Uploading: {object_key}")

            # Stream directly from ZIP to MinIO.
            # No need to extract all files to disk.
            with archive.open(member) as file_data:

                s3.upload_fileobj(
                    Fileobj=file_data,
                    Bucket=BUCKET,
                    Key=object_key
                )

            uploaded += 1

    print(f"\nCompleted {archive_name}")
    print(f"Files uploaded: {uploaded}")
    print(f"Species found: {len(species_found)}")

    if uploaded == 0:
        raise RuntimeError(
            f"No matching files found in {archive_name}. "
            "Check the archive's internal structure."
        )

    return uploaded, species_found


# =====================================
# 4. PROCESS IMAGES AND TEXT
# =====================================

image_count, image_species = process_archive(
    "images.zip",
    "images"
)

text_count, text_species = process_archive(
    "text.zip",
    "text"
)


# =====================================
# 5. VERIFY RESULTS
# =====================================

print("\n========== SUMMARY ==========")

print(f"Image files uploaded: {image_count}")
print(f"Text files uploaded: {text_count}")

print(f"Image species found: {len(image_species)}")
print(f"Text species found: {len(text_species)}")

expected_count = END_BIRD - START_BIRD + 1

if (
    len(image_species) == expected_count
    and len(text_species) == expected_count
):
    print("SUCCESS: All 16 bird species found in both archives!")
else:
    print("WARNING: Some species folders were not found.")
    print("Image species:", sorted(image_species))
    print("Text species:", sorted(text_species))

print("\nMinIO destination:")
print(f"{BUCKET}/{DESTINATION}/")

print("\nUpload process finished!")
