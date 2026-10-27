import os
import re
import zipfile
from pathlib import Path
from dotenv import load_dotenv
import boto3
from huggingface_hub import hf_hub_download

load_dotenv()

# Configuration
HF_REPO = "KAKIZHOU/CUB-200"

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

BUCKET = "landingzone"
DESTINATION = "temporal_landing"

START_BIRD = 1
END_BIRD = 100
MAX_FILES_PER_SPECIES = 5  

DOWNLOAD_DIR = Path("hf_downloads")

# Connect to MinIO
s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)

try:
    s3.head_bucket(Bucket=BUCKET)
    print("Connected to MinIO successfully!")
except Exception as error:
    raise RuntimeError(
        "Cannot access the landingzone bucket "
        "Check Docker, MinIO, and your credentials"
    ) from error


# Find bird species in the ZIP archive
def get_species_folder(zip_path):
    parts = Path(zip_path).parts

    for index, part in enumerate(parts):
        match = re.match(r"^(\d{3})\.(.+)$", part)

        if match:
            bird_number = int(match.group(1))

            if START_BIRD <= bird_number <= END_BIRD:
                return index

    return None

# Download ZIP and upload selected files to MinIO
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
    species_file_count = {}  

    with zipfile.ZipFile(archive_path, "r") as archive:

        for member in archive.infolist():

            if member.is_dir():
                continue

            species_index = get_species_folder(member.filename)

            if species_index is None:
                continue

            parts = Path(member.filename).parts
            species_name = parts[species_index]

            current_count = species_file_count.get(species_name, 0)
            if current_count >= MAX_FILES_PER_SPECIES:
                continue

            relative_path = "/".join(parts[species_index:])
            species_found.add(species_name)

            object_key = f"{DESTINATION}/{destination_folder}/{relative_path}"

            print(f"Uploading: {object_key} ({current_count + 1}/{MAX_FILES_PER_SPECIES})")

            with archive.open(member) as file_data:
                s3.upload_fileobj(
                    Fileobj=file_data,
                    Bucket=BUCKET,
                    Key=object_key
                )

            uploaded += 1
            species_file_count[species_name] = current_count + 1

    print(f"\nCompleted {archive_name}")
    print(f"Total files uploaded: {uploaded}")
    print(f"Species found: {len(species_found)}")

    if uploaded == 0:
        raise RuntimeError(
            f"No matching files found in {archive_name} "
            "Check the archive's internal structure"
        )

    return uploaded, species_found



image_count, image_species = process_archive("images.zip","images")
text_count, text_species = process_archive("text.zip","text")


print(f"\nImage files uploaded: {image_count}")
print(f"Text files uploaded: {text_count}")
print(f"Image species found: {len(image_species)}")
print(f"Text species found: {len(text_species)}")

expected_count = END_BIRD - START_BIRD + 1

if (
    len(image_species) == expected_count
    and len(text_species) == expected_count
):
    print(f"SUCCESS: All {expected_count} bird species found in both archives!")
else:
    print("WARNING: Some species folders were not found")
    print("Image species:", sorted(image_species))
    print("Text species:", sorted(text_species))

print(f"\nMinIO destination: {BUCKET}/{DESTINATION}/")
print("Upload process finished!")