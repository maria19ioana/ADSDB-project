import json
import os
import re
import time

import boto3
import requests
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# Your existing MinIO setup
BUCKET = "landingzone"
IMAGES = "temporal_landing/images/"
AUDIO = "temporal_landing/audio/"
MAX_AUDIO = 5

API_KEY = os.getenv("XENO_CANTO_API_KEY")

if not API_KEY:
    raise RuntimeError("Add XENO_CANTO_API_KEY to your .env file")

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

http = requests.Session()



def get_species():
    """Read existing bird folders from MinIO."""
    folders = []

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(
        Bucket=BUCKET,
        Prefix=IMAGES,
        Delimiter="/"
    ):
        for item in page.get("CommonPrefixes", []):
            folder = item["Prefix"].rstrip("/").split("/")[-1]

            match = re.match(r"^\d+[._](.+)$", folder)

            if match:
                name = match.group(1).replace("_", " ")
                folders.append((folder, name))

    return sorted(folders)



def normalize(name):
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())



def get_recordings(species_name):
    """Find up to five recordings using name variations."""

    url = "https://xeno-canto.org/api/3/recordings"
    candidates = []

    words = species_name.split()

    # Generate possible name variations.
    search_names = [species_name]

    # Example: Black billed Cuckoo -> Black-billed Cuckoo
    if len(words) >= 3:
        search_names.append(
            "-".join(words[:2]) + " " + " ".join(words[2:])
        )

    # Example: Scissor tailed Flycatcher
    # -> Scissor-tailed Flycatcher
    search_names = list(dict.fromkeys(search_names))

    for search_name in search_names:
        print("  Trying Xeno-canto name:", search_name)

        for page in range(1, 4):
            response = http.get(
                url,
                params={
                    "query": f'grp:birds en:"{search_name}"',
                    "key": API_KEY,
                    "page": page
                },
                timeout=60
            )

            response.raise_for_status()
            data = response.json()

            for record in data.get("recordings", []):
                if normalize(record.get("en")) != normalize(species_name):
                    continue

                if record.get("q") not in ("A", "B"):
                    continue

                sound = (record.get("type") or "").lower()

                if not ("song" in sound or "call" in sound):
                    continue

                if record.get("file") and record.get("id"):
                    candidates.append(record)

            if page >= int(data.get("numPages", 1)):
                break

            time.sleep(1)

        if candidates:
            break

    scientific_names = {
        (
            normalize(r.get("gen")),
            normalize(r.get("sp"))
        )
        for r in candidates
    }

    if len(scientific_names) != 1:
        print("  Skipped: ambiguous or missing species match")
        return []

    candidates.sort(
        key=lambda r: 0 if r["q"] == "A" else 1
    )

    selected = []
    used = set()

    # Include one song and one call if available.
    for kind in ("song", "call"):
        for record in candidates:
            rid = str(record["id"])

            if kind in record["type"].lower() and rid not in used:
                selected.append(record)
                used.add(rid)
                break

    # Fill remaining slots with quality A/B recordings.
    for record in candidates:
        if len(selected) >= MAX_AUDIO:
            break

        rid = str(record["id"])

        if rid not in used:
            selected.append(record)
            used.add(rid)

    return selected[:MAX_AUDIO]



def exists(key):
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError as error:
        if error.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
            return False
        raise


def upload_audio(folder, record):
    """Stream audio directly from Xeno-canto into MinIO."""

    rid = str(record["id"])
    file_url = record["file"]

    if file_url.startswith("//"):
        file_url = "https:" + file_url

    if not file_url.startswith("https://"):
        raise ValueError("Unexpected audio URL")

    original_name = record.get("file-name", "")
    extension = os.path.splitext(original_name)[1].lower()

    if extension not in (".mp3", ".wav", ".flac", ".ogg"):
        extension = ".mp3"

    key = f"{AUDIO}{folder}/XC{rid}{extension}"
    metadata_key = f"{AUDIO}{folder}/XC{rid}.json"

    if not exists(key):
        with http.get(
            file_url,
            params={"key": API_KEY},
            stream=True,
            timeout=120
        ) as response:
            response.raise_for_status()
            response.raw.decode_content = True

            # Stream directly into MinIO.
            s3.upload_fileobj(response.raw, BUCKET, key)

        print("  Uploaded:", key)
    else:
        print("  Already exists:", key)

    # Preserve attribution and license information.
    metadata = {
        "xeno_canto_id": rid,
        "common_name": record.get("en"),
        "scientific_name": (
            f"{record.get('gen', '')} {record.get('sp', '')}"
        ).strip(),
        "type": record.get("type"),
        "quality": record.get("q"),
        "recordist": record.get("rec"),
        "license": record.get("lic"),
        "source_url": f"https://xeno-canto.org/{rid}",
        "original_filename": original_name,
        "minio_audio_key": key
    }

    s3.put_object(
        Bucket=BUCKET,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2).encode("utf-8"),
        ContentType="application/json"
    )


def main():
    s3.head_bucket(Bucket=BUCKET)

    species = get_species()
    print(f"Found {len(species)} species in MinIO")

    total = 0

    for folder, name in species:
        print(f"\nSearching: {name}")

        try:
            recordings = get_recordings(name)
            print(f"  Found {len(recordings)} suitable recordings")

            for record in recordings:
                try:
                    upload_audio(folder, record)
                    total += 1
                except Exception as error:
                    print("  Upload failed:", error)

        except Exception as error:
            print("  Search failed:", error)

        time.sleep(2)

    print(f"\nFinished! {total} recordings processed.")


if __name__ == "__main__":
    main()
