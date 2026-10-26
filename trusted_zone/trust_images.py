import csv
import hashlib
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from trusted_zone.trusted_zone import (
    s3,
    SOURCE_BUCKET,
    TARGET_BUCKET,
    ensure_bucket,
    list_files,
    exists,
    species_from_key
)

PREFIX = "huggingface_cub200/images/"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = PROJECT_ROOT / "quality_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_SIZE = (256, 256)


def inspect_image(key):
    """Validate image and collect quality metrics"""
    response = s3.get_object(
        Bucket=SOURCE_BUCKET,
        Key=key
    )
    data = response["Body"].read()

    with Image.open(BytesIO(data)) as image:
        image.load()  

        if image.format != "JPEG":
            raise ValueError(
                f"Expected JPEG, found {image.format}"
            )

        if image.mode != "RGB":
            raise ValueError(
                f"Expected RGB, found {image.mode}"
            )

        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError("Invalid image dimensions")

        # hash the data to check for duplicates
        digest = hashlib.sha256()
        digest.update(width.to_bytes(8, "big"))
        digest.update(height.to_bytes(8, "big"))
        digest.update(image.tobytes())

        return {
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 4),
            "size_bytes": len(data),
            "pixel_hash": digest.hexdigest()
        }


def main():
    ensure_bucket()

    seen_hashes = {}
    rows = []
    counts = Counter()

    for key in list_files(PREFIX):
        if not key.lower().endswith((".jpg", ".jpeg")):
            continue

        species = species_from_key(key)

        row = {
            "key": key,
            "species": species,
            "status": "",
            "reason": "",
            "width": "",
            "height": "",
            "aspect_ratio": "",
            "size_bytes": "",
            "pixel_hash": ""
        }

        try:
            metrics = inspect_image(key)
            row.update(metrics)

            pixel_hash = metrics["pixel_hash"]

            if pixel_hash in seen_hashes:
                row["status"] = "excluded"
                row["reason"] = (
                    "Exact duplicate of "
                    + seen_hashes[pixel_hash]
                )
                counts["duplicates"] += 1

            else:
                seen_hashes[pixel_hash] = key

                if exists(key):
                    row["status"] = "already_trusted"
                    counts["skipped"] += 1
                else:
                    # resize and pad the image to the target size
                    response = s3.get_object(Bucket=SOURCE_BUCKET, Key=key)
                    raw_data = response["Body"].read()
                    
                    with Image.open(BytesIO(raw_data)) as img:
                        clean_img = ImageOps.pad(img, TARGET_SIZE, color=(0, 0, 0))
                        out_buffer = BytesIO()
                        clean_img.save(out_buffer, format='JPEG', quality=95)
                        
                    s3.put_object(
                        Bucket=TARGET_BUCKET,
                        Key=key,
                        Body=out_buffer.getvalue(),
                        ContentType="image/jpeg",
                        Metadata={"original-key": key}
                    )
                    
                    row["status"] = "trusted"
                    counts["trusted"] += 1

            print(row["status"].upper(), key)

        except Exception as error:
            row["status"] = "excluded"
            row["reason"] = str(error)
            counts["invalid"] += 1
            print("INVALID:", key, error)

        rows.append(row)

    report_path = REPORT_DIR / "image_quality.csv"

    fields = [
        "key", "species", "status", "reason",
        "width", "height", "aspect_ratio",
        "size_bytes", "pixel_hash"
    ]

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Trusted:", counts["trusted"])
    print("Already trusted:", counts["skipped"])
    print("Exact duplicates:", counts["duplicates"])
    print("Invalid:", counts["invalid"])
    print("Total inspected:", len(rows))
    print("Report:", report_path)


if __name__ == "__main__":
    main()
