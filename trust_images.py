
import csv
import hashlib
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from trusted_zone import (
    s3,
    SOURCE_BUCKET,
    ensure_bucket,
    list_files,
    exists,
    copy_to_trusted,
    species_from_key
)

PREFIX = "huggingface_cub200/images/"
REPORT_DIR = Path("quality_reports")
REPORT_DIR.mkdir(exist_ok=True)


def inspect_image(key):
    """Validate image and collect quality metrics."""
    response = s3.get_object(
        Bucket=SOURCE_BUCKET,
        Key=key
    )
    data = response["Body"].read()

    with Image.open(BytesIO(data)) as image:
        image.load()  # Force full decoding.

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

        # Hash actual decoded pixel data.
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
                    copy_to_trusted(key)
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

    print("\n========== IMAGE QUALITY ==========")
    print("Trusted:", counts["trusted"])
    print("Already trusted:", counts["skipped"])
    print("Exact duplicates:", counts["duplicates"])
    print("Invalid:", counts["invalid"])
    print("Total inspected:", len(rows))
    print("Report:", report_path)
    print("===================================")


if __name__ == "__main__":
    main()
