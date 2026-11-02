
import csv
import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path

from trusted_zone import (
    s3,
    SOURCE_BUCKET,
    TARGET_BUCKET,
    ensure_bucket,
    list_files,
    species_from_key,
    exists
)

PREFIX = "huggingface_cub200/text/"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = PROJECT_ROOT / "quality_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = REPORT_DIR / "text_quality.csv"

MIN_WORDS = 5


def clean_text(text):
    """Apply conservative, model-independent text cleaning"""

    # normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # normalize unicode to NFC form
    text = unicodedata.normalize("NFC", text)

    # remove control characters except for newlines and tabs
    text = "".join(
        char for char in text
        if char in ("\n", "\t")
        or unicodedata.category(char) != "Cc"
    )

    # replace runs of whitespace with a single space
    text = re.sub(r"[^\S\n]+", " ", text)

    # remove whitespaces from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # check for multiple consecutive newlines and replace with two newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def inspect_text(key):
    """Download, decode, and clean a text file."""

    response = s3.get_object(
        Bucket=SOURCE_BUCKET,
        Key=key
    )
    raw = response["Body"].read()

    original = raw.decode("utf-8-sig", errors="strict")
    cleaned = clean_text(original)

    original_words = len(original.split())
    cleaned_words = len(cleaned.split())

    return {
        "original": original,
        "cleaned": cleaned,
        "original_chars": len(original),
        "cleaned_chars": len(cleaned),
        "original_words": original_words,
        "cleaned_words": cleaned_words,
        "changed": original != cleaned
    }


def main():
    ensure_bucket()

    counts = Counter()
    rows = []

    # deduplication check
    seen_hashes = {}

    keys = sorted(
        key for key in list_files(PREFIX)
        if key.lower().endswith(".txt")
    )

    for key in keys:
        species = species_from_key(key)

        row = {
            "key": key,
            "species": species,
            "status": "",
            "reason": "",
            "original_chars": "",
            "cleaned_chars": "",
            "original_words": "",
            "cleaned_words": "",
            "changed": "",
            "short_text": ""
        }

        try:
            info = inspect_text(key)

            for field in (
                "original_chars",
                "cleaned_chars",
                "original_words",
                "cleaned_words",
                "changed"
            ):
                row[field] = info[field]

            cleaned = info["cleaned"]

            if not cleaned:
                row["status"] = "excluded"
                row["reason"] = "Empty after cleaning"
                counts["empty"] += 1

            else:
                digest = hashlib.sha256(
                    cleaned.encode("utf-8")
                ).hexdigest()

                duplicate_id = (species, digest)

                if duplicate_id in seen_hashes:
                    row["status"] = "excluded"
                    row["reason"] = (
                        "Duplicate of "
                        + seen_hashes[duplicate_id]
                    )
                    counts["duplicates"] += 1

                else:
                    seen_hashes[duplicate_id] = key

                    row["short_text"] = (
                        info["cleaned_words"] < MIN_WORDS
                    )

                    if row["short_text"]:
                        counts["short"] += 1

                    # keep clean text in Trusted Zone
                    if exists(key):
                        row["status"] = "already_trusted"
                        counts["skipped"] += 1
                    else:
                        s3.put_object(
                            Bucket=TARGET_BUCKET,
                            Key=key,
                            Body=cleaned.encode("utf-8"),
                            ContentType="text/plain; charset=utf-8",
                            Metadata={"original-key": key}
                        )

                        row["status"] = (
                            "cleaned" if info["changed"] else "trusted"
                        )

                        counts["accepted"] += 1

                        if info["changed"]:
                            counts["cleaned"] += 1

            print(row["status"].upper(), key)

        except UnicodeDecodeError as error:
            row["status"] = "excluded"
            row["reason"] = f"Invalid UTF-8: {error}"
            counts["invalid"] += 1
            print("INVALID TEXT:", key, error)

        except Exception as error:
            row["status"] = "error"
            row["reason"] = str(error)
            counts["errors"] += 1
            print("ERROR:", key, error)

        rows.append(row)

    fields = [
        "key",
        "species",
        "status",
        "reason",
        "original_chars",
        "cleaned_chars",
        "original_words",
        "cleaned_words",
        "changed",
        "short_text"
    ]

    with open(
        REPORT_FILE,
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

    print("Total inspected:", len(rows))
    print("Accepted:", counts["accepted"])
    print("Cleaned:", counts["cleaned"])
    print("Empty excluded:", counts["empty"])
    print("Duplicates excluded:", counts["duplicates"])
    print("Invalid UTF-8:", counts["invalid"])
    print("Processing errors:", counts["errors"])
    print("Short texts flagged:", counts["short"])
    print("Already trusted:", counts["skipped"])
    print("Report:", REPORT_FILE)


if __name__ == "__main__":
    main()
