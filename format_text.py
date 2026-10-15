
import json
import re
from pathlib import PurePosixPath

from formatted_zone import (
    s3, SOURCE_BUCKET, TARGET_BUCKET,
    PERSISTENT_PREFIX, ensure_bucket,
    list_files, target_key, exists
)

PREFIX = PERSISTENT_PREFIX + "huggingface_cub200/text/"


def extract_text(raw, extension):
    if extension == ".json":
        data = json.loads(raw.decode("utf-8-sig"))

        if isinstance(data, str):
            return data

        if isinstance(data, dict):
            for field in (
                "text", "description", "caption", "sentences"
            ):
                value = data.get(field)

                if isinstance(value, str):
                    return value

                if isinstance(value, list):
                    if all(isinstance(x, str) for x in value):
                        return "\n".join(value)

        raise ValueError("No supported text field in JSON")

    # check for common text encodings
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeError:
            continue

    raise ValueError("Unsupported text encoding")


def normalize_text(text):
    # replace endlines with \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # ensure that each line is stripped of trailing whitespace
    lines = [line.rstrip() for line in text.split("\n")]

    # remove any extra blank lines at the start and end of the text
    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip() + "\n"


def main():
    ensure_bucket()
    converted = skipped = failed = 0

    for key in list_files(PREFIX):
        extension = PurePosixPath(key).suffix.lower()

        if extension not in (".txt", ".json", ".csv", ".md", ".log"):
            continue

        destination = target_key(key, ".txt")

        if exists(destination):
            skipped += 1
            continue

        try:
            response = s3.get_object(
                Bucket=SOURCE_BUCKET, Key=key
            )

            raw = response["Body"].read()
            text = extract_text(raw, extension)
            text = normalize_text(text)

            s3.put_object(
                Bucket=TARGET_BUCKET,
                Key=destination,
                Body=text.encode("utf-8"),
                ContentType="text/plain; charset=utf-8",
                Metadata={"original-key": key}
            )

            print("TXT:", destination)
            converted += 1

        except Exception as error:
            print("ERROR:", key, error)
            failed += 1

    print(
        f"\nText: converted={converted}, "
        f"skipped={skipped}, failed={failed}"
    )


if __name__ == "__main__":
    main()
