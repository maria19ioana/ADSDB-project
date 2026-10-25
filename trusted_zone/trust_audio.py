
import csv
import hashlib
import json
import wave
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np

from trusted_zone.trusted_zone import (
    s3,
    SOURCE_BUCKET,
    TARGET_BUCKET,
    ensure_bucket,
    list_files,
    copy_to_trusted,
    species_from_key,
    exists
)

PREFIX = "xeno_canto/audio/"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = PROJECT_ROOT / "quality_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = REPORT_DIR / "audio_quality.csv"

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH = 2

MIN_DURATION_SECONDS = 1.0
SILENCE_THRESHOLD = 0.01
MOSTLY_SILENT_RATIO = 0.90
CLIPPING_THRESHOLD = 0.999
CLIPPING_RATIO_WARNING = 0.01


def inspect_audio(key):
    """Validate WAV and calculate quality metrics"""

    response = s3.get_object(
        Bucket=SOURCE_BUCKET,
        Key=key
    )
    raw = response["Body"].read()

    with wave.open(BytesIO(raw), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()
        compression = wav.getcomptype()

        if channels != EXPECTED_CHANNELS:
            raise ValueError(
                f"Expected mono, found {channels} channels"
            )

        if sample_rate != EXPECTED_SAMPLE_RATE:
            raise ValueError(
                f"Expected 16000 Hz, found {sample_rate} Hz"
            )

        if sample_width != EXPECTED_SAMPLE_WIDTH:
            raise ValueError(
                f"Expected 16-bit PCM, found "
                f"{sample_width * 8}-bit samples"
            )

        if compression != "NONE":
            raise ValueError("Expected uncompressed PCM")

        if frame_count <= 0:
            raise ValueError("Audio contains no samples")

        frames = wav.readframes(frame_count)

        expected_bytes = (
            frame_count * channels * sample_width
        )

        if len(frames) != expected_bytes:
            raise ValueError(
                "Audio data is truncated or incomplete"
            )

    samples = np.frombuffer(
        frames,
        dtype="<i2"
    ).astype(np.float64) / 32768.0

    if samples.size == 0:
        raise ValueError("Audio contains no samples")

    duration = frame_count / sample_rate

    # amplitude metrics
    rms = float(np.sqrt(np.mean(samples ** 2)))

    peak = float(np.max(np.abs(samples)))

    # calculate the fraction of samples that are near zero amplitude
    silence_ratio = float(
        np.mean(np.abs(samples) < SILENCE_THRESHOLD)
    )

    # calculate the fraction of samples that are at or above the clipping threshold
    clipping_ratio = float(
        np.mean(np.abs(samples) >= CLIPPING_THRESHOLD)
    )

    # duplicate detection
    audio_hash = hashlib.sha256(frames).hexdigest()

    return {
        "duration_seconds": round(duration, 3),
        "sample_rate": sample_rate,
        "channels": channels,
        "bit_depth": sample_width * 8,
        "size_bytes": len(raw),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "silence_ratio": round(silence_ratio, 4),
        "clipping_ratio": round(clipping_ratio, 4),
        "audio_hash": audio_hash,
        "short_audio": duration < MIN_DURATION_SECONDS,
        "mostly_silent": silence_ratio >= MOSTLY_SILENT_RATIO,
        "clipping_warning": (
            clipping_ratio >= CLIPPING_RATIO_WARNING
        )
    }


def copy_metadata(audio_key, available_keys):
    """Preserve the JSON attribution file for accepted audio"""

    json_key = str(
        Path(audio_key).with_suffix(".json")
    ).replace("\\", "/")

    if json_key not in available_keys:
        return "missing"

    if exists(json_key):
        return "already_trusted"

    try:
        response = s3.get_object(
            Bucket=SOURCE_BUCKET,
            Key=json_key
        )

        metadata_bytes = response["Body"].read()

        # check that the JSON is valid and can be decoded
        metadata = json.loads(metadata_bytes.decode("utf-8-sig"))
        s3.head_object(Bucket=TARGET_BUCKET, Key=audio_key)
        metadata.pop("minio_audio_key", None)
        metadata["audio_bucket"] = TARGET_BUCKET
        metadata["audio_key"] = audio_key

        s3.put_object(
            Bucket=TARGET_BUCKET,
            Key=json_key,
            Body=json.dumps(metadata, indent=2).encode("utf-8"),
            ContentType="application/json"
        )

        return "copied"

    except (ValueError, UnicodeDecodeError):
        return "invalid"

    except Exception as error:
        print("METADATA ERROR:", json_key, error)
        return "error"


def main():
    ensure_bucket()

    keys = set(list_files(PREFIX))

    audio_keys = sorted(
        key for key in keys
        if key.lower().endswith(".wav")
    )

    counts = Counter()
    seen_hashes = {}
    rows = []

    for key in audio_keys:
        species = species_from_key(key)

        row = {
            "key": key,
            "species": species,
            "status": "",
            "reason": "",
            "duration_seconds": "",
            "sample_rate": "",
            "channels": "",
            "bit_depth": "",
            "size_bytes": "",
            "rms": "",
            "peak": "",
            "silence_ratio": "",
            "clipping_ratio": "",
            "audio_hash": "",
            "short_audio": "",
            "mostly_silent": "",
            "clipping_warning": "",
            "metadata_status": ""
        }

        try:
            metrics = inspect_audio(key)
            row.update(metrics)

            audio_hash = metrics["audio_hash"]

            if audio_hash in seen_hashes:
                row["status"] = "excluded"
                row["reason"] = (
                    "Exact duplicate of "
                    + seen_hashes[audio_hash]
                )
                counts["duplicates"] += 1

            else:
                seen_hashes[audio_hash] = key

                if exists(key):
                    row["status"] = "already_trusted"
                    counts["skipped"] += 1

                else:
                    copy_to_trusted(key)

                    row["status"] = "trusted"
                    counts["trusted"] += 1

                # Check JSON metadata separately.
                # This also handles WAV files that already existed.
                row["metadata_status"] = copy_metadata(
                    key, keys
                )

                if row["metadata_status"] == "copied":
                    counts["metadata_copied"] += 1
                elif row["metadata_status"] == "already_trusted":
                    counts["metadata_skipped"] += 1
                else:
                    counts["metadata_issues"] += 1

                if metrics["short_audio"]:
                    counts["short"] += 1

                if metrics["mostly_silent"]:
                    counts["mostly_silent"] += 1

                if metrics["clipping_warning"]:
                    counts["clipping"] += 1

            print(row["status"].upper(), key)

        except (wave.Error, EOFError, ValueError) as error:
            row["status"] = "excluded"
            row["reason"] = str(error)
            counts["invalid"] += 1
            print("INVALID AUDIO:", key, error)

        except Exception as error:
            row["status"] = "error"
            row["reason"] = str(error)
            counts["errors"] += 1
            print("ERROR:", key, error)

        rows.append(row)

    fields = [
        "key", "species", "status", "reason",
        "duration_seconds", "sample_rate",
        "channels", "bit_depth", "size_bytes",
        "rms", "peak", "silence_ratio",
        "clipping_ratio", "audio_hash",
        "short_audio", "mostly_silent",
        "clipping_warning", "metadata_status"
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

    print("Total WAV inspected:", len(rows))
    print("Trusted:", counts["trusted"])
    print("Duplicates excluded:", counts["duplicates"])
    print("Invalid excluded:", counts["invalid"])
    print("Processing errors:", counts["errors"])
    print("Short recordings flagged:", counts["short"])
    print("Mostly silent flagged:", counts["mostly_silent"])
    print("Clipping flagged:", counts["clipping"])
    print("JSON metadata copied:", counts["metadata_copied"])
    print("Metadata issues:", counts["metadata_issues"])
    print("Already trusted WAV:", counts["skipped"])
    print("JSON metadata already trusted:", counts["metadata_skipped"])
    print("Report:", REPORT_FILE)


if __name__ == "__main__":
    main()
