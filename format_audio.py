
import json
import shutil
import subprocess
import tempfile
from pathlib import PurePosixPath
from pathlib import Path

from formatted_zone import (
    s3, SOURCE_BUCKET, TARGET_BUCKET,
    PERSISTENT_PREFIX, ensure_bucket,
    list_files, target_key, exists
)

PREFIX = PERSISTENT_PREFIX + "xeno_canto/audio/"

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg",
    ".m4a", ".aac", ".opus"
}


def main():
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not installed"
        )

    ensure_bucket()
    converted = skipped = failed = 0

    keys = sorted(list_files(PREFIX), key=lambda key: key.endswith(".json"))
    for key in keys:
        extension = PurePosixPath(key).suffix.lower()

        # keep metadata JSON files in the Formatted Zone, and convert audio files to WAV
        if extension == ".json":
            destination = key[len(PERSISTENT_PREFIX):]

            try:
                response = s3.get_object(
                    Bucket=SOURCE_BUCKET, Key=key
                )
                raw = response["Body"].read()

                # check that the JSON is valid before uploading
                metadata = json.loads(raw.decode("utf-8"))
                audio_key = str(PurePosixPath(destination).with_suffix(".wav"))
                if not exists(audio_key):
                    raise ValueError("The associated WAV is missing from Formatted Zone")

                metadata.pop("minio_audio_key", None)
                metadata["audio_bucket"] = TARGET_BUCKET
                metadata["audio_key"] = audio_key

                if exists(destination):
                    response = s3.get_object(Bucket=TARGET_BUCKET, Key=destination)
                    existing = json.loads(response["Body"].read().decode("utf-8"))
                    if existing == metadata:
                        skipped += 1
                        continue

                s3.put_object(
                    Bucket=TARGET_BUCKET,
                    Key=destination,
                    Body=json.dumps(metadata, indent=2).encode("utf-8"),
                    ContentType="application/json",
                    Metadata={"original-key": key}
                )

                print("METADATA:", destination)
                converted += 1

            except Exception as error:
                print("ERROR:", key, error)
                failed += 1

            continue

        if extension not in AUDIO_EXTENSIONS:
            continue

        destination = target_key(key, ".wav")

        if exists(destination):
            skipped += 1
            continue

        try:
            with tempfile.TemporaryDirectory() as folder:
                input_file = Path(folder) / ("input" + extension)
                output_file = Path(folder) / "output.wav"

                s3.download_file(
                    SOURCE_BUCKET, key, str(input_file)
                )

                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel", "error",
                        "-y",
                        "-i", str(input_file),
                        "-map", "0:a:0",
                        "-vn",
                        "-ac", "1",
                        "-ar", "16000",
                        "-c:a", "pcm_s16le",
                        str(output_file)
                    ],
                    check=True,
                    timeout=600
                )

                s3.upload_file(
                    str(output_file),
                    TARGET_BUCKET,
                    destination,
                    ExtraArgs={
                        "ContentType": "audio/wav",
                        "Metadata": {"original-key": key}
                    }
                )

            print("WAV:", destination)
            converted += 1

        except Exception as error:
            print("ERROR:", key, error)
            failed += 1

    print(
        f"\nAudio and metadata: converted={converted}, "
        f"skipped={skipped}, failed={failed}"
    )


if __name__ == "__main__":
    main()
