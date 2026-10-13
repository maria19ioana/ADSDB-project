from io import BytesIO
from pathlib import PurePosixPath

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from formatted_zone import (
    s3,
    SOURCE_BUCKET,
    TARGET_BUCKET,
    PERSISTENT_PREFIX,
    ensure_bucket,
    list_files,
    target_key,
    exists
)

register_heif_opener()

# Images in Persistent Landing
PREFIX = PERSISTENT_PREFIX + "huggingface_cub200/images/"

SUPPORTED = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".tif", ".tiff", ".heic", ".heif"
}


def format_image(key):
    """Convert an image to the canonical RGB JPG format."""

    destination = target_key(key, ".jpg")

    if exists(destination):
        print("SKIP:", destination)
        return "skipped"

    # Download source image from Persistent Landing.
    response = s3.get_object(
        Bucket=SOURCE_BUCKET,
        Key=key
    )
    original = response["Body"].read()

    with Image.open(BytesIO(original)) as image:
        # Apply EXIF rotation before saving.
        image = ImageOps.exif_transpose(image)

        # Existing RGB JPEGs can be copied without re-encoding,
        # provided they have no EXIF orientation adjustment.
        orientation = image.getexif().get(274, 1)

        can_copy = (
            PurePosixPath(key).suffix.lower() == ".jpg"
            and image.mode == "RGB"
            and orientation == 1
        )

        if can_copy:
            s3.copy_object(
                Bucket=TARGET_BUCKET,
                CopySource={
                    "Bucket": SOURCE_BUCKET,
                    "Key": key
                },
                Key=destination,
                MetadataDirective="REPLACE",
                Metadata={"original-key": key},
                ContentType="image/jpeg"
            )

            print("COPIED JPG:", destination)
            return "copied"

        # Convert images that are not already compliant.
        # Transparency is placed over a white background.
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")

        output = BytesIO()
        image.save(
            output,
            format="JPEG",
            quality=95,
            subsampling=0
        )
        output.seek(0)

        s3.put_object(
            Bucket=TARGET_BUCKET,
            Key=destination,
            Body=output.getvalue(),
            ContentType="image/jpeg",
            Metadata={"original-key": key}
        )

        print("CONVERTED JPG:", destination)
        return "converted"


def main():
    ensure_bucket()

    copied = 0
    converted = 0
    skipped = 0
    failed = 0

    for key in list_files(PREFIX):
        extension = PurePosixPath(key).suffix.lower()

        if extension not in SUPPORTED:
            continue

        try:
            result = format_image(key)

            if result == "copied":
                copied += 1
            elif result == "converted":
                converted += 1
            elif result == "skipped":
                skipped += 1

        except Exception as error:
            print("ERROR:", key, error)
            failed += 1

    print("\n========== IMAGE FORMATTING ==========")
    print("Copied JPG:", copied)
    print("Converted to JPG:", converted)
    print("Already formatted:", skipped)
    print("Failed:", failed)
    print("======================================")


if __name__ == "__main__":
    main()
