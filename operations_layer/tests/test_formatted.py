import sys
from pathlib import Path
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "formatted_zone"))

import formatted_zone as formatted
import format_text
import format_images


# TEST 1: Generate correct destination paths
@pytest.mark.parametrize(
    ("source", "extension", "expected"),
    [
        (
            "persistent_landing/huggingface_cub200/images/031.Bird/photo.png",
            ".jpg",
            "huggingface_cub200/images/031.Bird/photo.jpg",
        ),
        (
            "persistent_landing/huggingface_cub200/text/031.Bird/info.json",
            ".txt",
            "huggingface_cub200/text/031.Bird/info.txt",
        ),
        (
            "persistent_landing/xeno_canto/audio/031.Bird/XC123.mp3",
            ".wav",
            "xeno_canto/audio/031.Bird/XC123.wav",
        ),
    ],
)
def test_target_key(source, extension, expected):
    assert formatted.target_key(source, extension) == expected


# TEST 2: Extract text from supported formats
@pytest.mark.parametrize(
    ("raw", "extension", "expected"),
    [
        (b"Hello bird", ".txt", "Hello bird"),
        (b'{"description": "Small bird"}', ".json", "Small bird"),
        (b'{"text": "Bird song"}', ".json", "Bird song"),
        (
            b'{"sentences": ["First", "Second"]}',
            ".json",
            "First\nSecond",
        ),
        (b'"Bird description"', ".json", "Bird description"),
    ],
)
def test_extract_text(raw, extension, expected):
    assert format_text.extract_text(raw, extension) == expected


# TEST 3: Reject unsupported JSON structure
def test_extract_text_invalid_json_structure():
    raw = b'{"unknown": 123}'

    with pytest.raises(ValueError, match="No supported text field"):
        format_text.extract_text(raw, ".json")


# TEST 4: Normalize whitespace and line endings
def test_normalize_text():
    original = "  Bird song  \r\n\r\n\r\nSmall bird   \r\n"

    expected = "Bird song\n\nSmall bird\n"

    assert format_text.normalize_text(original) == expected


# TEST 5: Convert transparent PNG to JPEG
def test_format_image_converts_png(monkeypatch):
    fake_s3 = MagicMock()

    source_image = Image.new("RGBA", (20, 20), (255, 0, 0, 128))
    buffer = BytesIO()
    source_image.save(buffer, format="PNG")

    fake_s3.get_object.return_value = {
        "Body": BytesIO(buffer.getvalue())
    }

    monkeypatch.setattr(format_images, "s3", fake_s3)
    monkeypatch.setattr(format_images, "exists", lambda key: False)

    result = format_images.format_image(
        "persistent_landing/huggingface_cub200/images/031.Bird/photo.png"
    )

    assert result == "converted"

    fake_s3.put_object.assert_called_once()

    uploaded = fake_s3.put_object.call_args.kwargs

    assert uploaded["ContentType"] == "image/jpeg"

    with Image.open(BytesIO(uploaded["Body"])) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (20, 20)


# TEST 6: Skip image if it already exists
def test_format_image_skips_existing(monkeypatch):
    fake_s3 = MagicMock()

    monkeypatch.setattr(format_images, "s3", fake_s3)
    monkeypatch.setattr(format_images, "exists", lambda key: True)

    result = format_images.format_image(
        "persistent_landing/huggingface_cub200/images/031.Bird/photo.jpg"
    )

    assert result == "skipped"

    fake_s3.get_object.assert_not_called()
    fake_s3.put_object.assert_not_called()
    fake_s3.copy_object.assert_not_called()


# TEST 7: Copy a valid RGB JPEG without re-encoding
def test_format_image_copies_valid_jpeg(monkeypatch):
    fake_s3 = MagicMock()

    image = Image.new("RGB", (20, 20), "blue")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    fake_s3.get_object.return_value = {
        "Body": BytesIO(buffer.getvalue())
    }

    monkeypatch.setattr(format_images, "s3", fake_s3)
    monkeypatch.setattr(format_images, "exists", lambda key: False)

    result = format_images.format_image(
        "persistent_landing/huggingface_cub200/images/031.Bird/photo.jpg"
    )

    assert result == "copied"

    fake_s3.copy_object.assert_called_once()
    fake_s3.put_object.assert_not_called()
