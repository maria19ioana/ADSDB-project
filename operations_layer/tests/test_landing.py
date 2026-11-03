from unittest.mock import MagicMock
import pytest
from landing_zone import persistent_landing as persistent


# TEST 1: Identify the correct source and data format
@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("temporal_landing/images/001.Bird/a.jpg", ("huggingface_cub200", "images")),
        ("temporal_landing/text/001.Bird/a.txt", ("huggingface_cub200", "text")),
        ("temporal_landing/audio/001.Bird/a.mp3", ("xeno_canto", "audio")),
        ("temporal_landing/unknown/a.txt", (None, None)),
        ("other/images/a.jpg", (None, None)),
    ],
)
def test_get_source_and_format(key, expected):
    assert persistent.get_source_and_format(key) == expected


# TEST 2: Generate correct destination path for images
def test_make_destination_image():
    key = "temporal_landing/images/031.Species/photo.jpg"
    timestamp = "20261102T210000Z"

    expected = (
        "persistent_landing/huggingface_cub200/images/031.Species/"
        "huggingface_cub200_20261102T210000Z_photo.jpg"
    )

    assert persistent.make_destination(key, timestamp) == expected


# TEST 3: Generate correct destination path for audio
def test_make_destination_audio():
    key = "temporal_landing/audio/031.Species/XC123.mp3"

    expected = (
        "persistent_landing/xeno_canto/audio/031.Species/"
        "xeno_canto_20261102T210000Z_XC123.mp3"
    )

    assert persistent.make_destination(key, "20261102T210000Z") == expected


# TEST 4: Skip invalid paths and folder markers
@pytest.mark.parametrize(
    "key",
    [
        "temporal_landing/images/",
        "temporal_landing/text/031.Species/",
        "temporal_landing/unrecognized/file.bin",
    ],
)
def test_make_destination_skips_invalid_or_folder_keys( key):
    assert persistent.make_destination(key, "20261102T210000Z") is None


# TEST 5: Recover the original key from a persistent destination
def test_destination_round_trip():
    original = "temporal_landing/text/031.Species/description.txt"

    destination = persistent.make_destination(
        original, "20261102T210000Z"
    )

    assert persistent.original_key_from_destination(destination) == original


# TEST 6: Successfully copy a file and verify its size
def test_main_copies_and_verifies_size(monkeypatch):
    source = "temporal_landing/images/031.Species/a.jpg"

    fake_s3 = MagicMock()
    fake_s3.head_object.return_value = {"ContentLength": 123}

    monkeypatch.setattr(persistent, "s3", fake_s3)

    monkeypatch.setattr(
        persistent,
        "get_existing_destinations",
        lambda: set()
    )

    monkeypatch.setattr(
        persistent,
        "list_temporal_files",
        lambda: iter([{"Key": source, "Size": 123}])
    )

    persistent.main()

    fake_s3.copy_object.assert_called_once()
    fake_s3.head_object.assert_called_once()

    assert fake_s3.copy_object.call_args.kwargs["CopySource"] == {
        "Bucket": "landingzone",
        "Key": source
    }


# TEST 7: Avoid copying files that already exist
def test_main_does_not_recopy_existing_file(monkeypatch):
    source = "temporal_landing/images/031.Species/a.jpg"

    existing = persistent.make_destination(
        source, "20261102T210000Z"
    )

    fake_s3 = MagicMock()

    monkeypatch.setattr(persistent, "s3", fake_s3)

    monkeypatch.setattr(
        persistent,
        "get_existing_destinations",
        lambda: {existing}
    )

    monkeypatch.setattr(
        persistent,
        "list_temporal_files",
        lambda: iter([{"Key": source, "Size": 123}])
    )

    persistent.main()

    fake_s3.copy_object.assert_not_called()


# TEST 8: Detect a file size mismatch after copying
def test_main_reports_failed_size_validation(monkeypatch, capsys):
    source = "temporal_landing/images/031.Species/a.jpg"

    fake_s3 = MagicMock()
    fake_s3.head_object.return_value = {"ContentLength": 5}

    monkeypatch.setattr(persistent, "s3", fake_s3)

    monkeypatch.setattr(
        persistent,
        "get_existing_destinations",
        lambda: set()
    )

    monkeypatch.setattr(
        persistent,
        "list_temporal_files",
        lambda: iter([{"Key": source, "Size": 123}])
    )

    with pytest.raises(RuntimeError):
        persistent.main()

    output = capsys.readouterr().out

    assert "Copied file size mismatch" in output
    assert "Failed: 1" in output
