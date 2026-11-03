import sys
import wave
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image


# Import Trusted Zone scripts
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "trusted_zone"))

import trust_text
import trust_images
import trust_audio


# Helpers: Create test data in memory
def make_image(image_format="JPEG", mode="RGB"):
    image = Image.new(mode, (20, 10), "red")
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def make_wav(sample_rate=16000, channels=1, samples=None):
    if samples is None:
        samples = np.zeros(sample_rate, dtype=np.int16)

    samples = np.asarray(samples, dtype=np.int16)

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())

    return buffer.getvalue()


def mock_object(monkeypatch, module, raw):
    fake_s3 = MagicMock()
    fake_s3.get_object.return_value = {
        "Body": BytesIO(raw)
    }

    monkeypatch.setattr(module, "s3", fake_s3)
    return fake_s3


# TEST 1: Clean text and normalize whitespace
@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("  Bird   song  ", "Bird song"),
        ("First\r\nSecond", "First\nSecond"),
        ("First\n\n\nSecond", "First\n\nSecond"),
        ("Bird\t\t song", "Bird song"),
        ("Bird\x00song", "Birdsong"),
        ("", ""),
    ],
)
def test_clean_text(original, expected):
    assert trust_text.clean_text(original) == expected


# TEST 2: Validate a correct JPEG image
def test_inspect_image_valid_jpeg(monkeypatch):
    raw = make_image()

    mock_object(monkeypatch, trust_images, raw)

    result = trust_images.inspect_image("bird.jpg")

    assert result["width"] == 20
    assert result["height"] == 10
    assert result["aspect_ratio"] == 2.0
    assert result["size_bytes"] == len(raw)
    assert len(result["pixel_hash"]) == 64


# TEST 3: Reject unsupported image format
def test_inspect_image_rejects_png(monkeypatch):
    raw = make_image(image_format="PNG")

    mock_object(monkeypatch, trust_images, raw)

    with pytest.raises(ValueError, match="Expected JPEG"):
        trust_images.inspect_image("bird.png")


# TEST 4: Reject grayscale images
def test_inspect_image_rejects_grayscale(monkeypatch):
    raw = make_image(mode="L")

    mock_object(monkeypatch, trust_images, raw)

    with pytest.raises(ValueError, match="Expected RGB"):
        trust_images.inspect_image("gray.jpg")


# TEST 5: Verify identical images have the same hash
def test_inspect_image_same_pixels_same_hash(monkeypatch):
    raw = make_image()

    fake_s3 = MagicMock()

    # Return a fresh BytesIO for each request
    fake_s3.get_object.side_effect = lambda **kwargs: {
        "Body": BytesIO(raw)
    }

    monkeypatch.setattr(trust_images, "s3", fake_s3)

    first = trust_images.inspect_image("bird1.jpg")
    second = trust_images.inspect_image("bird2.jpg")

    assert first["pixel_hash"] == second["pixel_hash"]
    assert fake_s3.get_object.call_count == 2


# TEST 6: Validate a correct WAV audio file
def test_inspect_audio_valid_wav(monkeypatch):
    raw = make_wav()

    mock_object(monkeypatch, trust_audio, raw)

    result = trust_audio.inspect_audio("bird.wav")

    assert result["sample_rate"] == 16000
    assert result["channels"] == 1
    assert result["bit_depth"] == 16
    assert result["duration_seconds"] == 1.0
    assert result["short_audio"] is False
    assert len(result["audio_hash"]) == 64


# TEST 7: Reject incorrect audio sample rate
def test_inspect_audio_wrong_sample_rate(monkeypatch):
    raw = make_wav(sample_rate=8000)

    mock_object(monkeypatch, trust_audio, raw)

    with pytest.raises(ValueError, match="Expected 16000 Hz"):
        trust_audio.inspect_audio("wrong_rate.wav")


# TEST 8: Reject stereo audio
def test_inspect_audio_rejects_stereo(monkeypatch):
    samples = np.zeros(32000, dtype=np.int16)
    raw = make_wav(channels=2, samples=samples)

    mock_object(monkeypatch, trust_audio, raw)

    with pytest.raises(ValueError, match="Expected mono"):
        trust_audio.inspect_audio("stereo.wav")