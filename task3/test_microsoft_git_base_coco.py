"""Task 3: test Microsoft GIT by generating descriptions for three images."""

import argparse
from io import BytesIO
import os
from pathlib import Path
import random

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_ID = "microsoft/git-base-coco"


def select_images(s3, bucket, seed):
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix="huggingface_cub200/images/"
    ):
        keys.extend(
            item["Key"] for item in page.get("Contents", [])
            if item["Key"].lower().endswith((".jpg", ".jpeg", ".png"))
        )
    if len(keys) < 3:
        raise RuntimeError(f"Expected at least 3 Trusted Zone images; found {len(keys)}")

    # Sort the images so the same seed selects the same three each time
    return random.Random(seed).sample(sorted(keys), 3)


def generate_captions(s3, bucket, selected):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    model.eval()

    for index, key in enumerate(selected, 1):
        body = s3.get_object(Bucket=bucket, Key=key)["Body"]
        try:
            raw = body.read()
        finally:
            body.close()
        with Image.open(BytesIO(raw)) as source:
            image = source.convert("RGB")
        # GIT generates a caption from the image alone (no question or retrieved context is supplied)
        inputs = processor(images=image, return_tensors="pt").to(device)
        # Generate the description without random sampling.
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=50, do_sample=False, num_beams=1)
        caption = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        if not caption:
            raise RuntimeError(f"Empty caption for s3://{bucket}/{key}")
        print(f"\nResult {index}", flush=True)
        print(f"Image: s3://{bucket}/{key}", flush=True)
        print(f"Description: {caption}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    load_dotenv(PROJECT_DIR / ".env")
    bucket = os.getenv("TRUSTED_BUCKET", "trustedzone")
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        config=Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 1}),
    )
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    selected = select_images(s3, bucket, seed)
    print(f"Selection seed: {seed}", flush=True)
    for index, key in enumerate(selected, 1):
        print(f"Image {index}: s3://{bucket}/{key}", flush=True)
    generate_captions(s3, bucket, selected)


if __name__ == "__main__":
    main()
