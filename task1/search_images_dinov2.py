
import os
import hashlib
from io import BytesIO
from pathlib import Path
import sys

import boto3
import chromadb
import torch
import torch.nn.functional as F
from PIL import Image
from dotenv import load_dotenv
from transformers import AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt

# Configuration
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

BUCKET = os.getenv("TRUSTED_BUCKET", "trustedzone")

MODEL_NAME = "facebook/dinov2-base"
COLLECTION_NAME = "bird_images_dinov2"
CHROMA_PATH = PROJECT_DIR / "chroma_data"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# One image key from the ones already embedded
QUERY_IMAGE_KEY = (
    "huggingface_cub200/images/031.Black_billed_Cuckoo/"
    "huggingface_cub200_20261015T094354Z_Black_Billed_Cuckoo_0001_26242.jpg"
)

# Connect to MinIO
s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin")
)

# Connect to ChromaDB
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection(name=COLLECTION_NAME)


def load_image_from_minio(key):
    response = s3.get_object(Bucket=BUCKET, Key=key)
    image_bytes = response["Body"].read()
    return Image.open(BytesIO(image_bytes)).convert("RGB")


# Load a query image from a local file or MinIO
def load_query_image():

    if len(sys.argv) > 1:
        image_path = sys.argv[1]

        print("Loading local image:", image_path)

        with Image.open(image_path) as img:
            return img.convert("RGB").copy(), None

    print("Loading query image from MinIO...")
    image = load_image_from_minio(QUERY_IMAGE_KEY)

    return image, QUERY_IMAGE_KEY


# Generate a normalized DINOv2 image embedding
def generate_embedding(image, model, processor):

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

        # Use the CLS token as the image representation
        embedding = outputs.last_hidden_state[:, 0, :]

        embedding = F.normalize(embedding, dim=-1)

    return embedding.squeeze(0).cpu().tolist()


# Display query image and top 5 similar images
def visualize_results(query_image, results):

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    # Display query image
    axes[0].imshow(query_image)
    axes[0].set_title("Query Image", fontsize=12)
    axes[0].axis("off")

    # Display retrieved images
    for i, (metadata, distance) in enumerate(results[:5], start=1):
        image = load_image_from_minio(metadata["minio_key"])

        species = metadata["species"].split(".", 1)[-1]
        species = species.replace("_", " ")

        similarity = 1 - distance

        axes[i].imshow(image)
        axes[i].set_title(
            f"Rank {i}: {species}\n"
            f"Similarity: {similarity:.3f}",
            fontsize=10
        )
        axes[i].axis("off")

    # Hide unused axes
    for ax in axes[len(results) + 1:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("similarity_results_dinov2.png", dpi=150)
    plt.show()


def main():

    print("Loading DINOv2 model...")
    print(f"Using device: {DEVICE}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    image, query_key = load_query_image()

    query_embedding = generate_embedding(image, model, processor)

    print("Searching similar images...")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(6, collection.count()),
        include=["metadatas", "distances"]
    )

    query_id = (
        hashlib.sha256(query_key.encode("utf-8")).hexdigest()
        if query_key is not None
        else None
    )

    print("\nTop similar images:\n")

    similar_images = []

    for result_id, metadata, distance in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        if result_id == query_id:
            continue

        similar_images.append((metadata, distance))

        print(f"Rank {len(similar_images)}")
        print("Species:", metadata["species"])
        print("Distance:", round(distance, 4))
        print("-" * 50)

        if len(similar_images) == 5:
            break

    visualize_results(image, similar_images)


if __name__ == "__main__":
    main()
